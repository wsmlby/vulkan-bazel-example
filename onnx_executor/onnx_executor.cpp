#include "onnx_executor.hpp"
#include "operators/operator_registry.hpp"
#include "operators/register_all.hpp"
#include <iostream>
#include <filesystem>
#include <fstream>
#include <set>
#include <algorithm>

namespace fs = std::filesystem;

namespace onnxrt {

Executor::Executor(vkcompute::Context& ctx, const std::string& shaderDir)
    : ctx_(&ctx), shaderDir_(shaderDir) {
    // Register all operators
    registerAllOperators();

    if (shaderDir_.empty()) {
        shaderDir_ = findShaderDir();
    }
}

Executor::~Executor() = default;

std::string Executor::findShaderDir() const {
    // Try common shader locations
    std::vector<std::string> paths = {
        "onnx_executor",  // Local build
        "bazel-bin/onnx_executor",  // Bazel build output
    };

    // Try RUNFILES_DIR
    const char* runfiles = std::getenv("RUNFILES_DIR");
    if (runfiles) {
        paths.insert(paths.begin(), std::string(runfiles) + "/gpucompute/onnx_executor");
    }

    // Try relative to executable
    try {
        fs::path exePath = fs::read_symlink("/proc/self/exe").parent_path();
        paths.push_back((exePath / "run_onnx.runfiles/gpucompute/onnx_executor").string());
        paths.push_back((exePath / "onnx_executor").string());
    } catch (...) {}

    for (const auto& path : paths) {
        if (fs::exists(path + "/add_shader.spv")) {
            std::cout << "Found shaders at: " << path << std::endl;
            return path;
        }
    }

    std::cout << "Warning: Shader directory not found, using current directory" << std::endl;
    return ".";
}

void Executor::loadModel(const std::string& onnxPath) {
    model_ = std::make_unique<Model>(Model::load(onnxPath, *ctx_));
    prepare();
}

std::vector<TensorInfo> Executor::getInputInfo() const {
    std::vector<TensorInfo> result;
    for (const auto& input : model_->inputs()) {
        result.push_back({input.name, input.shape, input.dtype});
    }
    return result;
}

std::vector<TensorInfo> Executor::getOutputInfo() const {
    std::vector<TensorInfo> result;
    for (const auto& output : model_->outputs()) {
        result.push_back({output.name, output.shape, output.dtype});
    }
    return result;
}

std::map<std::string, Tensor> Executor::createInputs() const {
    std::map<std::string, Tensor> inputs;
    for (const auto& info : model_->inputs()) {
        // Create pinned tensors for inputs (easy CPU access)
        inputs.emplace(info.name, Tensor(*ctx_, info.shape, info.dtype, true));
    }
    return inputs;
}

void Executor::prepare() {
    std::cout << "Preparing executor..." << std::endl;

    auto& registry = OperatorRegistry::instance();

    // ============================================================
    // SiLU Fusion Detection: Conv -> Sigmoid -> Mul
    // ============================================================
    std::set<std::string> fusedNodes;  // Nodes to skip (Sigmoid, Mul that are fused)
    std::set<std::string> siluConvOutputs;  // Conv outputs that should use SiLU activation
    
    // Build maps for pattern detection
    std::map<std::string, const Node*> nodeByOutput;  // tensor_name -> node that produces it
    std::map<std::string, const Node*> nodeByName;    // node_name -> node
    
    for (const auto& node : model_->nodes()) {
        nodeByName[node.name] = &node;
        for (const auto& out : node.outputs) {
            if (!out.empty()) {
                nodeByOutput[out] = &node;
            }
        }
    }
    
    // Find SiLU patterns: Conv -> Sigmoid -> Mul (where Mul takes Conv and Sigmoid outputs)
    for (const auto& node : model_->nodes()) {
        if (node.opType == "Mul") {
            // Check if this Mul is part of SiLU pattern
            const Node* sigmoidNode = nullptr;
            const Node* convNode = nullptr;
            std::string convOutput;
            std::string sigOutput;
            
            for (const auto& input : node.inputs) {
                auto it = nodeByOutput.find(input);
                if (it != nodeByOutput.end()) {
                    if (it->second->opType == "Sigmoid") {
                        sigmoidNode = it->second;
                        sigOutput = input;
                    } else if (it->second->opType == "Conv") {
                        convNode = it->second;
                        convOutput = input;
                    }
                }
            }
            
            // Check if Sigmoid's input is the Conv output
            if (sigmoidNode && convNode && !sigmoidNode->inputs.empty()) {
                if (sigmoidNode->inputs[0] == convOutput) {
                    // Found SiLU pattern!
                    fusedNodes.insert(sigmoidNode->name);
                    fusedNodes.insert(node.name);
                    siluConvOutputs.insert(convOutput);
                }
            }
        }
    }
    
    if (!siluConvOutputs.empty()) {
        std::cout << "SiLU fusion: found " << siluConvOutputs.size() << " Conv+SiLU patterns" << std::endl;
    }

    // Track which values have been produced
    std::set<std::string> producedValues;

    // Initializers are already produced
    for (const auto& [name, _] : model_->initializers()) {
        producedValues.insert(name);
    }

    // Create device-local tensors for graph inputs
    // These will be used as targets for copying from user-provided pinned tensors
    for (const auto& input : model_->inputs()) {
        if (!input.shape.empty()) {
            auto tensor = std::make_unique<Tensor>(*ctx_, input.shape, input.dtype, false);
            inputTensors_[input.name] = std::move(tensor);
            producedValues.insert(input.name);
            std::cout << "Created input tensor: " << input.name << " " << shapeStr(input.shape) << std::endl;
        }
    }

    // Helper to get tensor by name
    auto getTensor = [&](const std::string& name) -> Tensor* {
        auto initIt = model_->initializers().find(name);
        if (initIt != model_->initializers().end()) return &initIt->second;
        auto inputIt = inputTensors_.find(name);
        if (inputIt != inputTensors_.end()) return inputIt->second.get();
        auto intIt = intermediates_.find(name);
        if (intIt != intermediates_.end()) return intIt->second.get();
        return nullptr;
    };

    // Helper to read int64 values from a constant tensor
    auto readInt64Tensor = [&](const std::string& name) -> std::vector<int64_t> {
        Tensor* t = getTensor(name);
        if (!t) return {};
        
        // Copy to host
        std::vector<int64_t> result(t->elementCount());
        t->copyToHost(result.data(), result.size() * sizeof(int64_t));
        return result;
    };

    // Helper to compute Slice output shape
    auto computeSliceShape = [&](const Shape& inputShape, const Node& node) -> Shape {
        // Slice inputs: [data, starts, ends, axes (opt), steps (opt)]
        if (node.inputs.size() < 3) return inputShape;
        
        auto starts = readInt64Tensor(node.inputs[1]);
        auto ends = readInt64Tensor(node.inputs[2]);
        std::vector<int64_t> axes, steps;
        
        if (node.inputs.size() > 3 && !node.inputs[3].empty()) {
            axes = readInt64Tensor(node.inputs[3]);
        }
        if (node.inputs.size() > 4 && !node.inputs[4].empty()) {
            steps = readInt64Tensor(node.inputs[4]);
        }
        
        // Default axes: all axes in order
        if (axes.empty()) {
            for (size_t i = 0; i < starts.size(); i++) axes.push_back(i);
        }
        // Default steps: all 1s
        if (steps.empty()) {
            steps.resize(axes.size(), 1);
        }
        
        Shape output = inputShape;
        for (size_t i = 0; i < axes.size(); i++) {
            int64_t axis = axes[i];
            if (axis < 0) axis += inputShape.size();
            
            int64_t dimSize = inputShape[axis];
            int64_t start = starts[i];
            int64_t end = ends[i];
            int64_t step = steps[i];
            
            // Handle negative indices
            if (start < 0) start += dimSize;
            if (end < 0) end += dimSize;
            
            // Clamp to valid range
            start = std::max((int64_t)0, std::min(start, dimSize));
            end = std::max((int64_t)0, std::min(end, dimSize));
            
            // Handle INT_MAX for end
            if (ends[i] > dimSize) end = dimSize;
            
            output[axis] = (end - start + step - 1) / step;
            if (output[axis] < 0) output[axis] = 0;
        }
        
        return output;
    };

    // Create tensors for intermediate values
    for (const auto& node : model_->nodes()) {
        // Skip fused nodes (Sigmoid and Mul that are part of SiLU)
        if (fusedNodes.count(node.name) > 0) {
            // Mark outputs as produced (they'll be written by the fused Conv)
            for (const auto& outputName : node.outputs) {
                if (!outputName.empty()) {
                    producedValues.insert(outputName);
                }
            }
            continue;
        }
        
        // Check if this Conv uses SiLU fusion
        bool useSiluFusion = false;
        std::string siluOutputName;
        if (node.opType == "Conv" && !node.outputs.empty()) {
            if (siluConvOutputs.count(node.outputs[0]) > 0) {
                useSiluFusion = true;
                // Find the Mul node's output (which becomes our final output)
                for (const auto& mulNode : model_->nodes()) {
                    if (fusedNodes.count(mulNode.name) > 0 && mulNode.opType == "Mul") {
                        // Check if this Mul uses our Conv's Sigmoid output
                        for (const auto& input : mulNode.inputs) {
                            auto it = nodeByOutput.find(input);
                            if (it != nodeByOutput.end() && it->second->opType == "Sigmoid") {
                                if (!it->second->inputs.empty() && it->second->inputs[0] == node.outputs[0]) {
                                    siluOutputName = mulNode.outputs[0];
                                    break;
                                }
                            }
                        }
                    }
                    if (!siluOutputName.empty()) break;
                }
                if (useSiluFusion) {
                    // Debug: print fusion info but commented out
                    // std::cout << "  Fusing Conv " << node.name << " -> " << siluOutputName << std::endl;
                }
            }
        }
        
        // Check if operator is supported
        std::string opType = node.opType;
        if (useSiluFusion && opType == "Conv") {
            opType = "ConvSilu";  // Use fused operator
        }
        
        if (!registry.hasOp(opType)) {
            std::cout << "Warning: Unsupported operator: " << opType << std::endl;
            continue;
        }

        // Create operator instance
        auto op = registry.createOp(opType);
        if (!op) {
            std::cout << "Warning: Failed to create operator: " << opType << std::endl;
            continue;
        }

        // Gather input tensors
        std::vector<Tensor*> inputTensors;
        std::vector<Shape> inputShapes;
        bool allInputsAvailable = true;

        for (const auto& inputName : node.inputs) {
            if (inputName.empty()) {
                inputTensors.push_back(nullptr);
                inputShapes.push_back({});
                continue;
            }

            Tensor* tensor = getTensor(inputName);

            if (tensor) {
                inputTensors.push_back(tensor);
                inputShapes.push_back(tensor->shape());
            } else {
                std::cout << "  Missing input: " << inputName << " for " << node.opType << std::endl;
                allInputsAvailable = false;
                break;
            }
        }

        if (!allInputsAvailable) {
            continue;
        }

        // Infer output shapes - special handling for certain ops
        std::vector<Shape> outputShapes;
        
        if (node.opType == "Slice") {
            outputShapes = {computeSliceShape(inputShapes[0], node)};
        } else if (node.opType == "Reshape") {
            // Read shape from second input
            if (inputTensors.size() >= 2 && inputTensors[1]) {
                auto shapeData = readInt64Tensor(node.inputs[1]);
                // Handle -1 (infer dimension)
                int64_t total = 1;
                for (int64_t d : inputShapes[0]) total *= d;
                int64_t unknown = -1;
                int64_t known = 1;
                for (size_t i = 0; i < shapeData.size(); i++) {
                    if (shapeData[i] == -1) unknown = i;
                    else if (shapeData[i] == 0) shapeData[i] = inputShapes[0][i];
                    if (shapeData[i] > 0) known *= shapeData[i];
                }
                if (unknown >= 0) shapeData[unknown] = total / known;
                outputShapes = {shapeData};
            } else {
                outputShapes = op->inferShapes(inputShapes, node);
            }
        } else if (node.opType == "Resize") {
            // Read scales or sizes from inputs
            // Resize: [X, roi, scales, sizes]
            if (inputTensors.size() >= 4 && inputTensors[3]) {
                // sizes provided
                auto sizes = readInt64Tensor(node.inputs[3]);
                outputShapes = {sizes};
            } else if (inputTensors.size() >= 3 && inputTensors[2]) {
                // scales provided
                std::vector<float> scales(inputTensors[2]->elementCount());
                inputTensors[2]->copyToHost(scales.data(), scales.size() * sizeof(float));
                Shape out = inputShapes[0];
                for (size_t i = 0; i < out.size() && i < scales.size(); i++) {
                    out[i] = static_cast<int64_t>(out[i] * scales[i]);
                }
                outputShapes = {out};
            } else {
                outputShapes = op->inferShapes(inputShapes, node);
            }
        } else if (node.opType == "Split") {
            // Use the operator's shape inference
            outputShapes = op->inferShapes(inputShapes, node);
            // If split sizes come from input, read them
            if (outputShapes.empty() && inputTensors.size() >= 2 && inputTensors[1]) {
                auto splitSizes = readInt64Tensor(node.inputs[1]);
                int64_t axis = node.getAttr<int64_t>("axis", 0);
                if (axis < 0) axis += inputShapes[0].size();
                for (int64_t s : splitSizes) {
                    Shape out = inputShapes[0];
                    out[axis] = s;
                    outputShapes.push_back(out);
                }
            }
        } else {
            outputShapes = op->inferShapes(inputShapes, node);
        }

        // Create output tensors
        std::vector<Tensor*> outputTensors;
        std::vector<std::string> actualOutputNames;  // Track actual names (may differ for fused ops)
        for (size_t i = 0; i < node.outputs.size() && i < outputShapes.size(); i++) {
            std::string outputName = node.outputs[i];
            if (outputName.empty()) continue;

            // For SiLU fusion, the Conv's output becomes the Mul's output
            if (useSiluFusion && i == 0 && !siluOutputName.empty()) {
                // Debug: commented out
                // std::cout << "    Remapping output: " << outputName << " -> " << siluOutputName << std::endl;
                outputName = siluOutputName;
            }

            Shape shape = outputShapes[i];
            if (shape.empty()) {
                // Try to get shape from value info
                shape = model_->getShape(outputName);
            }

            if (!shape.empty()) {
                auto tensor = std::make_unique<Tensor>(*ctx_, shape, DataType::FLOAT32, false);
                outputTensors.push_back(tensor.get());
                actualOutputNames.push_back(outputName);
                intermediates_[outputName] = std::move(tensor);
                producedValues.insert(outputName);
                
                // For SiLU fusion, also mark the original Conv output as produced
                // (since some ops might reference it directly)
                if (useSiluFusion && i == 0 && !siluOutputName.empty()) {
                    producedValues.insert(node.outputs[0]);
                }
            }
        }

        if (outputTensors.empty()) continue;

        // Prepare operator
        try {
            op->prepare(*ctx_, inputTensors, outputTensors, node, shaderDir_);

            PreparedOp prepared;
            prepared.op = std::move(op);
            prepared.inputs = inputTensors;
            prepared.outputs = outputTensors;
            prepared.node = &node;
            prepared.outputNames = actualOutputNames;
            preparedOps_.push_back(std::move(prepared));
        } catch (const std::exception& e) {
            std::cout << "Warning: Failed to prepare " << node.opType << ": " << e.what() << std::endl;
        }
    }

    // Identify output tensors
    for (const auto& output : model_->outputs()) {
        auto it = intermediates_.find(output.name);
        if (it != intermediates_.end()) {
            outputs_[output.name] = std::move(it->second);
            intermediates_.erase(it);
        }
    }

    // Create sequence for recording
    sequence_ = std::make_unique<vkcompute::Sequence>(*ctx_);
    
    // Create transfer sequence for input copies
    transferSequence_ = std::make_unique<vkcompute::Sequence>(*ctx_);

    // Dependency analysis: determine which operations need barriers before them
    // A barrier is needed when an op reads a tensor that was written by a previous op
    // without any barrier in between.
    
    // Track tensors that have been written since the last barrier
    std::set<std::string> dirtyTensors;
    
    // Also track which tensors are written by compute (vs initializers/inputs which don't need barriers)
    std::set<std::string> computeWrittenTensors;
    
    // Analyze dependencies and mark which ops need a barrier before them
    std::vector<bool> needsBarrierBefore(preparedOps_.size(), false);
    
    for (size_t i = 0; i < preparedOps_.size(); i++) {
        auto& prepared = preparedOps_[i];
        
        // Check if any input is in the dirty set (written since last barrier)
        bool needsBarrier = false;
        for (const auto& inputName : prepared.node->inputs) {
            if (inputName.empty()) continue;
            if (dirtyTensors.count(inputName) > 0) {
                needsBarrier = true;
                break;
            }
        }
        
        if (needsBarrier) {
            needsBarrierBefore[i] = true;
            dirtyTensors.clear();  // Barrier will flush all dirty tensors
        }
        
        // Mark outputs as dirty and compute-written
        // Use actualOutputNames (may differ from node->outputs for fused ops)
        for (const auto& outputName : prepared.outputNames) {
            if (!outputName.empty()) {
                dirtyTensors.insert(outputName);
                computeWrittenTensors.insert(outputName);
            }
        }
    }
    
    // Count barriers for logging
    int barrierCount = 0;
    for (bool b : needsBarrierBefore) if (b) barrierCount++;
    
    // Pre-record all operations (reusable command buffer)
    sequence_->begin(true);  // reusable = true
    for (size_t i = 0; i < preparedOps_.size(); i++) {
        auto& prepared = preparedOps_[i];
        
        // Insert barrier if needed
        if (needsBarrierBefore[i]) {
            sequence_->barrier();
        }
        
        prepared.op->record(*sequence_, prepared.inputs, prepared.outputs, *prepared.node);
    }
    // Final barrier before end (needed for output reads)
    sequence_->barrier();
    sequence_->end();

    std::cout << "Prepared " << preparedOps_.size() << " operators with " 
              << barrierCount << " barriers (was " << preparedOps_.size() << ")" << std::endl;
}

std::map<std::string, Tensor*> Executor::run(std::map<std::string, Tensor>& inputs) {
    std::map<std::string, Tensor*> outputPtrs;
    runTimed(inputs, outputPtrs);
    return outputPtrs;
}

double Executor::runTimed(std::map<std::string, Tensor>& inputs,
                          std::map<std::string, Tensor*>& outputs) {
    // Record input copies into transfer sequence
    transferSequence_->begin();
    for (auto& [name, userTensor] : inputs) {
        auto it = inputTensors_.find(name);
        if (it != inputTensors_.end()) {
            // Record copy from pinned (user) tensor to device-local tensor
            transferSequence_->recordCopy(userTensor.buffer(), it->second->buffer());
        }
    }
    // Barrier to ensure transfers complete before compute reads
    transferSequence_->transferBarrier();
    transferSequence_->end();

    // Submit both transfer and compute commands in one batch
    double timeMs = sequence_->submitWithPrefixAndWait(transferSequence_->cmdBuffer());

    // Return output pointers
    for (auto& [name, tensor] : outputs_) {
        outputs[name] = tensor.get();
    }

    return timeMs;
}

std::map<std::string, double> Executor::runProfile(std::map<std::string, Tensor>& inputs) {
    std::map<std::string, double> timings;
    
    // Count operators by type
    std::map<std::string, int> opCounts;
    for (const auto& prepared : preparedOps_) {
        opCounts[prepared.node->opType]++;
    }
    
    // Run full inference and get total time
    std::map<std::string, Tensor*> outputs;
    double totalMs = runTimed(inputs, outputs);
    
    std::cout << "\n=== Operator Counts ===" << std::endl;
    for (const auto& [opType, count] : opCounts) {
        std::cout << "  " << opType << ": " << count << std::endl;
        timings[opType + "_count"] = count;
    }
    timings["total_ms"] = totalMs;
    
    return timings;
}

} // namespace onnxrt
