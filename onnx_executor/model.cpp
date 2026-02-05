#include "model.hpp"
#include <onnx/onnx.pb.h>
#include <fstream>
#include <stdexcept>
#include <iostream>
#include <set>

namespace onnxrt {

// Convert ONNX data type to our enum
static DataType fromOnnxDataType(int onnxType) {
    switch (onnxType) {
        case 1:  return DataType::FLOAT32;
        case 2:  return DataType::UINT8;
        case 3:  return DataType::INT8;
        case 5:  return DataType::INT16;
        case 6:  return DataType::INT32;
        case 7:  return DataType::INT64;
        case 10: return DataType::FLOAT16;
        case 11: return DataType::DOUBLE;
        case 9:  return DataType::BOOL;
        default: return DataType::FLOAT32;  // Default to float32
    }
}

// Extract shape from TensorTypeProto
static Shape extractShape(const onnx::TensorShapeProto& shapeProto) {
    Shape shape;
    for (const auto& dim : shapeProto.dim()) {
        if (dim.has_dim_value()) {
            shape.push_back(dim.dim_value());
        } else {
            // Dynamic dimension - use -1 as placeholder
            shape.push_back(-1);
        }
    }
    return shape;
}

Model Model::load(const std::string& onnxPath, vkcompute::Context& ctx) {
    // Read the file
    std::ifstream file(onnxPath, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open ONNX file: " + onnxPath);
    }

    // Parse protobuf
    onnx::ModelProto modelProto;
    if (!modelProto.ParseFromIstream(&file)) {
        throw std::runtime_error("Failed to parse ONNX file: " + onnxPath);
    }

    std::cout << "Loading ONNX model: " << onnxPath << std::endl;
    std::cout << "  Producer: " << modelProto.producer_name() << " " << modelProto.producer_version() << std::endl;
    std::cout << "  IR version: " << modelProto.ir_version() << std::endl;

    const auto& graph = modelProto.graph();
    Model model;

    // Collect names of initializers (constants/weights)
    std::set<std::string> initializerNames;
    for (const auto& init : graph.initializer()) {
        initializerNames.insert(init.name());
    }

    // Parse graph inputs (excluding initializers)
    for (const auto& input : graph.input()) {
        Value val;
        val.name = input.name();

        // Skip initializers in the input list
        if (initializerNames.count(val.name)) continue;

        if (input.has_type() && input.type().has_tensor_type()) {
            const auto& tensorType = input.type().tensor_type();
            val.dtype = fromOnnxDataType(tensorType.elem_type());
            if (tensorType.has_shape()) {
                val.shape = extractShape(tensorType.shape());
            }
        }
        model.inputs_.push_back(val);
        model.valueInfos_[val.name] = val;
    }

    // Parse graph outputs
    for (const auto& output : graph.output()) {
        Value val;
        val.name = output.name();
        if (output.has_type() && output.type().has_tensor_type()) {
            const auto& tensorType = output.type().tensor_type();
            val.dtype = fromOnnxDataType(tensorType.elem_type());
            if (tensorType.has_shape()) {
                val.shape = extractShape(tensorType.shape());
            }
        }
        model.outputs_.push_back(val);
        model.valueInfos_[val.name] = val;
    }

    // Parse value_info for intermediate tensors
    for (const auto& vi : graph.value_info()) {
        Value val;
        val.name = vi.name();
        if (vi.has_type() && vi.type().has_tensor_type()) {
            const auto& tensorType = vi.type().tensor_type();
            val.dtype = fromOnnxDataType(tensorType.elem_type());
            if (tensorType.has_shape()) {
                val.shape = extractShape(tensorType.shape());
            }
        }
        model.valueInfos_[val.name] = val;
    }

    // Load initializers (weights) into GPU tensors
    std::cout << "  Loading " << graph.initializer_size() << " initializers..." << std::endl;
    for (const auto& init : graph.initializer()) {
        Shape shape;
        for (int64_t dim : init.dims()) {
            shape.push_back(dim);
        }

        DataType dtype = fromOnnxDataType(init.data_type());

        // Create tensor and copy data
        Tensor tensor(ctx, shape, dtype, false);  // Device-local

        // Get raw data
        const void* rawData = nullptr;
        size_t rawSize = 0;

        if (init.has_raw_data()) {
            rawData = init.raw_data().data();
            rawSize = init.raw_data().size();
        } else if (init.float_data_size() > 0) {
            rawData = init.float_data().data();
            rawSize = init.float_data_size() * sizeof(float);
        } else if (init.int32_data_size() > 0) {
            rawData = init.int32_data().data();
            rawSize = init.int32_data_size() * sizeof(int32_t);
        } else if (init.int64_data_size() > 0) {
            rawData = init.int64_data().data();
            rawSize = init.int64_data_size() * sizeof(int64_t);
        }

        if (rawData && rawSize > 0) {
            tensor.copyFromHost(rawData, rawSize);
        }

        // Store value info for initializer
        Value val;
        val.name = init.name();
        val.shape = shape;
        val.dtype = dtype;
        model.valueInfos_[val.name] = val;

        model.initializers_[init.name()] = std::move(tensor);
    }

    // Parse nodes and extract Constant values as initializers
    for (const auto& node : graph.node()) {
        // Handle Constant nodes specially - extract their value as an initializer
        if (node.op_type() == "Constant" && node.output_size() > 0) {
            for (const auto& attr : node.attribute()) {
                if (attr.name() == "value" && attr.has_t()) {
                    const auto& tensorProto = attr.t();
                    std::string outputName = node.output(0);

                    Shape shape;
                    for (int64_t dim : tensorProto.dims()) {
                        shape.push_back(dim);
                    }
                    if (shape.empty()) shape.push_back(1);  // Scalar

                    DataType dtype = fromOnnxDataType(tensorProto.data_type());
                    Tensor tensor(ctx, shape, dtype, false);

                    // Get raw data
                    const void* rawData = nullptr;
                    size_t rawSize = 0;

                    if (tensorProto.has_raw_data()) {
                        rawData = tensorProto.raw_data().data();
                        rawSize = tensorProto.raw_data().size();
                    } else if (tensorProto.int64_data_size() > 0) {
                        rawData = tensorProto.int64_data().data();
                        rawSize = tensorProto.int64_data_size() * sizeof(int64_t);
                    } else if (tensorProto.float_data_size() > 0) {
                        rawData = tensorProto.float_data().data();
                        rawSize = tensorProto.float_data_size() * sizeof(float);
                    }

                    if (rawData && rawSize > 0) {
                        tensor.copyFromHost(rawData, rawSize);
                    }

                    Value val;
                    val.name = outputName;
                    val.shape = shape;
                    val.dtype = dtype;
                    model.valueInfos_[outputName] = val;
                    model.initializers_[outputName] = std::move(tensor);
                }
            }
            continue;  // Don't add Constant to nodes list
        }

        Node n;
        n.name = node.name();
        n.opType = node.op_type();

        for (const auto& input : node.input()) {
            n.inputs.push_back(input);
        }
        for (const auto& output : node.output()) {
            n.outputs.push_back(output);
        }

        // Parse attributes
        for (const auto& attr : node.attribute()) {
            switch (attr.type()) {
                case onnx::AttributeProto::INT:
                    n.attributes[attr.name()] = attr.i();
                    break;
                case onnx::AttributeProto::FLOAT:
                    n.attributes[attr.name()] = attr.f();
                    break;
                case onnx::AttributeProto::STRING:
                    n.attributes[attr.name()] = attr.s();
                    break;
                case onnx::AttributeProto::INTS: {
                    std::vector<int64_t> ints;
                    for (int64_t v : attr.ints()) ints.push_back(v);
                    n.attributes[attr.name()] = ints;
                    break;
                }
                case onnx::AttributeProto::FLOATS: {
                    std::vector<float> floats;
                    for (float v : attr.floats()) floats.push_back(v);
                    n.attributes[attr.name()] = floats;
                    break;
                }
                case onnx::AttributeProto::STRINGS: {
                    std::vector<std::string> strs;
                    for (const auto& s : attr.strings()) strs.push_back(s);
                    n.attributes[attr.name()] = strs;
                    break;
                }
                default:
                    // Skip unsupported attribute types
                    break;
            }
        }

        model.nodes_.push_back(n);
    }

    std::cout << "  Inputs: " << model.inputs_.size() << std::endl;
    for (const auto& inp : model.inputs_) {
        std::cout << "    " << inp.name << ": " << shapeStr(inp.shape) << std::endl;
    }
    std::cout << "  Outputs: " << model.outputs_.size() << std::endl;
    for (const auto& out : model.outputs_) {
        std::cout << "    " << out.name << ": " << shapeStr(out.shape) << std::endl;
    }
    std::cout << "  Nodes: " << model.nodes_.size() << std::endl;

    return model;
}

bool Model::isInitializer(const std::string& name) const {
    return initializers_.find(name) != initializers_.end();
}

const Value* Model::getValueInfo(const std::string& name) const {
    auto it = valueInfos_.find(name);
    if (it != valueInfos_.end()) {
        return &it->second;
    }
    return nullptr;
}

Shape Model::getShape(const std::string& name) const {
    // Check value info first
    if (auto* val = getValueInfo(name)) {
        return val->shape;
    }
    // Check initializers
    auto it = initializers_.find(name);
    if (it != initializers_.end()) {
        return it->second.shape();
    }
    return {};
}

} // namespace onnxrt
