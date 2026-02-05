#pragma once

#include "model.hpp"
#include "tensor.hpp"
#include "operators/operator_base.hpp"
#include "../vk_compute.hpp"
#include <map>
#include <memory>

namespace onnxrt {

/**
 * TensorInfo - Metadata about model inputs/outputs
 */
struct TensorInfo {
    std::string name;
    Shape shape;
    DataType dtype;
};

/**
 * Executor - Runs ONNX models on the GPU
 *
 * Usage:
 *   vkcompute::Context ctx;
 *   onnxrt::Executor executor(ctx);
 *   executor.loadModel("model.onnx");
 *
 *   auto inputs = executor.createInputs();
 *   // Fill input tensors...
 *
 *   auto outputs = executor.run(inputs);
 *   // Read output tensors...
 */
class Executor {
public:
    explicit Executor(vkcompute::Context& ctx, const std::string& shaderDir = "");
    ~Executor();

    // Load an ONNX model
    void loadModel(const std::string& onnxPath);

    // Get input tensor descriptions
    std::vector<TensorInfo> getInputInfo() const;

    // Get output tensor descriptions
    std::vector<TensorInfo> getOutputInfo() const;

    // Create input tensors (caller fills with data)
    std::map<std::string, Tensor> createInputs() const;

    // Run inference
    // Returns output tensors (caller should copy to host if needed)
    std::map<std::string, Tensor*> run(std::map<std::string, Tensor>& inputs);

    // Run inference and return timing in milliseconds
    double runTimed(std::map<std::string, Tensor>& inputs,
                    std::map<std::string, Tensor*>& outputs);

    // Run inference with per-operator profiling
    // Returns a map of operator name/type -> time in milliseconds
    std::map<std::string, double> runProfile(std::map<std::string, Tensor>& inputs);

private:
    vkcompute::Context* ctx_;
    std::string shaderDir_;
    std::unique_ptr<Model> model_;

    // Input tensors (device-local copies)
    std::map<std::string, std::unique_ptr<Tensor>> inputTensors_;

    // Intermediate tensors
    std::map<std::string, std::unique_ptr<Tensor>> intermediates_;

    // Output tensors
    std::map<std::string, std::unique_ptr<Tensor>> outputs_;

    // Prepared operators for each node
    struct PreparedOp {
        std::unique_ptr<Operator> op;
        std::vector<Tensor*> inputs;
        std::vector<Tensor*> outputs;
        const Node* node;
        std::vector<std::string> outputNames;  // Actual output tensor names (may differ from node->outputs for fused ops)
    };
    std::vector<PreparedOp> preparedOps_;

    // Sequence for recording commands
    std::unique_ptr<vkcompute::Sequence> sequence_;
    
    // Transfer sequence for input copies (recorded fresh each frame)
    std::unique_ptr<vkcompute::Sequence> transferSequence_;

    // Prepare the graph (allocate intermediates, create pipelines)
    void prepare();

    // Find shader directory from runfiles
    std::string findShaderDir() const;
};

} // namespace onnxrt
