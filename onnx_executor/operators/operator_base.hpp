#pragma once

#include "../tensor.hpp"
#include "../model.hpp"
#include "../../vk_compute.hpp"
#include <vector>
#include <memory>
#include <string>

namespace onnxrt {

// Forward declaration
class ShaderManager;

/**
 * Operator - Base class for ONNX operator implementations
 *
 * Each operator knows how to:
 * 1. Infer output shapes from input shapes
 * 2. Create/bind compute pipelines
 * 3. Record dispatch commands
 */
class Operator {
public:
    virtual ~Operator() = default;

    // Get operator type name (e.g., "Add", "Conv")
    virtual std::string opType() const = 0;

    // Infer output shapes given input shapes and node attributes
    // Returns one shape per output
    virtual std::vector<Shape> inferShapes(
        const std::vector<Shape>& inputShapes,
        const Node& node) const = 0;

    // Prepare the operator (create pipelines, allocate resources)
    // Called once during model loading
    virtual void prepare(
        vkcompute::Context& ctx,
        const std::vector<Tensor*>& inputs,
        const std::vector<Tensor*>& outputs,
        const Node& node,
        const std::string& shaderDir) = 0;

    // Record dispatch commands to the sequence
    // Called during inference
    virtual void record(
        vkcompute::Sequence& seq,
        const std::vector<Tensor*>& inputs,
        const std::vector<Tensor*>& outputs,
        const Node& node) = 0;
};

} // namespace onnxrt
