#include "reshape.hpp"
#include <stdexcept>
#include <iostream>
#include <vulkan/vulkan.h>

namespace onnxrt {

// ============================================================================
// ReshapeOp - Metadata-only reshape (just copy data)
// ============================================================================

std::vector<Shape> ReshapeOp::inferShapes(const std::vector<Shape>& inputShapes, const Node& node) const {
    // Shape is determined by the second input (shape tensor)
    // Executor handles this by reading the shape tensor values
    return {{}};
}

void ReshapeOp::prepare(vkcompute::Context& ctx, const std::vector<Tensor*>& inputs,
                        const std::vector<Tensor*>& outputs, const Node& node,
                        const std::string& shaderDir) {
    inputBuffer_ = &inputs[0]->buffer();
    outputBuffer_ = &outputs[0]->buffer();
    byteSize_ = inputs[0]->byteSize();
}

void ReshapeOp::record(vkcompute::Sequence& seq, const std::vector<Tensor*>& inputs,
                       const std::vector<Tensor*>& outputs, const Node& node) {
    // Just copy the data (reshape is metadata-only)
    VkBufferCopy copyRegion{};
    copyRegion.srcOffset = 0;
    copyRegion.dstOffset = 0;
    copyRegion.size = byteSize_;
    vkCmdCopyBuffer(seq.cmdBuffer(), inputBuffer_->buffer, outputBuffer_->buffer, 1, &copyRegion);
}

REGISTER_OPERATOR("Reshape", ReshapeOp);

// ============================================================================
// ConstantOp - Constant tensor (already loaded)
// ============================================================================

std::vector<Shape> ConstantOp::inferShapes(const std::vector<Shape>& inputShapes, const Node& node) const {
    // Shape is determined by the constant value attribute
    if (node.hasAttr("value")) {
        // Would need to parse the tensor attribute
    }
    return {{}};
}

void ConstantOp::prepare(vkcompute::Context& ctx, const std::vector<Tensor*>& inputs,
                         const std::vector<Tensor*>& outputs, const Node& node,
                         const std::string& shaderDir) {
    // Constant values are typically loaded as initializers
    // No additional GPU work needed
}

void ConstantOp::record(vkcompute::Sequence& seq, const std::vector<Tensor*>& inputs,
                        const std::vector<Tensor*>& outputs, const Node& node) {
    // No GPU work needed
}

REGISTER_OPERATOR("Constant", ConstantOp);

} // namespace onnxrt
