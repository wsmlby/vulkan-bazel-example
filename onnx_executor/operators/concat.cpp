#include "concat.hpp"
#include <stdexcept>
#include <vulkan/vulkan.h>

namespace onnxrt {

std::vector<Shape> ConcatOp::inferShapes(const std::vector<Shape>& inputShapes, const Node& node) const {
    if (inputShapes.empty()) return {{}};

    int64_t axis = node.getAttr<int64_t>("axis", 0);
    if (axis < 0) axis += inputShapes[0].size();

    Shape output = inputShapes[0];
    for (size_t i = 1; i < inputShapes.size(); i++) {
        if (!inputShapes[i].empty()) {
            output[axis] += inputShapes[i][axis];
        }
    }
    return {output};
}

void ConcatOp::prepare(vkcompute::Context& ctx, const std::vector<Tensor*>& inputs,
                       const std::vector<Tensor*>& outputs, const Node& node, const std::string& shaderDir) {
    axis_ = node.getAttr<int64_t>("axis", 0);
    if (axis_ < 0 && !inputs.empty()) axis_ += inputs[0]->shape().size();
    
    ctx_ = &ctx;
    
    // Store info for each input
    inputInfos_.clear();
    size_t offset = 0;
    for (auto* input : inputs) {
        if (input) {
            InputInfo info;
            info.elementCount = input->elementCount();
            info.offset = offset;
            inputInfos_.push_back(info);
            offset += info.elementCount;
        }
    }
    
    // Store buffer pointers for runtime
    inputBuffers_.clear();
    for (auto* input : inputs) {
        if (input) inputBuffers_.push_back(&input->buffer());
    }
    outputBuffer_ = &outputs[0]->buffer();
}

void ConcatOp::record(vkcompute::Sequence& seq, const std::vector<Tensor*>& inputs,
                      const std::vector<Tensor*>& outputs, const Node& node) {
    // Use buffer copies for concatenation
    // This is correct for concat along axis 0, or when lower dims are all 1
    VkCommandBuffer cmd = seq.cmdBuffer();
    
    size_t dstOffset = 0;
    for (size_t i = 0; i < inputBuffers_.size(); i++) {
        size_t size = inputInfos_[i].elementCount * sizeof(float);
        
        VkBufferCopy copyRegion{};
        copyRegion.srcOffset = 0;
        copyRegion.dstOffset = dstOffset;
        copyRegion.size = size;
        vkCmdCopyBuffer(cmd, inputBuffers_[i]->buffer, outputBuffer_->buffer, 1, &copyRegion);
        
        dstOffset += size;
    }
}

REGISTER_OPERATOR("Concat", ConcatOp);

} // namespace onnxrt
