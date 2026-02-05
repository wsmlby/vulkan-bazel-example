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
    VkCommandBuffer cmd = seq.cmdBuffer();
    
    const auto& outShape = outputs[0]->shape();
    int64_t axis = axis_;
    
    // Calculate sizes for proper concatenation
    // innerSize: number of elements in dimensions after axis (stride for axis)
    // outerSize: number of elements in dimensions before axis
    int64_t innerSize = 1;
    for (size_t i = axis + 1; i < outShape.size(); i++) {
        innerSize *= outShape[i];
    }
    
    int64_t outerSize = 1;
    for (size_t i = 0; i < (size_t)axis; i++) {
        outerSize *= outShape[i];
    }
    
    // For each outer position, copy each input's slice at axis position
    for (int64_t outer = 0; outer < outerSize; outer++) {
        VkDeviceSize dstOffset = outer * outShape[axis] * innerSize * sizeof(float);
        
        for (size_t i = 0; i < inputs.size(); i++) {
            const auto& inShape = inputs[i]->shape();
            int64_t axisSize = inShape[axis];
            VkDeviceSize copySize = axisSize * innerSize * sizeof(float);
            
            VkDeviceSize srcOffset = outer * axisSize * innerSize * sizeof(float);
            
            VkBufferCopy copyRegion{};
            copyRegion.srcOffset = srcOffset;
            copyRegion.dstOffset = dstOffset;
            copyRegion.size = copySize;
            vkCmdCopyBuffer(cmd, inputs[i]->buffer().buffer, outputBuffer_->buffer, 1, &copyRegion);
            
            dstOffset += copySize;
        }
    }
}

REGISTER_OPERATOR("Concat", ConcatOp);

} // namespace onnxrt
