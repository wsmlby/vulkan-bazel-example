#include "slice.hpp"
#include <algorithm>
#include <cstring>
#include <vulkan/vulkan.h>

namespace onnxrt {

// Push constant structure for 4D slice
struct SliceParams {
    uint32_t outN, outC, outH, outW;
    uint32_t inN, inC, inH, inW;
    int32_t startN, startC, startH, startW;
    int32_t stepN, stepC, stepH, stepW;
};

std::vector<Shape> SliceOp::inferShapes(const std::vector<Shape>& inputShapes, const Node& node) const {
    // Slice: inputs are [data, starts, ends, axes (optional), steps (optional)]
    // We can't infer shape without the actual values of starts/ends/axes/steps
    // The executor will compute actual shape by reading constant inputs
    if (inputShapes.empty()) return {{}};
    return {inputShapes[0]};  // Placeholder - executor will compute actual shape
}

void SliceOp::prepare(vkcompute::Context& ctx, const std::vector<Tensor*>& inputs,
                      const std::vector<Tensor*>& outputs, const Node& node, const std::string& shaderDir) {
    // Store parameters for slice operation
    const auto& inputShape = inputs[0]->shape();
    const auto& outputShape = outputs[0]->shape();
    
    inputShape_ = inputShape;
    outputShape_ = outputShape;
    
    // Initialize default slice parameters (no-op slice)
    starts_.resize(4, 0);
    steps_.resize(4, 1);
    
    // Read slice parameters from constant inputs if available
    if (inputs.size() >= 3) {
        // starts tensor
        if (inputs[1] && inputs[1]->elementCount() > 0) {
            std::vector<int64_t> startsData(inputs[1]->elementCount());
            inputs[1]->copyToHost(startsData.data(), startsData.size() * sizeof(int64_t));
            
            // ends tensor
            std::vector<int64_t> endsData(inputs[2]->elementCount());
            inputs[2]->copyToHost(endsData.data(), endsData.size() * sizeof(int64_t));
            
            // axes (optional)
            std::vector<int64_t> axesData;
            if (inputs.size() > 3 && inputs[3] && inputs[3]->elementCount() > 0) {
                axesData.resize(inputs[3]->elementCount());
                inputs[3]->copyToHost(axesData.data(), axesData.size() * sizeof(int64_t));
            } else {
                for (size_t i = 0; i < startsData.size(); i++) axesData.push_back(i);
            }
            
            // steps (optional)
            std::vector<int64_t> stepsData;
            if (inputs.size() > 4 && inputs[4] && inputs[4]->elementCount() > 0) {
                stepsData.resize(inputs[4]->elementCount());
                inputs[4]->copyToHost(stepsData.data(), stepsData.size() * sizeof(int64_t));
            } else {
                stepsData.resize(axesData.size(), 1);
            }
            
            // Apply to our 4D parameters
            for (size_t i = 0; i < axesData.size(); i++) {
                int64_t axis = axesData[i];
                if (axis < 0) axis += inputShape.size();
                if (axis >= 0 && axis < 4) {
                    int64_t start = startsData[i];
                    if (start < 0) start += inputShape[axis];
                    start = std::max((int64_t)0, std::min(start, inputShape[axis]));
                    starts_[axis] = start;
                    steps_[axis] = stepsData[i];
                }
            }
        }
    }
    
    totalElements_ = outputs[0]->elementCount();
    
    std::string shaderPath = shaderDir + "/slice_shader.spv";
    pipeline_ = std::make_unique<vkcompute::ComputePipeline>(
        ctx, shaderPath, 2, 256, sizeof(SliceParams));
    pipeline_->bindBuffers({&inputs[0]->buffer(), &outputs[0]->buffer()});
}

void SliceOp::record(vkcompute::Sequence& seq, const std::vector<Tensor*>& inputs,
                     const std::vector<Tensor*>& outputs, const Node& node) {
    // Pad shapes to 4D
    auto pad4D = [](const Shape& s) -> std::array<int64_t, 4> {
        std::array<int64_t, 4> r = {1, 1, 1, 1};
        for (size_t i = 0; i < s.size() && i < 4; i++) {
            r[4 - s.size() + i] = s[i];
        }
        return r;
    };
    
    auto in4D = pad4D(inputShape_);
    auto out4D = pad4D(outputShape_);
    
    SliceParams params;
    params.outN = out4D[0]; params.outC = out4D[1]; params.outH = out4D[2]; params.outW = out4D[3];
    params.inN = in4D[0]; params.inC = in4D[1]; params.inH = in4D[2]; params.inW = in4D[3];
    
    // Pad starts and steps to 4D
    std::array<int64_t, 4> starts4D = {0, 0, 0, 0};
    std::array<int64_t, 4> steps4D = {1, 1, 1, 1};
    for (size_t i = 0; i < starts_.size() && i < 4; i++) {
        starts4D[4 - starts_.size() + i] = starts_[i];
        steps4D[4 - steps_.size() + i] = steps_[i];
    }
    
    params.startN = starts4D[0]; params.startC = starts4D[1]; params.startH = starts4D[2]; params.startW = starts4D[3];
    params.stepN = steps4D[0]; params.stepC = steps4D[1]; params.stepH = steps4D[2]; params.stepW = steps4D[3];
    
    pipeline_->setPushConstants(&params, sizeof(params));
    pipeline_->recordTo(seq.cmdBuffer(), (totalElements_ + 255) / 256);
}

REGISTER_OPERATOR("Slice", SliceOp);

std::vector<Shape> SplitOp::inferShapes(const std::vector<Shape>& inputShapes, const Node& node) const {
    if (inputShapes.empty()) return {};
    
    const auto& input = inputShapes[0];
    int64_t axis = node.getAttr<int64_t>("axis", 0);
    if (axis < 0) axis += input.size();
    
    auto split = node.getAttr<std::vector<int64_t>>("split", {});
    int64_t numOutputs = node.getAttr<int64_t>("num_outputs", 0);
    
    std::vector<Shape> outputs;
    
    if (!split.empty()) {
        // Split sizes provided
        for (int64_t s : split) {
            Shape out = input;
            out[axis] = s;
            outputs.push_back(out);
        }
    } else if (numOutputs > 0) {
        // Equal split
        int64_t splitSize = input[axis] / numOutputs;
        for (int64_t i = 0; i < numOutputs; i++) {
            Shape out = input;
            out[axis] = splitSize;
            outputs.push_back(out);
        }
    }
    
    return outputs;
}

void SplitOp::prepare(vkcompute::Context& ctx, const std::vector<Tensor*>& inputs,
                      const std::vector<Tensor*>& outputs, const Node& node, const std::string& shaderDir) {
    axis_ = node.getAttr<int64_t>("axis", 0);
    if (axis_ < 0) axis_ += inputs[0]->shape().size();
    
    // Store input and output info
    inputShape_ = inputs[0]->shape();
    for (auto* out : outputs) {
        outputShapes_.push_back(out->shape());
    }
    
    ctx_ = &ctx;
}

void SplitOp::record(vkcompute::Sequence& seq, const std::vector<Tensor*>& inputs,
                     const std::vector<Tensor*>& outputs, const Node& node) {
    // Split tensor along axis into multiple outputs
    // We use buffer copies for contiguous memory regions
    
    const auto& inShape = inputShape_;
    int64_t axis = axis_;
    
    // Calculate the stride for the split axis
    int64_t innerSize = 1;  // Elements after axis
    for (size_t i = axis + 1; i < inShape.size(); i++) {
        innerSize *= inShape[i];
    }
    
    int64_t outerSize = 1;  // Elements before axis
    for (size_t i = 0; i < (size_t)axis; i++) {
        outerSize *= inShape[i];
    }
    
    VkDeviceSize srcOffset = 0;
    for (size_t outIdx = 0; outIdx < outputs.size(); outIdx++) {
        const auto& outShape = outputs[outIdx]->shape();
        int64_t splitSize = outShape[axis];
        
        // Total elements in this output
        int64_t totalElements = 1;
        for (auto d : outShape) totalElements *= d;
        
        // If contiguous (last dimension split or axis has all elements after it contiguous)
        // we can use a single copy
        int64_t copySize = splitSize * innerSize * sizeof(float);
        
        // For each outer slice, copy the split portion
        for (int64_t outer = 0; outer < outerSize; outer++) {
            VkDeviceSize srcPos = (outer * inShape[axis] * innerSize) * sizeof(float) + srcOffset;
            VkDeviceSize dstPos = (outer * splitSize * innerSize) * sizeof(float);
            
            VkBufferCopy copyRegion{};
            copyRegion.srcOffset = srcPos;
            copyRegion.dstOffset = dstPos;
            copyRegion.size = copySize;
            vkCmdCopyBuffer(seq.cmdBuffer(), inputs[0]->buffer().buffer, 
                           outputs[outIdx]->buffer().buffer, 1, &copyRegion);
        }
        
        srcOffset += splitSize * innerSize * sizeof(float);
    }
}

REGISTER_OPERATOR("Split", SplitOp);

} // namespace onnxrt
