#pragma once

#include "operator_base.hpp"
#include "operator_registry.hpp"

namespace onnxrt {

// Push constants for Conv2D
struct Conv2DParams {
    uint32_t N;         // Batch size
    uint32_t C;         // Input channels
    uint32_t H;         // Input height
    uint32_t W;         // Input width
    uint32_t K;         // Output channels (num filters)
    uint32_t R;         // Kernel height
    uint32_t S;         // Kernel width
    uint32_t outH;      // Output height
    uint32_t outW;      // Output width
    uint32_t padH;      // Padding height
    uint32_t padW;      // Padding width
    uint32_t strideH;   // Stride height
    uint32_t strideW;   // Stride width
    uint32_t dilationH; // Dilation height
    uint32_t dilationW; // Dilation width
    uint32_t groups;    // Number of groups
    uint32_t hasBias;   // 1 if bias present
};

/**
 * ConvOp - 2D Convolution
 */
class ConvOp : public Operator {
public:
    std::string opType() const override { return "Conv"; }

    std::vector<Shape> inferShapes(
        const std::vector<Shape>& inputShapes,
        const Node& node) const override;

    void prepare(
        vkcompute::Context& ctx,
        const std::vector<Tensor*>& inputs,
        const std::vector<Tensor*>& outputs,
        const Node& node,
        const std::string& shaderDir) override;

    void record(
        vkcompute::Sequence& seq,
        const std::vector<Tensor*>& inputs,
        const std::vector<Tensor*>& outputs,
        const Node& node) override;

private:
    std::unique_ptr<vkcompute::ComputePipeline> pipeline_;
    Conv2DParams params_;
};

} // namespace onnxrt
