#pragma once

#include "operator_base.hpp"
#include "operator_registry.hpp"

namespace onnxrt {

struct MaxPoolParams {
    uint32_t N, C, H, W;
    uint32_t outH, outW;
    uint32_t kH, kW;
    uint32_t strideH, strideW;
    uint32_t padH, padW;
};

class MaxPoolOp : public Operator {
public:
    std::string opType() const override { return "MaxPool"; }
    std::vector<Shape> inferShapes(const std::vector<Shape>& inputShapes, const Node& node) const override;
    void prepare(vkcompute::Context& ctx, const std::vector<Tensor*>& inputs,
                 const std::vector<Tensor*>& outputs, const Node& node, const std::string& shaderDir) override;
    void record(vkcompute::Sequence& seq, const std::vector<Tensor*>& inputs,
                const std::vector<Tensor*>& outputs, const Node& node) override;
private:
    std::unique_ptr<vkcompute::ComputePipeline> pipeline_;
    MaxPoolParams params_;
};

} // namespace onnxrt
