#pragma once
#include "operator_base.hpp"
#include "operator_registry.hpp"

namespace onnxrt {

struct ResizeParams {
    uint32_t N, C, inH, inW, outH, outW;
    float scaleH, scaleW;
};

class ResizeOp : public Operator {
public:
    std::string opType() const override { return "Resize"; }
    std::vector<Shape> inferShapes(const std::vector<Shape>& inputShapes, const Node& node) const override;
    void prepare(vkcompute::Context& ctx, const std::vector<Tensor*>& inputs,
                 const std::vector<Tensor*>& outputs, const Node& node, const std::string& shaderDir) override;
    void record(vkcompute::Sequence& seq, const std::vector<Tensor*>& inputs,
                const std::vector<Tensor*>& outputs, const Node& node) override;
private:
    std::unique_ptr<vkcompute::ComputePipeline> pipeline_;
    ResizeParams params_;
};

} // namespace onnxrt
