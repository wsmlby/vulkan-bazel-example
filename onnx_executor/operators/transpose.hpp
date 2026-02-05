#pragma once
#include "operator_base.hpp"
#include "operator_registry.hpp"

namespace onnxrt {

class TransposeOp : public Operator {
public:
    std::string opType() const override { return "Transpose"; }
    std::vector<Shape> inferShapes(const std::vector<Shape>& inputShapes, const Node& node) const override;
    void prepare(vkcompute::Context& ctx, const std::vector<Tensor*>& inputs,
                 const std::vector<Tensor*>& outputs, const Node& node, const std::string& shaderDir) override;
    void record(vkcompute::Sequence& seq, const std::vector<Tensor*>& inputs,
                const std::vector<Tensor*>& outputs, const Node& node) override;
private:
    std::unique_ptr<vkcompute::ComputePipeline> pipeline_;
    std::vector<int64_t> perm_;
    Shape inputShape_;
    Shape outputShape_;
    uint32_t totalElements_ = 0;
};

} // namespace onnxrt
