#pragma once
#include "operator_base.hpp"
#include "operator_registry.hpp"

namespace onnxrt {

class SliceOp : public Operator {
public:
    std::string opType() const override { return "Slice"; }
    std::vector<Shape> inferShapes(const std::vector<Shape>& inputShapes, const Node& node) const override;
    void prepare(vkcompute::Context& ctx, const std::vector<Tensor*>& inputs,
                 const std::vector<Tensor*>& outputs, const Node& node, const std::string& shaderDir) override;
    void record(vkcompute::Sequence& seq, const std::vector<Tensor*>& inputs,
                const std::vector<Tensor*>& outputs, const Node& node) override;
private:
    std::unique_ptr<vkcompute::ComputePipeline> pipeline_;
    Shape inputShape_;
    Shape outputShape_;
    std::vector<int64_t> inputStrides_;
    std::vector<int64_t> outputStrides_;
    std::vector<int64_t> starts_;
    std::vector<int64_t> steps_;
    uint32_t totalElements_ = 0;
};

class SplitOp : public Operator {
public:
    std::string opType() const override { return "Split"; }
    std::vector<Shape> inferShapes(const std::vector<Shape>& inputShapes, const Node& node) const override;
    void prepare(vkcompute::Context& ctx, const std::vector<Tensor*>& inputs,
                 const std::vector<Tensor*>& outputs, const Node& node, const std::string& shaderDir) override;
    void record(vkcompute::Sequence& seq, const std::vector<Tensor*>& inputs,
                const std::vector<Tensor*>& outputs, const Node& node) override;
private:
    vkcompute::Context* ctx_ = nullptr;
    int64_t axis_ = 0;
    Shape inputShape_;
    std::vector<Shape> outputShapes_;
};

} // namespace onnxrt
