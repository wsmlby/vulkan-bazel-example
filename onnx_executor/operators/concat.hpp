#pragma once
#include "operator_base.hpp"
#include "operator_registry.hpp"

namespace onnxrt {

class ConcatOp : public Operator {
public:
    std::string opType() const override { return "Concat"; }
    std::vector<Shape> inferShapes(const std::vector<Shape>& inputShapes, const Node& node) const override;
    void prepare(vkcompute::Context& ctx, const std::vector<Tensor*>& inputs,
                 const std::vector<Tensor*>& outputs, const Node& node, const std::string& shaderDir) override;
    void record(vkcompute::Sequence& seq, const std::vector<Tensor*>& inputs,
                const std::vector<Tensor*>& outputs, const Node& node) override;
private:
    vkcompute::Context* ctx_ = nullptr;
    int64_t axis_ = 0;
    struct InputInfo {
        size_t elementCount;
        size_t offset;
    };
    std::vector<InputInfo> inputInfos_;
    std::vector<vkcompute::Buffer*> inputBuffers_;
    vkcompute::Buffer* outputBuffer_ = nullptr;
};

} // namespace onnxrt
