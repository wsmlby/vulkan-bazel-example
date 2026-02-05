#pragma once

#include "operator_base.hpp"
#include "operator_registry.hpp"

namespace onnxrt {

/**
 * ReshapeOp - Reshape tensor (data copy, different shape)
 */
class ReshapeOp : public Operator {
public:
    std::string opType() const override { return "Reshape"; }

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
    vkcompute::Buffer* inputBuffer_ = nullptr;
    vkcompute::Buffer* outputBuffer_ = nullptr;
    size_t byteSize_ = 0;
};

/**
 * ConstantOp - Constant tensor (already loaded as initializer)
 */
class ConstantOp : public Operator {
public:
    std::string opType() const override { return "Constant"; }

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
};

} // namespace onnxrt
