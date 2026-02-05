#pragma once

#include "operator_base.hpp"
#include "operator_registry.hpp"

namespace onnxrt {

// Push constants for elementwise binary ops with broadcasting
struct BinaryOpParams {
    uint32_t totalElements;
    uint32_t aElements;
    uint32_t bElements;
    uint32_t pad;
};

// Push constants for elementwise unary ops
struct UnaryOpParams {
    uint32_t totalElements;
    float param;      // Extra parameter (e.g., exponent for Pow)
    uint32_t pad[2];
};

/**
 * AddOp - Element-wise addition
 */
class AddOp : public Operator {
public:
    std::string opType() const override { return "Add"; }

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
    uint32_t totalElements_ = 0;
    uint32_t aElements_ = 0;
    uint32_t bElements_ = 0;
};

/**
 * MulOp - Element-wise multiplication
 */
class MulOp : public Operator {
public:
    std::string opType() const override { return "Mul"; }

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
    uint32_t totalElements_ = 0;
    uint32_t aElements_ = 0;
    uint32_t bElements_ = 0;
};

/**
 * PowOp - Element-wise power
 */
class PowOp : public Operator {
public:
    std::string opType() const override { return "Pow"; }

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
    uint32_t totalElements_ = 0;
    uint32_t aElements_ = 0;
    uint32_t bElements_ = 0;
};

/**
 * SigmoidOp - Sigmoid activation: 1 / (1 + exp(-x))
 */
class SigmoidOp : public Operator {
public:
    std::string opType() const override { return "Sigmoid"; }

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
    uint32_t totalElements_ = 0;
};

} // namespace onnxrt
