#pragma once

#include "conv.hpp"

namespace onnxrt {

/**
 * ConvSiluOp - Fused Conv + SiLU activation
 * SiLU(x) = x * sigmoid(x)
 * 
 * This operator fuses Conv -> Sigmoid -> Mul patterns commonly found in YOLOv5
 * to eliminate 2 memory round-trips.
 */
class ConvSiluOp : public ConvOp {
public:
    std::string opType() const override { return "ConvSilu"; }

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
