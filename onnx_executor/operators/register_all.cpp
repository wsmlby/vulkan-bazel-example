#include "register_all.hpp"
#include "operator_registry.hpp"
#include "elementwise.hpp"
#include "conv.hpp"
#include "conv_silu.hpp"
#include "pool.hpp"
#include "resize.hpp"
#include "concat.hpp"
#include "slice.hpp"
#include "transpose.hpp"
#include "reshape.hpp"

namespace onnxrt {

void registerAllOperators() {
    static bool registered = false;
    if (registered) return;
    registered = true;

    auto& reg = OperatorRegistry::instance();

    // Elementwise operators
    reg.registerOp("Add", []() { return std::make_unique<AddOp>(); });
    reg.registerOp("Mul", []() { return std::make_unique<MulOp>(); });
    reg.registerOp("Pow", []() { return std::make_unique<PowOp>(); });
    reg.registerOp("Sigmoid", []() { return std::make_unique<SigmoidOp>(); });

    // Convolution
    reg.registerOp("Conv", []() { return std::make_unique<ConvOp>(); });
    reg.registerOp("ConvSilu", []() { return std::make_unique<ConvSiluOp>(); });

    // Pooling
    reg.registerOp("MaxPool", []() { return std::make_unique<MaxPoolOp>(); });

    // Resize
    reg.registerOp("Resize", []() { return std::make_unique<ResizeOp>(); });

    // Concat
    reg.registerOp("Concat", []() { return std::make_unique<ConcatOp>(); });

    // Slice and Split
    reg.registerOp("Slice", []() { return std::make_unique<SliceOp>(); });
    reg.registerOp("Split", []() { return std::make_unique<SplitOp>(); });

    // Transpose
    reg.registerOp("Transpose", []() { return std::make_unique<TransposeOp>(); });

    // Reshape and Constant
    reg.registerOp("Reshape", []() { return std::make_unique<ReshapeOp>(); });
    reg.registerOp("Constant", []() { return std::make_unique<ConstantOp>(); });
}

} // namespace onnxrt
