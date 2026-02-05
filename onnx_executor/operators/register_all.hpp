#pragma once

namespace onnxrt {

// Force registration of all operators
// This must be called before using the registry
void registerAllOperators();

} // namespace onnxrt
