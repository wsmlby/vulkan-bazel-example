#pragma once

#include "../vk_compute.hpp"
#include <vector>
#include <string>
#include <cstdint>
#include <numeric>
#include <memory>

namespace onnxrt {

// Data types matching ONNX TensorProto::DataType
enum class DataType {
    UNDEFINED = 0,
    FLOAT32 = 1,
    UINT8 = 2,
    INT8 = 3,
    INT16 = 5,
    INT32 = 6,
    INT64 = 7,
    FLOAT16 = 10,
    DOUBLE = 11,
    BOOL = 9,
};

// Returns byte size of a data type
inline size_t dataTypeSize(DataType dt) {
    switch (dt) {
        case DataType::FLOAT32: return 4;
        case DataType::FLOAT16: return 2;
        case DataType::DOUBLE: return 8;
        case DataType::INT64: return 8;
        case DataType::INT32: return 4;
        case DataType::INT16: return 2;
        case DataType::INT8: return 1;
        case DataType::UINT8: return 1;
        case DataType::BOOL: return 1;
        default: return 0;
    }
}

// Tensor shape representation
using Shape = std::vector<int64_t>;

// Compute total element count from shape
inline size_t shapeSize(const Shape& shape) {
    if (shape.empty()) return 1;  // Scalar
    return std::accumulate(shape.begin(), shape.end(), (size_t)1, std::multiplies<size_t>());
}

// Format shape as string for debugging
inline std::string shapeStr(const Shape& shape) {
    std::string s = "[";
    for (size_t i = 0; i < shape.size(); i++) {
        if (i > 0) s += ", ";
        s += std::to_string(shape[i]);
    }
    return s + "]";
}

/**
 * Tensor - Multi-dimensional array stored in GPU memory
 *
 * Wraps a vkcompute::Buffer with shape and type metadata.
 * Supports both device-local (fast GPU) and pinned (CPU-accessible) memory.
 */
class Tensor {
public:
    Tensor() = default;

    // Create a tensor with specific shape and type
    // pinned: if true, creates CPU-accessible memory
    Tensor(vkcompute::Context& ctx, const Shape& shape, DataType dtype, bool pinned = false);

    // Move semantics
    Tensor(Tensor&& o) noexcept;
    Tensor& operator=(Tensor&& o) noexcept;

    // No copy
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    // Accessors
    const Shape& shape() const { return shape_; }
    DataType dtype() const { return dtype_; }
    size_t elementCount() const { return shapeSize(shape_); }
    size_t byteSize() const { return elementCount() * dataTypeSize(dtype_); }
    bool isPinned() const { return pinned_; }
    bool isValid() const { return ctx_ != nullptr; }

    // Access underlying buffer
    vkcompute::Buffer& buffer() { return buffer_; }
    const vkcompute::Buffer& buffer() const { return buffer_; }

    // For pinned tensors: access mapped memory directly
    template<typename T>
    T* data() {
        return static_cast<T*>(buffer_.mapped);
    }

    template<typename T>
    const T* data() const {
        return static_cast<const T*>(buffer_.mapped);
    }

    // Copy data from host memory to GPU
    void copyFromHost(const void* hostData, size_t bytes);

    // Copy data from GPU to host memory
    void copyToHost(void* hostData, size_t bytes) const;

    // Reshape tensor (only changes metadata, buffer must be compatible)
    void reshape(const Shape& newShape);

private:
    vkcompute::Context* ctx_ = nullptr;
    vkcompute::Buffer buffer_;
    Shape shape_;
    DataType dtype_ = DataType::UNDEFINED;
    bool pinned_ = false;

    // Staging buffer for CPU-GPU transfers (created on demand)
    mutable std::unique_ptr<vkcompute::Buffer> stagingBuffer_;
    void ensureStagingBuffer(size_t size) const;
};

} // namespace onnxrt
