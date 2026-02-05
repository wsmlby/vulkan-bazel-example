#include "tensor.hpp"
#include <cstring>
#include <stdexcept>
#include <memory>

namespace onnxrt {

Tensor::Tensor(vkcompute::Context& ctx, const Shape& shape, DataType dtype, bool pinned)
    : ctx_(&ctx), shape_(shape), dtype_(dtype), pinned_(pinned) {

    size_t size = byteSize();
    if (size == 0) {
        throw std::runtime_error("Cannot create zero-size tensor");
    }

    if (pinned_) {
        buffer_ = vkcompute::createPinnedBuffer(ctx, size);
    } else {
        buffer_ = vkcompute::createDeviceBuffer(ctx, size);
    }
}

Tensor::Tensor(Tensor&& o) noexcept
    : ctx_(o.ctx_), buffer_(std::move(o.buffer_)), shape_(std::move(o.shape_)),
      dtype_(o.dtype_), pinned_(o.pinned_), stagingBuffer_(std::move(o.stagingBuffer_)) {
    o.ctx_ = nullptr;
    o.dtype_ = DataType::UNDEFINED;
    o.pinned_ = false;
}

Tensor& Tensor::operator=(Tensor&& o) noexcept {
    if (this != &o) {
        ctx_ = o.ctx_;
        buffer_ = std::move(o.buffer_);
        shape_ = std::move(o.shape_);
        dtype_ = o.dtype_;
        pinned_ = o.pinned_;
        stagingBuffer_ = std::move(o.stagingBuffer_);
        o.ctx_ = nullptr;
        o.dtype_ = DataType::UNDEFINED;
        o.pinned_ = false;
    }
    return *this;
}

void Tensor::ensureStagingBuffer(size_t size) const {
    if (!stagingBuffer_ || stagingBuffer_->size < size) {
        stagingBuffer_ = std::make_unique<vkcompute::Buffer>(
            vkcompute::createPinnedBuffer(*ctx_, size));
    }
}

void Tensor::copyFromHost(const void* hostData, size_t bytes) {
    if (bytes > byteSize()) {
        throw std::runtime_error("Copy size exceeds tensor size");
    }

    if (pinned_) {
        // Direct copy for pinned memory
        std::memcpy(buffer_.mapped, hostData, bytes);
    } else {
        // Use staging buffer for device-local memory
        ensureStagingBuffer(bytes);
        std::memcpy(stagingBuffer_->mapped, hostData, bytes);
        buffer_.copyFrom(*stagingBuffer_, 0, 0, bytes);
    }
}

void Tensor::copyToHost(void* hostData, size_t bytes) const {
    if (bytes > byteSize()) {
        throw std::runtime_error("Copy size exceeds tensor size");
    }

    if (pinned_) {
        // Direct copy for pinned memory
        std::memcpy(hostData, buffer_.mapped, bytes);
    } else {
        // Use staging buffer for device-local memory
        ensureStagingBuffer(bytes);
        // Need to cast away const for the copy operation
        const_cast<vkcompute::Buffer*>(&*stagingBuffer_)->copyFrom(buffer_, 0, 0, bytes);
        std::memcpy(hostData, stagingBuffer_->mapped, bytes);
    }
}

void Tensor::reshape(const Shape& newShape) {
    size_t newSize = shapeSize(newShape);
    if (newSize != elementCount()) {
        throw std::runtime_error("Cannot reshape tensor: element count mismatch (was " +
                                 std::to_string(elementCount()) + ", new " +
                                 std::to_string(newSize) + ")");
    }
    shape_ = newShape;
}

} // namespace onnxrt
