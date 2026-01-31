#pragma once
// Minimal Vulkan Compute - header only declarations

#include <vulkan/vulkan.h>
#include <vector>
#include <string>

namespace vkcompute {

void check(VkResult result, const char* msg);

class Context {
public:
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    uint32_t queueFamily = 0;
    VkCommandPool cmdPool = VK_NULL_HANDLE;
    VkPhysicalDeviceMemoryProperties memProps;
    
    // Reusable transfer resources
    VkCommandBuffer transferCmd = VK_NULL_HANDLE;
    VkFence transferFence = VK_NULL_HANDLE;

    Context(const std::string& preferredDevice = "");
    ~Context();

    std::string deviceName() const;
    void printLimits() const;  // Print device limits including allocation count
    uint32_t findMemoryType(uint32_t typeBits, VkMemoryPropertyFlags props) const;
};

class Buffer {
public:
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    VkMemoryPropertyFlags propertyFlags = 0;
    void* mapped = nullptr;
    Context* ctx = nullptr;

    Buffer() = default;
    Buffer(Context& context, VkDeviceSize bufferSize, VkBufferUsageFlags usage,
           VkMemoryPropertyFlags memProps);
    ~Buffer();

    Buffer(Buffer&& o) noexcept;
    Buffer& operator=(Buffer&& o) noexcept;

    void upload(const void* data, VkDeviceSize dataSize);
    void download(void* data, VkDeviceSize dataSize);
    
    // Copy from another buffer (e.g., pinned memory)
    void copyFrom(const Buffer& src, VkDeviceSize srcOffset = 0, VkDeviceSize dstOffset = 0, VkDeviceSize size = VK_WHOLE_SIZE);
};

Buffer createDeviceBuffer(Context& ctx, VkDeviceSize size);
Buffer createPinnedBuffer(Context& ctx, VkDeviceSize size);

class ComputePipeline {
public:
    Context* ctx = nullptr;
    VkShaderModule shaderModule = VK_NULL_HANDLE;
    VkDescriptorSetLayout descLayout = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDescriptorPool descPool = VK_NULL_HANDLE;
    VkDescriptorSet descSet = VK_NULL_HANDLE;

    ComputePipeline() = default;
    ComputePipeline(Context& context, const std::string& spvPath, uint32_t bufferCount);
    ~ComputePipeline();

    ComputePipeline(ComputePipeline&& o) noexcept;

    void bindBuffers(const std::vector<Buffer*>& buffers);
    
    // Record commands to an external command buffer
    void recordTo(VkCommandBuffer cmd, uint32_t groupCountX, uint32_t groupCountY = 1, uint32_t groupCountZ = 1);
    
    static void barrier(VkCommandBuffer cmd);
};

class Sequence {
public:
    Context* ctx = nullptr;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;

    Sequence(Context& ctx);
    ~Sequence();

    void begin();
    void end();
    void submit(); // Submit and wait
    double submitAndWait(); // Returns execution time in ms

    void record(ComputePipeline& pipeline, uint32_t groupCountX, uint32_t groupCountY = 1, uint32_t groupCountZ = 1);
    void barrier();
};

} // namespace vkcompute
