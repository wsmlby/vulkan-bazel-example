#pragma once
// Minimal Vulkan Compute API - Simple abstraction for GPU compute operations
// Provides easy-to-use wrappers for Vulkan compute shaders without graphics complexity

#include <vulkan/vulkan.h>
#include <vector>
#include <string>

namespace vkcompute {

// Utility function to check Vulkan result codes and throw on error
void check(VkResult result, const char* msg);

/**
 * Context - Manages Vulkan instance, device, and command resources
 * 
 * This is the main entry point for the API. Create one Context per application.
 * The Context automatically selects a compute-capable GPU and sets up command pools.
 * 
 * Example:
 *   Context ctx("NVIDIA");  // Prefer NVIDIA GPU
 *   std::cout << ctx.deviceName() << std::endl;
 */
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

    // Timestamp support
    bool timestampsSupported = false;
    float timestampPeriod = 1.0f; // nanoseconds per tick

    // Create a Vulkan context, optionally filtering devices by name (e.g., "NVIDIA", "AMD", "Intel")
    Context(const std::string& preferredDevice = "");
    ~Context();

    // Get the name of the selected GPU device
    std::string deviceName() const;
    
    // Print GPU capabilities including memory limits and max allocations
    void printLimits() const;
    
    // Find a suitable memory type index for the given requirements
    uint32_t findMemoryType(uint32_t typeBits, VkMemoryPropertyFlags props) const;
};

/**
 * Buffer - GPU memory allocation with optional CPU mapping
 * 
 * Represents a Vulkan buffer with associated device memory.
 * Use createDeviceBuffer() for fast GPU-only memory.
 * Use createPinnedBuffer() for CPU-accessible memory (mapped pointer available).
 * 
 * Buffers are move-only to prevent accidental copying of GPU resources.
 */
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

    // Move semantics (buffers are not copyable)
    Buffer(Buffer&& o) noexcept;
    Buffer& operator=(Buffer&& o) noexcept;
    
    // GPU-to-GPU copy from another buffer (efficient, no CPU involvement)
    // Use VK_WHOLE_SIZE for size parameter to copy entire buffer
    void copyFrom(const Buffer& src, VkDeviceSize srcOffset = 0, VkDeviceSize dstOffset = 0, VkDeviceSize size = VK_WHOLE_SIZE);
};

// Create a device-local buffer (fast GPU memory, not CPU-accessible)
// Best for intermediate results and main compute working memory
Buffer createDeviceBuffer(Context& ctx, VkDeviceSize size);

// Create a pinned (host-visible) buffer (CPU-accessible via .mapped pointer)
// Best for CPU-GPU data transfer; slower for GPU computation
Buffer createPinnedBuffer(Context& ctx, VkDeviceSize size);

/**
 * ComputePipeline - Compiled compute shader with buffer bindings
 *
 * Loads a SPIR-V shader and creates a Vulkan pipeline for execution.
 * Buffers are bound in the order they appear in the shader (layout binding 0, 1, 2, ...).
 *
 * Example:
 *   ComputePipeline pipeline(ctx, "shader.spv", 3, 256);  // 3 buffers, 256 threads per workgroup
 *   pipeline.bindBuffers({&bufA, &bufB, &bufC});
 *
 * With push constants:
 *   ComputePipeline pipeline(ctx, "shader.spv", 3, 256, sizeof(MyParams));
 *   pipeline.bindBuffers({&bufA, &bufB, &bufC});
 *   pipeline.setPushConstants(&params, sizeof(params));
 */
class ComputePipeline {
public:
    ComputePipeline() = default;

    // Create a compute pipeline from a SPIR-V shader file
    // bufferCount: number of buffers the shader uses
    // workgroupSize: local workgroup size (should match shader's local_size_x)
    // pushConstantSize: size of push constants in bytes (0 to disable)
    ComputePipeline(Context& context, const std::string& spvPath, uint32_t bufferCount,
                    uint32_t workgroupSize = 256, uint32_t pushConstantSize = 0);
    ~ComputePipeline();

    ComputePipeline(ComputePipeline&& o) noexcept;

    // Bind buffers to descriptor set (order must match shader layout bindings)
    void bindBuffers(const std::vector<Buffer*>& buffers);

    // Set push constant data (must be called before recordTo if using push constants)
    void setPushConstants(const void* data, uint32_t size);

    // Record dispatch commands to an external command buffer
    // groupCount parameters specify the number of workgroups in each dimension
    void recordTo(VkCommandBuffer cmd, uint32_t groupCountX, uint32_t groupCountY = 1, uint32_t groupCountZ = 1);

    // Insert a pipeline barrier for memory synchronization
    static void barrier(VkCommandBuffer cmd);

private:
    Context* ctx = nullptr;
    VkShaderModule shaderModule = VK_NULL_HANDLE;
    VkDescriptorSetLayout descLayout = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDescriptorPool descPool = VK_NULL_HANDLE;
    VkDescriptorSet descSet = VK_NULL_HANDLE;

    // Push constants support
    uint32_t pushConstantSize_ = 0;
    std::vector<uint8_t> pushConstantData_;
};

/**
 * Sequence - Command buffer for batching GPU operations
 * 
 * Records multiple compute dispatches and barriers into a single submission.
 * Allows efficient batching of operations and provides timing information.
 * 
 * Example:
 *   Sequence seq(ctx);
 *   seq.begin();
 *   seq.record(pipeline, numGroups);
 *   seq.barrier();
 *   seq.end();
 *   double timeMs = seq.submitAndWait();  // Execute and measure time
 */
class Sequence {
public:
    Sequence(Context& ctx);
    ~Sequence();

    // Start recording commands
    // Set reusable=true if you plan to submit the same commands multiple times
    void begin(bool reusable = false);
    
    // Finish recording commands (must be called before submit)
    void end();
    
    // Submit commands and wait for completion
    void submit();
    
    // Submit commands, wait for completion, and return execution time in milliseconds
    double submitAndWait();

    // Enable GPU timestamp queries for profiling (must be called before begin)
    void enableTimestamps(uint32_t queryCount);

    // Write a GPU timestamp into the query pool at the given index
    void writeTimestamp(uint32_t index);

    // Fetch timestamp query results (nanoseconds ticks)
    std::vector<uint64_t> fetchTimestamps() const;

    // Record a compute shader dispatch
    // groupCount parameters specify the number of workgroups in each dimension
    void record(ComputePipeline& pipeline, uint32_t groupCountX, uint32_t groupCountY = 1, uint32_t groupCountZ = 1);
    
    // Record a buffer copy operation
    void recordCopy(const Buffer& src, Buffer& dst, VkDeviceSize size = VK_WHOLE_SIZE);
    
    // Insert a memory barrier to ensure previous operations complete before continuing
    void barrier();
    
    // Insert a transfer-to-compute barrier (for after buffer copies)
    void transferBarrier();

    // Submit this sequence with a prefix command buffer, wait for completion, return time in ms
    // Useful for batching transfer commands with pre-recorded compute commands
    double submitWithPrefixAndWait(VkCommandBuffer prefixCmd);

    // Get underlying command buffer (for advanced use)
    VkCommandBuffer cmdBuffer() const { return cmd_; }

private:
    Context* ctx = nullptr;
    VkCommandBuffer cmd_ = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;

    // Timestamp query pool
    VkQueryPool queryPool_ = VK_NULL_HANDLE;
    uint32_t queryCount_ = 0;
    bool timestampsEnabled_ = false;
};

} // namespace vkcompute