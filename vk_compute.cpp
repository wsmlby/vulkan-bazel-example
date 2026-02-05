#include "vk_compute.hpp"
#include <iostream>
#include <algorithm>
#include <stdexcept>
#include <fstream>
#include <cstring>
#include <chrono>

namespace vkcompute {

// Error checking utility - throws exception if Vulkan call fails
void check(VkResult result, const char* msg) {
    if (result != VK_SUCCESS) throw std::runtime_error(msg);
}

// ============================================================================
// Context: Vulkan instance, device, and resource management
// ============================================================================

Context::Context(const std::string& preferredDevice) {
    // Create Vulkan instance - minimal setup, no validation layers or extensions
    VkApplicationInfo appInfo{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    appInfo.apiVersion = VK_API_VERSION_1_0;

    VkInstanceCreateInfo instanceInfo{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    instanceInfo.pApplicationInfo = &appInfo;
    check(vkCreateInstance(&instanceInfo, nullptr, &instance), "Failed to create instance");

    // Enumerate all physical devices
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
    if (deviceCount == 0) throw std::runtime_error("No Vulkan devices found");

    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

    // Helper for case-insensitive device name matching
    auto toLower = [](std::string s) {
        for (char& c : s) c = std::tolower(static_cast<unsigned char>(c));
        return s;
    };
    std::string filter = toLower(preferredDevice);

    // Find a device with a compute queue that matches the filter (if provided)
    for (auto& dev : devices) {
        uint32_t queueCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(dev, &queueCount, nullptr);
        std::vector<VkQueueFamilyProperties> queueProps(queueCount);
        vkGetPhysicalDeviceQueueFamilyProperties(dev, &queueCount, queueProps.data());

        // Look for a queue family with compute capability
        bool hasCompute = false;
        uint32_t computeQueueFamily = 0;
        for (uint32_t i = 0; i < queueCount; i++) {
            if (queueProps[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                hasCompute = true;
                computeQueueFamily = i;
                break;
            }
        }

        if (hasCompute) {
            VkPhysicalDeviceProperties props;
            vkGetPhysicalDeviceProperties(dev, &props);
            std::string name = props.deviceName;
            std::cout << "Compatible device: " << name << std::endl;

            // Select first matching device
            if (!physicalDevice) {
                if (filter.empty() || toLower(name).find(filter) != std::string::npos) {
                    physicalDevice = dev;
                    queueFamily = computeQueueFamily;
                }
            }
        }
    }
    if (!physicalDevice) {
        if (filter.empty()) throw std::runtime_error("No compute-capable device found");
        else throw std::runtime_error("No compute-capable device found matching filter: " + preferredDevice);
    }

    // Query memory properties for later memory allocation
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProps);

    // Create logical device with one compute queue
    float priority = 1.0f;
    VkDeviceQueueCreateInfo queueInfo{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queueInfo.queueFamilyIndex = queueFamily;
    queueInfo.queueCount = 1;
    queueInfo.pQueuePriorities = &priority;

    VkDeviceCreateInfo deviceInfo{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    deviceInfo.queueCreateInfoCount = 1;
    deviceInfo.pQueueCreateInfos = &queueInfo;
    check(vkCreateDevice(physicalDevice, &deviceInfo, nullptr, &device), "Failed to create device");

    vkGetDeviceQueue(device, queueFamily, 0, &queue);

    // Create command pool for allocating command buffers
    VkCommandPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    poolInfo.queueFamilyIndex = queueFamily;
    check(vkCreateCommandPool(device, &poolInfo, nullptr, &cmdPool), "Failed to create command pool");
    
    // Create reusable transfer command buffer (used by Buffer::copyFrom)
    VkCommandBufferAllocateInfo cmdAllocInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cmdAllocInfo.commandPool = cmdPool;
    cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdAllocInfo.commandBufferCount = 1;
    check(vkAllocateCommandBuffers(device, &cmdAllocInfo, &transferCmd), "Failed to allocate transfer cmd buffer");
    
    // Create reusable transfer fence (used by Buffer::copyFrom for synchronization)
    VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    check(vkCreateFence(device, &fenceInfo, nullptr, &transferFence), "Failed to create transfer fence");
}

Context::~Context() {
    // Clean up in reverse order of creation
    if (device) {
        if (transferFence) vkDestroyFence(device, transferFence, nullptr);
        if (transferCmd) vkFreeCommandBuffers(device, cmdPool, 1, &transferCmd);
        vkDestroyCommandPool(device, cmdPool, nullptr);
        vkDestroyDevice(device, nullptr);
    }
    if (instance) vkDestroyInstance(instance, nullptr);
}

std::string Context::deviceName() const {
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(physicalDevice, &props);
    return props.deviceName;
}

void Context::printLimits() const {
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(physicalDevice, &props);
    
    std::cout << "Device: " << props.deviceName << "\n";
    std::cout << "Max memory allocations: " << props.limits.maxMemoryAllocationCount << "\n";
    std::cout << "Max buffer size: " << (props.limits.maxStorageBufferRange / 1024 / 1024) << " MB\n";
    std::cout << "Max compute shared memory: " << (props.limits.maxComputeSharedMemorySize / 1024) << " KB\n";
    std::cout << "Max workgroup invocations: " << props.limits.maxComputeWorkGroupInvocations << "\n";
    std::cout << "Max workgroup size: [" 
              << props.limits.maxComputeWorkGroupSize[0] << ", "
              << props.limits.maxComputeWorkGroupSize[1] << ", "
              << props.limits.maxComputeWorkGroupSize[2] << "]\n";
}

// Find a memory type that satisfies the required properties
// typeBits: bitmask of suitable memory types from VkMemoryRequirements
// props: required memory properties (e.g., device-local, host-visible)
uint32_t Context::findMemoryType(uint32_t typeBits, VkMemoryPropertyFlags props) const {
    for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
        if ((typeBits & (1 << i)) && (memProps.memoryTypes[i].propertyFlags & props) == props) {
            return i;
        }
    }
    throw std::runtime_error("Failed to find suitable memory type");
}

// ============================================================================
// Buffer: GPU memory allocation and management
// ============================================================================

Buffer::Buffer(Context& context, VkDeviceSize bufferSize, VkBufferUsageFlags usage,
               VkMemoryPropertyFlags memProps) : size(bufferSize), propertyFlags(memProps), ctx(&context) {

    // Create Vulkan buffer object
    VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    check(vkCreateBuffer(ctx->device, &bufferInfo, nullptr, &buffer), "Failed to create buffer");

    // Determine memory requirements
    VkMemoryRequirements memReq;
    vkGetBufferMemoryRequirements(ctx->device, buffer, &memReq);

    // Allocate device memory with appropriate properties
    VkMemoryAllocateInfo allocInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocInfo.allocationSize = memReq.size;
    allocInfo.memoryTypeIndex = ctx->findMemoryType(memReq.memoryTypeBits, memProps);
    check(vkAllocateMemory(ctx->device, &allocInfo, nullptr, &memory), "Failed to allocate memory");
    check(vkBindBufferMemory(ctx->device, buffer, memory, 0), "Failed to bind buffer memory");

    // Map memory if it's host-visible (for pinned buffers)
    if (propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
        check(vkMapMemory(ctx->device, memory, 0, size, 0, &mapped), "Failed to map memory");
    }
}

Buffer::~Buffer() {
    if (ctx && ctx->device) {
        if (mapped) vkUnmapMemory(ctx->device, memory);
        vkDestroyBuffer(ctx->device, buffer, nullptr);
        vkFreeMemory(ctx->device, memory, nullptr);
    }
}

// Move constructor - transfer ownership of GPU resources
Buffer::Buffer(Buffer&& o) noexcept 
    : buffer(o.buffer), memory(o.memory), size(o.size), 
      propertyFlags(o.propertyFlags), mapped(o.mapped), ctx(o.ctx) {
    o.buffer = VK_NULL_HANDLE;
    o.memory = VK_NULL_HANDLE;
    o.mapped = nullptr;
    o.ctx = nullptr;
}

// Move assignment - swap resources to leverage RAII cleanup
Buffer& Buffer::operator=(Buffer&& o) noexcept {
    if (this != &o) { 
        std::swap(buffer, o.buffer); 
        std::swap(memory, o.memory);
        std::swap(size, o.size); 
        std::swap(propertyFlags, o.propertyFlags);
        std::swap(mapped, o.mapped);
        std::swap(ctx, o.ctx); 
    }
    return *this;
}

// Perform GPU-to-GPU buffer copy using the context's transfer command buffer
void Buffer::copyFrom(const Buffer& src, VkDeviceSize srcOffset, VkDeviceSize dstOffset, VkDeviceSize copySize) {
    if (copySize == VK_WHOLE_SIZE) copySize = src.size;

    // Reuse the context's transfer command buffer for efficiency
    vkResetCommandBuffer(ctx->transferCmd, 0);

    VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(ctx->transferCmd, &beginInfo);

    VkBufferCopy copyRegion{};
    copyRegion.srcOffset = srcOffset;
    copyRegion.dstOffset = dstOffset;
    copyRegion.size = copySize;
    vkCmdCopyBuffer(ctx->transferCmd, src.buffer, buffer, 1, &copyRegion);

    vkEndCommandBuffer(ctx->transferCmd);

    // Submit and synchronize using the context's transfer fence
    vkResetFences(ctx->device, 1, &ctx->transferFence);

    VkSubmitInfo submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &ctx->transferCmd;

    vkQueueSubmit(ctx->queue, 1, &submitInfo, ctx->transferFence);
    vkWaitForFences(ctx->device, 1, &ctx->transferFence, VK_TRUE, UINT64_MAX);
}

// Factory: Create device-local buffer (fast GPU memory, not CPU-accessible)
Buffer createDeviceBuffer(Context& ctx, VkDeviceSize size) {
    return Buffer(ctx, size, 
                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
}

// Factory: Create pinned buffer (CPU-accessible, host-visible and coherent)
Buffer createPinnedBuffer(Context& ctx, VkDeviceSize size) {
    return Buffer(ctx, size, 
                  VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
}

// ============================================================================
// ComputePipeline: Shader loading and pipeline creation
// ============================================================================

ComputePipeline::ComputePipeline(Context& context, const std::string& spvPath, uint32_t bufferCount,
                                 uint32_t workgroupSize, uint32_t pushConstantSize)
    : ctx(&context), pushConstantSize_(pushConstantSize) {

    // Load compiled SPIR-V shader from file
    std::ifstream file(spvPath, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("Failed to open shader: " + spvPath);
    size_t fileSize = file.tellg();
    std::vector<uint32_t> code(fileSize / 4);
    file.seekg(0);
    file.read(reinterpret_cast<char*>(code.data()), fileSize);

    // Create shader module and measure compilation time
    VkShaderModuleCreateInfo moduleInfo{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    moduleInfo.codeSize = code.size() * 4;
    moduleInfo.pCode = code.data();
    auto compileStart = std::chrono::high_resolution_clock::now();
    check(vkCreateShaderModule(ctx->device, &moduleInfo, nullptr, &shaderModule),
          "Failed to create shader module");
    auto compileEnd = std::chrono::high_resolution_clock::now();
    double compileTimeMs = std::chrono::duration<double, std::milli>(compileEnd - compileStart).count();
    std::cout << "Shader compile time: " << compileTimeMs << " ms" << std::endl;

    // Create descriptor set layout with N storage buffer bindings
    std::vector<VkDescriptorSetLayoutBinding> bindings(bufferCount);
    for (uint32_t i = 0; i < bufferCount; i++) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    VkDescriptorSetLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layoutInfo.bindingCount = bufferCount;
    layoutInfo.pBindings = bindings.data();
    check(vkCreateDescriptorSetLayout(ctx->device, &layoutInfo, nullptr, &descLayout),
          "Failed to create descriptor layout");

    // Create pipeline layout with optional push constants
    VkPushConstantRange pushConstantRange{};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushConstantRange.offset = 0;
    pushConstantRange.size = pushConstantSize_;

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &descLayout;
    if (pushConstantSize_ > 0) {
        pipelineLayoutInfo.pushConstantRangeCount = 1;
        pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;
        pushConstantData_.resize(pushConstantSize_);
    }
    check(vkCreatePipelineLayout(ctx->device, &pipelineLayoutInfo, nullptr, &pipelineLayout),
          "Failed to create pipeline layout");

    // Set up specialization constant for workgroup size
    // This allows the shader to use the workgroup size specified at pipeline creation
    VkSpecializationMapEntry specEntry{};
    specEntry.constantID = 0;
    specEntry.offset = 0;
    specEntry.size = sizeof(uint32_t);

    VkSpecializationInfo specInfo{};
    specInfo.mapEntryCount = 1;
    specInfo.pMapEntries = &specEntry;
    specInfo.dataSize = sizeof(uint32_t);
    specInfo.pData = &workgroupSize;

    // Create compute pipeline
    VkPipelineShaderStageCreateInfo stageInfo{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stageInfo.module = shaderModule;
    stageInfo.pName = "main";
    stageInfo.pSpecializationInfo = &specInfo;

    VkComputePipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipelineInfo.stage = stageInfo;
    pipelineInfo.layout = pipelineLayout;
    check(vkCreateComputePipelines(ctx->device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline),
          "Failed to create compute pipeline");

    // Create descriptor pool to allocate descriptor sets from
    VkDescriptorPoolSize poolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, bufferCount};
    VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    poolInfo.maxSets = 1;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;
    check(vkCreateDescriptorPool(ctx->device, &poolInfo, nullptr, &descPool),
          "Failed to create descriptor pool");

    // Allocate descriptor set (will be populated later via bindBuffers)
    VkDescriptorSetAllocateInfo allocInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocInfo.descriptorPool = descPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &descLayout;
    check(vkAllocateDescriptorSets(ctx->device, &allocInfo, &descSet),
          "Failed to allocate descriptor set");
}

ComputePipeline::~ComputePipeline() {
    // Clean up Vulkan resources in reverse order of dependencies
    if (ctx && ctx->device) {
        vkDestroyPipeline(ctx->device, pipeline, nullptr);
        vkDestroyPipelineLayout(ctx->device, pipelineLayout, nullptr);
        vkDestroyDescriptorPool(ctx->device, descPool, nullptr);  // Also frees descriptor sets
        vkDestroyDescriptorSetLayout(ctx->device, descLayout, nullptr);
        vkDestroyShaderModule(ctx->device, shaderModule, nullptr);
    }
}

// Move constructor - transfer ownership of pipeline resources
ComputePipeline::ComputePipeline(ComputePipeline&& o) noexcept
    : ctx(o.ctx), shaderModule(o.shaderModule), descLayout(o.descLayout),
      pipelineLayout(o.pipelineLayout), pipeline(o.pipeline),
      descPool(o.descPool), descSet(o.descSet),
      pushConstantSize_(o.pushConstantSize_), pushConstantData_(std::move(o.pushConstantData_)) {
    o.ctx = nullptr;
    o.shaderModule = VK_NULL_HANDLE;
    o.descLayout = VK_NULL_HANDLE;
    o.pipelineLayout = VK_NULL_HANDLE;
    o.pipeline = VK_NULL_HANDLE;
    o.descPool = VK_NULL_HANDLE;
    o.descSet = VK_NULL_HANDLE;
    o.pushConstantSize_ = 0;
}

// Bind buffers to the descriptor set
// Buffers must be in the same order as the shader's layout bindings
void ComputePipeline::bindBuffers(const std::vector<Buffer*>& buffers) {
    std::vector<VkDescriptorBufferInfo> bufferInfos(buffers.size());
    std::vector<VkWriteDescriptorSet> writes(buffers.size());

    for (size_t i = 0; i < buffers.size(); i++) {
        bufferInfos[i].buffer = buffers[i]->buffer;
        bufferInfos[i].offset = 0;
        bufferInfos[i].range = buffers[i]->size;

        writes[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        writes[i].dstSet = descSet;
        writes[i].dstBinding = static_cast<uint32_t>(i);
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &bufferInfos[i];
    }
    vkUpdateDescriptorSets(ctx->device, static_cast<uint32_t>(writes.size()),
                           writes.data(), 0, nullptr);
}

// Set push constant data (stored internally, will be pushed during recordTo)
void ComputePipeline::setPushConstants(const void* data, uint32_t size) {
    if (size > pushConstantSize_) {
        throw std::runtime_error("Push constant data exceeds allocated size");
    }
    std::memcpy(pushConstantData_.data(), data, size);
}

// Record pipeline bind and dispatch commands to a command buffer
void ComputePipeline::recordTo(VkCommandBuffer cmd, uint32_t groupCountX, uint32_t groupCountY, uint32_t groupCountZ) {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout,
                            0, 1, &descSet, 0, nullptr);
    // Push constants if configured
    if (pushConstantSize_ > 0) {
        vkCmdPushConstants(cmd, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, pushConstantSize_, pushConstantData_.data());
    }
    vkCmdDispatch(cmd, groupCountX, groupCountY, groupCountZ);
}

// Insert a memory barrier to ensure shader writes complete before reads
void ComputePipeline::barrier(VkCommandBuffer cmd) {
    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd, 
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 1, &barrier, 0, nullptr, 0, nullptr);
}

// ============================================================================
// Sequence: Command buffer recording and submission
// ============================================================================

Sequence::Sequence(Context& context) : ctx(&context) {
    // Allocate command buffer for recording compute operations
    VkCommandBufferAllocateInfo allocInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    allocInfo.commandPool = ctx->cmdPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;
    check(vkAllocateCommandBuffers(ctx->device, &allocInfo, &cmd_), "Failed to allocate cmd buffer");

    // Create fence for CPU-GPU synchronization
    VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    check(vkCreateFence(ctx->device, &fenceInfo, nullptr, &fence), "Failed to create fence");
}

Sequence::~Sequence() {
    if (ctx && ctx->device) {
        if (cmd_) vkFreeCommandBuffers(ctx->device, ctx->cmdPool, 1, &cmd_);
        if (fence) vkDestroyFence(ctx->device, fence, nullptr);
    }
}

// Begin recording commands into the command buffer
void Sequence::begin(bool reusable) {
    vkResetCommandBuffer(cmd_, 0);
    VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    // ONE_TIME_SUBMIT is a hint that the buffer will only be submitted once
    // Omit it if we want to reuse the recorded commands multiple times
    if (!reusable) {
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    }
    check(vkBeginCommandBuffer(cmd_, &beginInfo), "Failed to begin cmd buffer");
}

// Finish recording commands
void Sequence::end() {
    check(vkEndCommandBuffer(cmd_), "Failed to end cmd buffer");
}

// Submit commands and wait for completion (no timing)
void Sequence::submit() {
    submitAndWait();
}

// Submit commands, wait for completion, and return execution time in milliseconds
double Sequence::submitAndWait() {
    vkResetFences(ctx->device, 1, &fence);

    VkSubmitInfo submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd_;

    // Measure GPU execution time using CPU timestamps (includes queue latency)
    auto start = std::chrono::high_resolution_clock::now();
    check(vkQueueSubmit(ctx->queue, 1, &submitInfo, fence), "Failed to submit queue");
    check(vkWaitForFences(ctx->device, 1, &fence, VK_TRUE, UINT64_MAX), "Failed to wait for fence");
    auto end = std::chrono::high_resolution_clock::now();
    
    return std::chrono::duration<double, std::milli>(end - start).count();
}

// Record a compute dispatch (delegates to pipeline's recordTo method)
void Sequence::record(ComputePipeline& pipeline, uint32_t x, uint32_t y, uint32_t z) {
    pipeline.recordTo(cmd_, x, y, z);
}

// Record a memory barrier (delegates to pipeline's static barrier method)
void Sequence::barrier() {
    ComputePipeline::barrier(cmd_);
}

} // namespace vkcompute