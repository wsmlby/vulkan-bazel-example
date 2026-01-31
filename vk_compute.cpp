#include "vk_compute.hpp"
#include <iostream>
#include <algorithm>
#include <stdexcept>
#include <fstream>
#include <cstring>
#include <chrono>

namespace vkcompute {

void check(VkResult result, const char* msg) {
    if (result != VK_SUCCESS) throw std::runtime_error(msg);
}

Context::Context(const std::string& preferredDevice) {
    // Instance - minimal, no extensions needed for compute-only
    VkApplicationInfo appInfo{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    appInfo.apiVersion = VK_API_VERSION_1_0;

    VkInstanceCreateInfo instanceInfo{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    instanceInfo.pApplicationInfo = &appInfo;
    check(vkCreateInstance(&instanceInfo, nullptr, &instance), "Failed to create instance");

    // Pick first device with compute queue
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
    if (deviceCount == 0) throw std::runtime_error("No Vulkan devices found");

    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

    auto toLower = [](std::string s) {
        for (char& c : s) c = std::tolower(static_cast<unsigned char>(c));
        return s;
    };
    std::string filter = toLower(preferredDevice);

    for (auto& dev : devices) {
        uint32_t queueCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(dev, &queueCount, nullptr);
        std::vector<VkQueueFamilyProperties> queueProps(queueCount);
        vkGetPhysicalDeviceQueueFamilyProperties(dev, &queueCount, queueProps.data());

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

    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProps);

    // Create logical device
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

    // Command pool
    VkCommandPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    poolInfo.queueFamilyIndex = queueFamily;
    check(vkCreateCommandPool(device, &poolInfo, nullptr, &cmdPool), "Failed to create command pool");
}

Context::~Context() {
    if (device) {
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

uint32_t Context::findMemoryType(uint32_t typeBits, VkMemoryPropertyFlags props) const {
    for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
        if ((typeBits & (1 << i)) && (memProps.memoryTypes[i].propertyFlags & props) == props) {
            return i;
        }
    }
    throw std::runtime_error("Failed to find suitable memory type");
}

Buffer::Buffer(Context& context, VkDeviceSize bufferSize, VkBufferUsageFlags usage,
               VkMemoryPropertyFlags memProps) : size(bufferSize), propertyFlags(memProps), ctx(&context) {

    VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    check(vkCreateBuffer(ctx->device, &bufferInfo, nullptr, &buffer), "Failed to create buffer");

    VkMemoryRequirements memReq;
    vkGetBufferMemoryRequirements(ctx->device, buffer, &memReq);

    VkMemoryAllocateInfo allocInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocInfo.allocationSize = memReq.size;
    allocInfo.memoryTypeIndex = ctx->findMemoryType(memReq.memoryTypeBits, memProps);
    check(vkAllocateMemory(ctx->device, &allocInfo, nullptr, &memory), "Failed to allocate memory");
    check(vkBindBufferMemory(ctx->device, buffer, memory, 0), "Failed to bind buffer memory");

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

Buffer::Buffer(Buffer&& o) noexcept : buffer(o.buffer), memory(o.memory), size(o.size), propertyFlags(o.propertyFlags), mapped(o.mapped), ctx(o.ctx) {
    o.buffer = VK_NULL_HANDLE; o.memory = VK_NULL_HANDLE; o.mapped = nullptr; o.ctx = nullptr;
}

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

void Buffer::upload(const void* data, VkDeviceSize dataSize) {
    if (mapped) {
        memcpy(mapped, data, dataSize);
    } else {
        Buffer staging(*ctx, dataSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        memcpy(staging.mapped, data, dataSize);
        copyFrom(staging, 0, 0, dataSize);
    }
}

void Buffer::download(void* data, VkDeviceSize dataSize) {
    if (mapped) {
        memcpy(data, mapped, dataSize);
    } else {
        Buffer staging(*ctx, dataSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        staging.copyFrom(*this, 0, 0, dataSize);
        memcpy(data, staging.mapped, dataSize);
    }
}

void Buffer::copyFrom(const Buffer& src, VkDeviceSize srcOffset, VkDeviceSize dstOffset, VkDeviceSize copySize) {
    if (copySize == VK_WHOLE_SIZE) copySize = src.size;

    VkCommandBufferAllocateInfo allocInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    allocInfo.commandPool = ctx->cmdPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer cmdBuf;
    check(vkAllocateCommandBuffers(ctx->device, &allocInfo, &cmdBuf), "Failed to allocate cmd buffer");

    VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmdBuf, &beginInfo);

    VkBufferCopy copyRegion{};
    copyRegion.srcOffset = srcOffset;
    copyRegion.dstOffset = dstOffset;
    copyRegion.size = copySize;
    vkCmdCopyBuffer(cmdBuf, src.buffer, buffer, 1, &copyRegion);

    vkEndCommandBuffer(cmdBuf);

    VkSubmitInfo submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmdBuf;

    vkQueueSubmit(ctx->queue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(ctx->queue);

    vkFreeCommandBuffers(ctx->device, ctx->cmdPool, 1, &cmdBuf);
}

Buffer createDeviceBuffer(Context& ctx, VkDeviceSize size) {
    return Buffer(ctx, size, 
                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
}

Buffer createPinnedBuffer(Context& ctx, VkDeviceSize size) {
    return Buffer(ctx, size, 
                  VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
}

ComputePipeline::ComputePipeline(Context& context, const std::string& spvPath, uint32_t bufferCount)
    : ctx(&context) {

    // Load SPIR-V
    std::ifstream file(spvPath, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("Failed to open shader: " + spvPath);
    size_t fileSize = file.tellg();
    std::vector<uint32_t> code(fileSize / 4);
    file.seekg(0);
    file.read(reinterpret_cast<char*>(code.data()), fileSize);

    VkShaderModuleCreateInfo moduleInfo{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    moduleInfo.codeSize = code.size() * 4;
    moduleInfo.pCode = code.data();
    auto compileStart = std::chrono::high_resolution_clock::now();
    check(vkCreateShaderModule(ctx->device, &moduleInfo, nullptr, &shaderModule),
          "Failed to create shader module");
    auto compileEnd = std::chrono::high_resolution_clock::now();
    double compileTimeMs = std::chrono::duration<double, std::milli>(compileEnd - compileStart).count();
    std::cout << "Shader compile time: " << compileTimeMs << " ms" << std::endl;

    // Descriptor layout - N storage buffers
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

    // Pipeline layout
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &descLayout;
    check(vkCreatePipelineLayout(ctx->device, &pipelineLayoutInfo, nullptr, &pipelineLayout),
          "Failed to create pipeline layout");

    // Compute pipeline
    VkPipelineShaderStageCreateInfo stageInfo{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stageInfo.module = shaderModule;
    stageInfo.pName = "main";

    VkComputePipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipelineInfo.stage = stageInfo;
    pipelineInfo.layout = pipelineLayout;
    check(vkCreateComputePipelines(ctx->device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline),
          "Failed to create compute pipeline");

    // Descriptor pool and set
    VkDescriptorPoolSize poolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, bufferCount};
    VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    poolInfo.maxSets = 1;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;
    check(vkCreateDescriptorPool(ctx->device, &poolInfo, nullptr, &descPool),
          "Failed to create descriptor pool");

    VkDescriptorSetAllocateInfo allocInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocInfo.descriptorPool = descPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &descLayout;
    check(vkAllocateDescriptorSets(ctx->device, &allocInfo, &descSet),
          "Failed to allocate descriptor set");

    // Allocate command buffer once
    VkCommandBufferAllocateInfo cmdAllocInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cmdAllocInfo.commandPool = ctx->cmdPool;
    cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdAllocInfo.commandBufferCount = 1;
    check(vkAllocateCommandBuffers(ctx->device, &cmdAllocInfo, &commandBuffer),
          "Failed to allocate command buffer");
}

ComputePipeline::~ComputePipeline() {
    if (ctx && ctx->device) {
        if (commandBuffer) vkFreeCommandBuffers(ctx->device, ctx->cmdPool, 1, &commandBuffer);
        vkDestroyPipeline(ctx->device, pipeline, nullptr);
        vkDestroyPipelineLayout(ctx->device, pipelineLayout, nullptr);
        vkDestroyDescriptorPool(ctx->device, descPool, nullptr);
        vkDestroyDescriptorSetLayout(ctx->device, descLayout, nullptr);
        vkDestroyShaderModule(ctx->device, shaderModule, nullptr);
    }
}

ComputePipeline::ComputePipeline(ComputePipeline&& o) noexcept
    : ctx(o.ctx), shaderModule(o.shaderModule), descLayout(o.descLayout),
      pipelineLayout(o.pipelineLayout), pipeline(o.pipeline),
      descPool(o.descPool), descSet(o.descSet) {
    o.ctx = nullptr; o.shaderModule = VK_NULL_HANDLE; o.descLayout = VK_NULL_HANDLE;
    o.pipelineLayout = VK_NULL_HANDLE; o.pipeline = VK_NULL_HANDLE;
    o.descPool = VK_NULL_HANDLE; o.descSet = VK_NULL_HANDLE;
}

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

void ComputePipeline::recordTo(VkCommandBuffer cmd, uint32_t groupCountX, uint32_t groupCountY, uint32_t groupCountZ) {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout,
                            0, 1, &descSet, 0, nullptr);
    vkCmdDispatch(cmd, groupCountX, groupCountY, groupCountZ);
}

void ComputePipeline::barrier(VkCommandBuffer cmd) {
    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd, 
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 1, &barrier, 0, nullptr, 0, nullptr);
}

void ComputePipeline::dispatch(uint32_t groupCountX, uint32_t groupCountY, uint32_t groupCountZ) {
    vkResetCommandBuffer(commandBuffer, 0);

    VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(commandBuffer, &beginInfo);

    recordTo(commandBuffer, groupCountX, groupCountY, groupCountZ);

    vkEndCommandBuffer(commandBuffer);

    VkSubmitInfo submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    VkFence fence;
    VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    vkCreateFence(ctx->device, &fenceInfo, nullptr, &fence);

    vkQueueSubmit(ctx->queue, 1, &submitInfo, fence);
    vkWaitForFences(ctx->device, 1, &fence, VK_TRUE, UINT64_MAX);

    vkDestroyFence(ctx->device, fence, nullptr);
}

Sequence::Sequence(Context& context) : ctx(&context) {
    VkCommandBufferAllocateInfo allocInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    allocInfo.commandPool = ctx->cmdPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;
    check(vkAllocateCommandBuffers(ctx->device, &allocInfo, &cmd), "Failed to allocate cmd buffer");

    VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    check(vkCreateFence(ctx->device, &fenceInfo, nullptr, &fence), "Failed to create fence");
}

Sequence::~Sequence() {
    if (ctx && ctx->device) {
        if (cmd) vkFreeCommandBuffers(ctx->device, ctx->cmdPool, 1, &cmd);
        if (fence) vkDestroyFence(ctx->device, fence, nullptr);
    }
}

void Sequence::begin() {
    vkResetCommandBuffer(cmd, 0);
    VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    check(vkBeginCommandBuffer(cmd, &beginInfo), "Failed to begin cmd buffer");
}

void Sequence::end() {
    check(vkEndCommandBuffer(cmd), "Failed to end cmd buffer");
}

void Sequence::submit() {
    submitAndWait();
}

double Sequence::submitAndWait() {
    vkResetFences(ctx->device, 1, &fence);

    VkSubmitInfo submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;

    auto start = std::chrono::high_resolution_clock::now();
    check(vkQueueSubmit(ctx->queue, 1, &submitInfo, fence), "Failed to submit queue");
    check(vkWaitForFences(ctx->device, 1, &fence, VK_TRUE, UINT64_MAX), "Failed to wait for fence");
    auto end = std::chrono::high_resolution_clock::now();
    
    return std::chrono::duration<double, std::milli>(end - start).count();
}

void Sequence::record(ComputePipeline& pipeline, uint32_t x, uint32_t y, uint32_t z, bool autoBarrier) {
    pipeline.recordTo(cmd, x, y, z);
    if (autoBarrier) {
        ComputePipeline::barrier(cmd);
    }
}

void Sequence::barrier() {
    ComputePipeline::barrier(cmd);
}

} // namespace vkcompute
