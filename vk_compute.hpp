#pragma once
// Minimal Vulkan Compute - single header, no graphics

#include <vulkan/vulkan.h>
#include <vector>
#include <stdexcept>
#include <fstream>
#include <cstring>
#include <chrono>
#include <iostream>

namespace vkcompute {

inline void check(VkResult result, const char* msg) {
    if (result != VK_SUCCESS) throw std::runtime_error(msg);
}

class Context {
public:
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    uint32_t queueFamily = 0;
    VkCommandPool cmdPool = VK_NULL_HANDLE;
    VkPhysicalDeviceMemoryProperties memProps;

    Context() {
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

        for (auto& dev : devices) {
            uint32_t queueCount = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(dev, &queueCount, nullptr);
            std::vector<VkQueueFamilyProperties> queueProps(queueCount);
            vkGetPhysicalDeviceQueueFamilyProperties(dev, &queueCount, queueProps.data());

            for (uint32_t i = 0; i < queueCount; i++) {
                if (queueProps[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                    physicalDevice = dev;
                    queueFamily = i;
                    break;
                }
            }
            if (physicalDevice) break;
        }
        if (!physicalDevice) throw std::runtime_error("No compute-capable device found");

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

    ~Context() {
        if (device) {
            vkDestroyCommandPool(device, cmdPool, nullptr);
            vkDestroyDevice(device, nullptr);
        }
        if (instance) vkDestroyInstance(instance, nullptr);
    }

    std::string deviceName() const {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(physicalDevice, &props);
        return props.deviceName;
    }

    uint32_t findMemoryType(uint32_t typeBits, VkMemoryPropertyFlags props) const {
        for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
            if ((typeBits & (1 << i)) && (memProps.memoryTypes[i].propertyFlags & props) == props) {
                return i;
            }
        }
        throw std::runtime_error("Failed to find suitable memory type");
    }
};

class Buffer {
public:
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    Context* ctx = nullptr;

    Buffer() = default;
    Buffer(Context& context, VkDeviceSize bufferSize, VkBufferUsageFlags usage,
           VkMemoryPropertyFlags memProps) : size(bufferSize), ctx(&context) {

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
    }

    ~Buffer() {
        if (ctx && ctx->device) {
            vkDestroyBuffer(ctx->device, buffer, nullptr);
            vkFreeMemory(ctx->device, memory, nullptr);
        }
    }

    Buffer(Buffer&& o) noexcept : buffer(o.buffer), memory(o.memory), size(o.size), ctx(o.ctx) {
        o.buffer = VK_NULL_HANDLE; o.memory = VK_NULL_HANDLE; o.ctx = nullptr;
    }
    Buffer& operator=(Buffer&& o) noexcept {
        if (this != &o) { std::swap(buffer, o.buffer); std::swap(memory, o.memory);
                         std::swap(size, o.size); std::swap(ctx, o.ctx); }
        return *this;
    }

    void upload(const void* data, VkDeviceSize dataSize) {
        void* mapped;
        check(vkMapMemory(ctx->device, memory, 0, dataSize, 0, &mapped), "Failed to map memory");
        memcpy(mapped, data, dataSize);
        vkUnmapMemory(ctx->device, memory);
    }

    void download(void* data, VkDeviceSize dataSize) {
        void* mapped;
        check(vkMapMemory(ctx->device, memory, 0, dataSize, 0, &mapped), "Failed to map memory");
        memcpy(data, mapped, dataSize);
        vkUnmapMemory(ctx->device, memory);
    }
};

inline Buffer createDeviceBuffer(Context& ctx, VkDeviceSize size) {
    return Buffer(ctx, size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
}

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
    ComputePipeline(Context& context, const std::string& spvPath, uint32_t bufferCount)
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
    }

    ~ComputePipeline() {
        if (ctx && ctx->device) {
            vkDestroyPipeline(ctx->device, pipeline, nullptr);
            vkDestroyPipelineLayout(ctx->device, pipelineLayout, nullptr);
            vkDestroyDescriptorPool(ctx->device, descPool, nullptr);
            vkDestroyDescriptorSetLayout(ctx->device, descLayout, nullptr);
            vkDestroyShaderModule(ctx->device, shaderModule, nullptr);
        }
    }

    ComputePipeline(ComputePipeline&& o) noexcept
        : ctx(o.ctx), shaderModule(o.shaderModule), descLayout(o.descLayout),
          pipelineLayout(o.pipelineLayout), pipeline(o.pipeline),
          descPool(o.descPool), descSet(o.descSet) {
        o.ctx = nullptr; o.shaderModule = VK_NULL_HANDLE; o.descLayout = VK_NULL_HANDLE;
        o.pipelineLayout = VK_NULL_HANDLE; o.pipeline = VK_NULL_HANDLE;
        o.descPool = VK_NULL_HANDLE; o.descSet = VK_NULL_HANDLE;
    }

    void bindBuffers(const std::vector<Buffer*>& buffers) {
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

    void dispatch(uint32_t groupCountX, uint32_t groupCountY = 1, uint32_t groupCountZ = 1) {
        VkCommandBufferAllocateInfo allocInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        allocInfo.commandPool = ctx->cmdPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = 1;

        VkCommandBuffer cmdBuf;
        check(vkAllocateCommandBuffers(ctx->device, &allocInfo, &cmdBuf),
              "Failed to allocate command buffer");

        VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmdBuf, &beginInfo);

        vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout,
                                0, 1, &descSet, 0, nullptr);
        vkCmdDispatch(cmdBuf, groupCountX, groupCountY, groupCountZ);

        vkEndCommandBuffer(cmdBuf);

        VkSubmitInfo submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &cmdBuf;

        VkFence fence;
        VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        vkCreateFence(ctx->device, &fenceInfo, nullptr, &fence);

        vkQueueSubmit(ctx->queue, 1, &submitInfo, fence);
        vkWaitForFences(ctx->device, 1, &fence, VK_TRUE, UINT64_MAX);

        vkDestroyFence(ctx->device, fence, nullptr);
        vkFreeCommandBuffers(ctx->device, ctx->cmdPool, 1, &cmdBuf);
    }
};

} // namespace vkcompute
