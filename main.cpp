#include "vk_compute.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <filesystem>

namespace fs = std::filesystem;

// Find shader path in bazel runfiles or relative to binary
std::string findShaderPath(const std::string& shaderName) {
    // Try runfiles location first (bazel run)
    const char* runfiles = std::getenv("RUNFILES_DIR");
    if (runfiles) {
        fs::path path = fs::path(runfiles) / "gpucompute" / shaderName;
        if (fs::exists(path)) return path.string();
    }

    // Try relative to executable
    fs::path exePath = fs::read_symlink("/proc/self/exe").parent_path();

    // Bazel runfiles structure
    fs::path runfilesPath = exePath / "gpu_compute.runfiles" / "gpucompute" / shaderName;
    if (fs::exists(runfilesPath)) return runfilesPath.string();

    // Direct relative path (for non-bazel builds)
    fs::path relativePath = exePath / shaderName;
    if (fs::exists(relativePath)) return relativePath.string();

    // Current directory fallback
    if (fs::exists(shaderName)) return shaderName;

    throw std::runtime_error("Shader not found: " + shaderName);
}

int main(int argc, char** argv) {
    using namespace vkcompute;

    try {
        // API: Create a Vulkan compute context
        // Pass an optional device name filter to select a specific GPU (e.g., "NVIDIA", "AMD", "Intel")
        std::string filter = (argc > 1) ? argv[1] : "";
        Context ctx(filter);
        std::cout << "Using device: " << ctx.deviceName() << "\n";
        ctx.printLimits();

        const uint32_t N = 1024 * 1024 * 40;
        const VkDeviceSize bufferSize = N * sizeof(float);

        // API: Create device-local buffers (fast GPU memory, not directly accessible from CPU)
        // Use these for intermediate computation results and main working memory
        Buffer bufA = createDeviceBuffer(ctx, bufferSize);
        Buffer bufB = createDeviceBuffer(ctx, bufferSize);
        Buffer bufC = createDeviceBuffer(ctx, bufferSize);
        Buffer bufD = createDeviceBuffer(ctx, bufferSize);

        // API: Create pinned (host-visible) buffers for CPU-GPU data transfer
        // These buffers are mapped to CPU memory and can be accessed directly via .mapped pointer
        Buffer pinA = createPinnedBuffer(ctx, bufferSize);
        Buffer pinB = createPinnedBuffer(ctx, bufferSize);
        Buffer pinC = createPinnedBuffer(ctx, bufferSize);
        Buffer pinD = createPinnedBuffer(ctx, bufferSize);

        // API: Access pinned buffer data directly through .mapped pointer
        // No need for explicit upload/download - write directly to GPU-visible memory
        float* hostA = static_cast<float*>(pinA.mapped);
        float* hostB = static_cast<float*>(pinB.mapped);
        float* hostC = static_cast<float*>(pinC.mapped);
        float * hostD = static_cast<float*>(pinD.mapped);

        // we will do C = A + B and A = C - D in a loop
        // meaning A = A + (B - D) * loop at the end
        // Initialize input data D = B - 1
        // A = A + loop at the end
        for (uint32_t i = 0; i < N; i++) {
            hostA[i] = static_cast<float>(i);
            hostB[i] = static_cast<float>(i + 10);
            hostD[i] = hostB[i] - 1;
        }

        // API: Copy data from pinned buffer to device buffer
        // This transfers data from host-visible memory to fast device-local GPU memory
        bufA.copyFrom(pinA);
        bufB.copyFrom(pinB);
        bufD.copyFrom(pinD);

        // Load SPIR-V shader path
        std::string shaderPath = findShaderPath("vector_add_shader.spv");
        std::cout << "Loading shader: " << shaderPath << "\n";
        std::string shaderPath2 = findShaderPath("vector_sub_shader.spv");
        std::cout << "Loading shader: " << shaderPath2 << "\n";

        // Determine workgroup size and number of workgroups, 256 is optimized for most GPUs
        const uint32_t workgroupSize = (argc > 2) ? std::atoi(argv[2]) : 256;
        // Calculate number of workgroups needed to cover N elements
        const uint32_t numGroups = (N + workgroupSize - 1) / workgroupSize;

        // API: Create a compute pipeline from a compiled SPIR-V shader
        // Parameters: context, shader path, number of buffers, workgroup size
        ComputePipeline pipeline(ctx, shaderPath, 3, workgroupSize);
        // API: Bind buffers to the pipeline's descriptor set (in shader binding order)
        pipeline.bindBuffers({&bufA, &bufB, &bufC});

        // Create a second pipeline for subtraction operation
        ComputePipeline pipeline2(ctx, shaderPath2, 3, workgroupSize);
        pipeline2.bindBuffers({&bufC, &bufD, &bufA});

        // API: Create a command sequence to batch multiple GPU operations
        Sequence seq(ctx);
        seq.begin();  // Start recording commands
        const int iterations = 40;
        for (int i = 0; i < iterations; i++) {
            // API: Record a compute dispatch with specified number of workgroups
            seq.record(pipeline, numGroups);
            // API: Insert a memory barrier to ensure previous operation completes
            // this is only needed because these 2 operations rely on each other's results (A / C)
            seq.barrier();
            seq.record(pipeline2, numGroups);
            seq.barrier();
        }
        seq.end();  // Finish recording

        // API: Submit the command sequence to GPU and wait for completion
        // Returns execution time in milliseconds for performance measurement
        double runTimeMs = seq.submitAndWait();
        int64_t totalOps = static_cast<int64_t>(N) * 2 * iterations;
        std::cout << "Shader run time: " << runTimeMs << " ms" 
                  << " | Throughput: " << (totalOps / (runTimeMs / 1000.0)) / 1e6 / 1e3 << " Gops/s\n";

        // API: Copy results back from device buffer to pinned buffer
        // After this, results are accessible via pinC.mapped pointer (hostC)
        pinC.copyFrom(bufA);

        // Verify computation results
        bool success = hostC[131072] == hostA[131072] + iterations;

        if (success) {
            std::cout << "Vector addition successful! Processed " << N << " elements.\n";
            std::cout << "Sample: " << hostA[5] << " + "  << iterations 
                      << " = " << hostC[5] << "\n";
            std::cout << "Sample: " << hostA[131072] << " + " << iterations
                      << " = " << hostC[131072] << "\n";
        }

        return success ? 0 : 1;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}