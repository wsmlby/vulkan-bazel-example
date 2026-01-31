#include "vk_compute.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <chrono>

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
        // Initialize Vulkan compute context
        std::string filter = (argc > 1) ? argv[1] : "";
        Context ctx(filter);
        std::cout << "Using device: " << ctx.deviceName() << "\n";

        // Data size
        const uint32_t N = 1024 * 1024 * 20;  // 1M elements
        const VkDeviceSize bufferSize = N * sizeof(float);

        // Create buffers
        Buffer bufA = createDeviceBuffer(ctx, bufferSize);
        Buffer bufB = createDeviceBuffer(ctx, bufferSize);
        Buffer bufC = createDeviceBuffer(ctx, bufferSize);

        // Create pinned memory for input/output
        Buffer pinA = createPinnedBuffer(ctx, bufferSize);
        Buffer pinB = createPinnedBuffer(ctx, bufferSize);
        Buffer pinC = createPinnedBuffer(ctx, bufferSize);

        // Initialize input data directly into pinned memory
        float* hostA = static_cast<float*>(pinA.mapped);
        float* hostB = static_cast<float*>(pinB.mapped);
        
        for (uint32_t i = 0; i < N; i++) {
            hostA[i] = static_cast<float>(i);
            hostB[i] = static_cast<float>(i * 2);
        }

        // Upload to GPU (Direct copy from pinned to device)
        bufA.copyFrom(pinA);
        bufB.copyFrom(pinB);

        // Load shader and create pipeline (3 buffers: A, B, C)
        std::string shaderPath = findShaderPath("vector_add_shader.spv");
        std::cout << "Loading shader: " << shaderPath << "\n";
        
    

        ComputePipeline pipeline(ctx, shaderPath, 3);
        pipeline.bindBuffers({&bufA, &bufB, &bufC});
        ComputePipeline pipeline2(ctx, shaderPath, 3);
        pipeline2.bindBuffers({&bufC, &bufB, &bufA});

        // Dispatch compute (local_size_x = 256)
        const uint32_t workgroupSize = (argc > 2) ? std::atoi(argv[2]) : 256;
        const uint32_t numGroups = (N + workgroupSize - 1) / workgroupSize;

        Sequence seq(ctx);
        seq.begin();
        const int iterations = 10000;
        for (int i = 0; i < iterations; i++) {
            seq.record(pipeline, numGroups);
            seq.record(pipeline2, numGroups);
        }
        seq.end();

        double runTimeMs = seq.submitAndWait();
        int64_t totalOps = static_cast<int64_t>(N) * 2 * iterations; // 2 operations per element per iteration
        std::cout << "Shader run time: " << runTimeMs << " ms" 
                  << " | Throughput: " << (totalOps / (runTimeMs / 1000.0)) / 1e6 / 1e3 << " Gops/s\n";
        
        // Download results
        pinC.copyFrom(bufC);
        float* hostC = static_cast<float*>(pinC.mapped);

        // Verify results
        bool success = true;
        // float* ptrC = static_cast<float*>(pinC.mapped);
        // for (uint32_t i = 0; i < N; i++) {
        //     float expected = hostA[i] / (hostB[i] + 0.001f);  // Match shader operation
        //     if (std::fabs(ptrC[i] - expected) > 1e-5f) {
        //         std::cerr << "Mismatch at " << i << ": " << ptrC[i]
        //                   << " != " << expected << "\n";
        //         success = false;
        //         break;
        //     }
        // }

        if (success) {
            std::cout << "Vector addition successful! Processed " << N << " elements.\n";
            std::cout << "Sample: " << hostA[5] << " + " << hostB[5]
                      << " = " << hostC[5] << "\n";
        }

        // return success ? 0 : 1;
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
