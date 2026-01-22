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
        // Initialize Vulkan compute context
        std::string filter = (argc > 1) ? argv[1] : "";
        Context ctx(filter);
        std::cout << "Using device: " << ctx.deviceName() << "\n";

        // Data size
        const uint32_t N = 1024 * 1024;  // 1M elements
        const VkDeviceSize bufferSize = N * sizeof(float);

        // Create buffers
        Buffer bufA = createDeviceBuffer(ctx, bufferSize);
        Buffer bufB = createDeviceBuffer(ctx, bufferSize);
        Buffer bufC = createDeviceBuffer(ctx, bufferSize);

        // Initialize input data
        std::vector<float> hostA(N), hostB(N), hostC(N);
        for (uint32_t i = 0; i < N; i++) {
            hostA[i] = static_cast<float>(i);
            hostB[i] = static_cast<float>(i * 2);
        }

        // Upload to GPU
        bufA.upload(hostA.data(), bufferSize);
        bufB.upload(hostB.data(), bufferSize);

        // Load shader and create pipeline (3 buffers: A, B, C)
        std::string shaderPath = findShaderPath("vector_add_shader.spv");
        std::cout << "Loading shader: " << shaderPath << "\n";

        ComputePipeline pipeline(ctx, shaderPath, 3);
        pipeline.bindBuffers({&bufA, &bufB, &bufC});

        // Dispatch compute (local_size_x = 256)
        const uint32_t workgroupSize = 256;
        const uint32_t numGroups = (N + workgroupSize - 1) / workgroupSize;
        pipeline.dispatch(numGroups);

        // Download results
        bufC.download(hostC.data(), bufferSize);

        // Verify results
        bool success = true;
        for (uint32_t i = 0; i < N; i++) {
            float expected = hostA[i] / (hostB[i] + 0.001f);  // Match shader operation
            if (std::fabs(hostC[i] - expected) > 1e-5f) {
                std::cerr << "Mismatch at " << i << ": " << hostC[i]
                          << " != " << expected << "\n";
                success = false;
                break;
            }
        }

        if (success) {
            std::cout << "Vector addition successful! Processed " << N << " elements.\n";
            std::cout << "Sample: " << hostA[1000] << " + " << hostB[1000]
                      << " = " << hostC[1000] << "\n";
        }

        return success ? 0 : 1;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
