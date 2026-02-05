#include "onnx_executor.hpp"
#include "../vk_compute.hpp"
#include <iostream>
#include <iomanip>
#include <cstring>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <vector>

// ============================================================================
// YOLOv5 Preprocessing - Letterbox resize
// ============================================================================
struct LetterboxInfo {
    float scale;      // Scale factor applied
    int padLeft;      // Padding on left
    int padTop;       // Padding on top
    int newWidth;     // Scaled width before padding
    int newHeight;    // Scaled height before padding
};

LetterboxInfo yoloPreprocess(
    const std::vector<uint8_t>& rawData,
    int srcWidth, int srcHeight,
    float* output,
    int targetWidth, int targetHeight)
{
    LetterboxInfo info;
    
    // Calculate scale to fit while maintaining aspect ratio
    float scaleW = (float)targetWidth / srcWidth;
    float scaleH = (float)targetHeight / srcHeight;
    info.scale = std::min(scaleW, scaleH);
    
    info.newWidth = (int)(srcWidth * info.scale);
    info.newHeight = (int)(srcHeight * info.scale);
    
    // Calculate padding (center the image)
    info.padLeft = (targetWidth - info.newWidth) / 2;
    info.padTop = (targetHeight - info.newHeight) / 2;
    
    // Initialize with gray (114/255 ≈ 0.447 is YOLOv5 standard)
    size_t totalSize = targetWidth * targetHeight * 3;
    for (size_t i = 0; i < totalSize; i++) {
        output[i] = 114.0f / 255.0f;
    }
    
    // Resize and place image with bilinear interpolation
    for (int c = 0; c < 3; c++) {
        for (int y = 0; y < info.newHeight; y++) {
            for (int x = 0; x < info.newWidth; x++) {
                // Map to source coordinates
                float srcX = x / info.scale;
                float srcY = y / info.scale;
                
                int x0 = (int)srcX;
                int y0 = (int)srcY;
                int x1 = std::min(x0 + 1, srcWidth - 1);
                int y1 = std::min(y0 + 1, srcHeight - 1);
                float fx = srcX - x0;
                float fy = srcY - y0;
                
                // Bilinear interpolation (input is HWC)
                auto getPixel = [&](int py, int px, int ch) -> float {
                    return rawData[(py * srcWidth + px) * 3 + ch] / 255.0f;
                };
                
                float v = getPixel(y0, x0, c) * (1-fx) * (1-fy) +
                          getPixel(y0, x1, c) * fx * (1-fy) +
                          getPixel(y1, x0, c) * (1-fx) * fy +
                          getPixel(y1, x1, c) * fx * fy;
                
                // Output is NCHW
                int outY = y + info.padTop;
                int outX = x + info.padLeft;
                output[c * targetHeight * targetWidth + outY * targetWidth + outX] = v;
            }
        }
    }
    
    return info;
}

// ============================================================================
// YOLOv5 Postprocessing - Detection structure and NMS
// ============================================================================
struct Detection {
    float x1, y1, x2, y2;  // Box coordinates in input image space
    float confidence;       // objectness * class_score
    int classId;
    float classScore;
};

float computeIoU(const Detection& a, const Detection& b) {
    float interX1 = std::max(a.x1, b.x1);
    float interY1 = std::max(a.y1, b.y1);
    float interX2 = std::min(a.x2, b.x2);
    float interY2 = std::min(a.y2, b.y2);
    
    float interArea = std::max(0.0f, interX2 - interX1) * std::max(0.0f, interY2 - interY1);
    float areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
    float areaB = (b.x2 - b.x1) * (b.y2 - b.y1);  // Fixed: was b.y2 - b.y2
    
    return interArea / (areaA + areaB - interArea + 1e-6f);
}

std::vector<Detection> nms(std::vector<Detection>& detections, float iouThreshold) {
    // Sort by confidence (descending)
    std::sort(detections.begin(), detections.end(), 
              [](const Detection& a, const Detection& b) { return a.confidence > b.confidence; });
    
    std::vector<Detection> result;
    std::vector<bool> suppressed(detections.size(), false);
    
    for (size_t i = 0; i < detections.size(); i++) {
        if (suppressed[i]) continue;
        
        result.push_back(detections[i]);
        
        for (size_t j = i + 1; j < detections.size(); j++) {
            if (suppressed[j]) continue;
            if (detections[i].classId != detections[j].classId) continue;
            
            if (computeIoU(detections[i], detections[j]) > iouThreshold) {
                suppressed[j] = true;
            }
        }
    }
    
    return result;
}

std::vector<Detection> yoloPostprocess(
    const float* output,
    int numDetections,
    int numClasses,
    const LetterboxInfo& letterbox,
    int srcWidth, int srcHeight,
    int targetWidth, int targetHeight,
    float confThreshold = 0.25f,
    float iouThreshold = 0.45f)
{
    std::vector<Detection> detections;
    
    int numValues = 5 + numClasses;  // x, y, w, h, obj_conf, class_scores...
    
    // Find max objectness for debug
    float maxObj = 0;
    int maxObjIdx = 0;
    for (int i = 0; i < numDetections; i++) {
        const float* det = output + i * numValues;
        if (det[4] > maxObj) {
            maxObj = det[4];
            maxObjIdx = i;
        }
    }
    std::cout << "  Max objectness: " << std::fixed << std::setprecision(4) << maxObj 
              << " at anchor " << maxObjIdx << std::endl;
    
    // Debug: print raw values of top detection
    {
        const float* det = output + maxObjIdx * numValues;
        std::cout << "  Top detection raw: cx=" << det[0] << " cy=" << det[1] 
                  << " w=" << det[2] << " h=" << det[3] << " obj=" << det[4];
        for (int c = 0; c < numClasses; c++) {
            std::cout << " c" << c << "=" << det[5+c];
        }
        std::cout << std::endl;
    }
    
    // Check coordinate range among confident detections
    float maxX = 0, maxY = 0, maxW = 0, maxH = 0;
    int validCount = 0;
    for (int i = 0; i < numDetections; i++) {
        const float* det = output + i * numValues;
        if (det[4] < 0.1f) continue;
        validCount++;
        maxX = std::max(maxX, det[0]);
        maxY = std::max(maxY, det[1]);
        maxW = std::max(maxW, det[2]);
        maxH = std::max(maxH, det[3]);
    }
    
    // Determine if coordinates are normalized (0-1) or absolute pixels
    bool isNormalized = (maxX <= 1.5f && maxY <= 1.5f);
    std::cout << "  Coordinate format: " << (isNormalized ? "normalized" : "pixels") 
              << " (maxX=" << maxX << ", maxY=" << maxY << ")" << std::endl;
    
    for (int i = 0; i < numDetections; i++) {
        const float* det = output + i * numValues;
        
        float objConf = det[4];
        // Use >= to filter out NaN values (NaN comparisons return false)
        if (!(objConf >= confThreshold)) continue;
        
        // Find best class
        int bestClass = 0;
        float bestClassScore = det[5];
        for (int c = 1; c < numClasses; c++) {
            if (det[5 + c] > bestClassScore) {
                bestClassScore = det[5 + c];
                bestClass = c;
            }
        }
        
        float confidence = objConf * bestClassScore;
        // Use >= to filter out NaN values
        if (!(confidence >= confThreshold)) continue;
        
        // Also check for valid box coordinates (filter NaN)
        if (std::isnan(det[0]) || std::isnan(det[1]) || std::isnan(det[2]) || std::isnan(det[3])) continue;
        
        // Get box coordinates (center x, center y, width, height)
        float cx = det[0];
        float cy = det[1];
        float w = det[2];
        float h = det[3];
        
        // If normalized, convert to target image pixel space
        if (isNormalized) {
            cx *= targetWidth;
            cy *= targetHeight;
            w *= targetWidth;
            h *= targetHeight;
        }
        
        // Convert from center format to corner format
        float x1 = cx - w / 2;
        float y1 = cy - h / 2;
        float x2 = cx + w / 2;
        float y2 = cy + h / 2;
        
        // Remove letterbox padding and scale back to original image
        x1 = (x1 - letterbox.padLeft) / letterbox.scale;
        y1 = (y1 - letterbox.padTop) / letterbox.scale;
        x2 = (x2 - letterbox.padLeft) / letterbox.scale;
        y2 = (y2 - letterbox.padTop) / letterbox.scale;
        
        // Clip to image bounds
        x1 = std::max(0.0f, std::min(x1, (float)srcWidth));
        y1 = std::max(0.0f, std::min(y1, (float)srcHeight));
        x2 = std::max(0.0f, std::min(x2, (float)srcWidth));
        y2 = std::max(0.0f, std::min(y2, (float)srcHeight));
        
        Detection d;
        d.x1 = x1; d.y1 = y1; d.x2 = x2; d.y2 = y2;
        d.confidence = confidence;
        d.classId = bestClass;
        d.classScore = bestClassScore;
        
        // Debug first few detections
        if (detections.size() < 3) {
            std::cout << "  Debug det[" << detections.size() << "]: anchor=" << i 
                      << " raw=(" << det[0] << "," << det[1] << "," << det[2] << "," << det[3] << ")"
                      << " -> box=(" << x1 << "," << y1 << "," << x2 << "," << y2 << ")"
                      << " conf=" << confidence << " class=" << bestClass << std::endl;
        }
        
        detections.push_back(d);
    }
    
    std::cout << "  Pre-NMS detections: " << detections.size() << std::endl;
    
    // Apply NMS
    return nms(detections, iouThreshold);
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char** argv) {
    try {
        std::string modelPath = "model.onnx";
        std::string deviceFilter = "";
        std::string inputPath = "";
        int inputWidth = 0;
        int inputHeight = 0;
        float confThreshold = 0.25f;
        float iouThreshold = 0.45f;

        for (int i = 1; i < argc; i++) {
            if (std::strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
                modelPath = argv[++i];
            } else if (std::strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
                deviceFilter = argv[++i];
            } else if (std::strcmp(argv[i], "--input") == 0 && i + 1 < argc) {
                inputPath = argv[++i];
            } else if (std::strcmp(argv[i], "--input-size") == 0 && i + 2 < argc) {
                inputWidth = std::atoi(argv[++i]);
                inputHeight = std::atoi(argv[++i]);
            } else if (std::strcmp(argv[i], "--conf") == 0 && i + 1 < argc) {
                confThreshold = std::atof(argv[++i]);
            } else if (std::strcmp(argv[i], "--iou") == 0 && i + 1 < argc) {
                iouThreshold = std::atof(argv[++i]);
            } else if (std::strcmp(argv[i], "--help") == 0) {
                std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
                std::cout << "  --model <path>         ONNX model file (default: model.onnx)" << std::endl;
                std::cout << "  --device <filter>      GPU device filter" << std::endl;
                std::cout << "  --input <raw_file>     Raw RGB uint8 image file" << std::endl;
                std::cout << "  --input-size <W> <H>   Input image dimensions" << std::endl;
                std::cout << "  --conf <threshold>     Confidence threshold (default: 0.25)" << std::endl;
                std::cout << "  --iou <threshold>      NMS IoU threshold (default: 0.45)" << std::endl;
                return 0;
            }
        }

        std::cout << "=== YOLOv5 ONNX Executor ===" << std::endl;

        // Create Vulkan context
        vkcompute::Context ctx(deviceFilter);
        std::cout << "Using device: " << ctx.deviceName() << std::endl;

        // Create executor and load model
        onnxrt::Executor executor(ctx);
        std::cout << "Loading model: " << modelPath << std::endl;
        executor.loadModel(modelPath);

        // Get model info
        auto inputInfos = executor.getInputInfo();
        auto outputInfos = executor.getOutputInfo();
        
        std::cout << "\nModel inputs:" << std::endl;
        for (const auto& info : inputInfos) {
            std::cout << "  " << info.name << ": " << onnxrt::shapeStr(info.shape) << std::endl;
        }
        std::cout << "Model outputs:" << std::endl;
        for (const auto& info : outputInfos) {
            std::cout << "  " << info.name << ": " << onnxrt::shapeStr(info.shape) << std::endl;
        }

        // Create input tensors
        auto inputs = executor.createInputs();
        
        // Get target dimensions from model
        int64_t targetH = inputInfos[0].shape[2];  // 640
        int64_t targetW = inputInfos[0].shape[3];  // 416
        
        LetterboxInfo letterbox = {};
        
        // Load and preprocess input
        for (auto& [name, tensor] : inputs) {
            if (!tensor.isPinned()) continue;

            if (!inputPath.empty() && inputWidth > 0 && inputHeight > 0) {
                // Load raw RGB uint8 file
                std::ifstream file(inputPath, std::ios::binary);
                if (!file) {
                    throw std::runtime_error("Failed to open input file: " + inputPath);
                }

                size_t rawSize = inputWidth * inputHeight * 3;
                std::vector<uint8_t> rawData(rawSize);
                file.read(reinterpret_cast<char*>(rawData.data()), rawSize);
                file.close();

                std::cout << "\nInput image: " << inputWidth << "x" << inputHeight << " (WxH)" << std::endl;
                
                // YOLOv5 letterbox preprocessing
                letterbox = yoloPreprocess(rawData, inputWidth, inputHeight,
                                           tensor.data<float>(), targetW, targetH);
                
                std::cout << "Letterbox: scale=" << letterbox.scale 
                          << ", newSize=" << letterbox.newWidth << "x" << letterbox.newHeight
                          << ", pad=(" << letterbox.padLeft << "," << letterbox.padTop << ")"
                          << std::endl;
            } else {
                // Fallback: gray image
                float* data = tensor.data<float>();
                size_t count = tensor.elementCount();
                for (size_t i = 0; i < count; i++) {
                    data[i] = 114.0f / 255.0f;
                }
                std::cout << "\nUsing gray test image" << std::endl;
            }
        }

        // Run inference
        std::cout << "\nRunning inference..." << std::endl;
        std::map<std::string, onnxrt::Tensor*> outputs;
        double timeMs = executor.runTimed(inputs, outputs);
        std::cout << "Inference time: " << std::fixed << std::setprecision(2) << timeMs << " ms" << std::endl;

        // Process output
        for (const auto& [name, tensor] : outputs) {
            const auto& shape = tensor->shape();
            std::cout << "\nOutput " << name << ": " << onnxrt::shapeStr(shape) << std::endl;
            
            // Copy to host
            size_t count = tensor->elementCount();
            std::vector<float> hostData(count);
            tensor->copyToHost(hostData.data(), count * sizeof(float));
            
            // YOLOv5 output: [1, num_detections, 5+num_classes]
            if (shape.size() == 3) {
                int numDetections = shape[1];
                int numValues = shape[2];
                int numClasses = numValues - 5;
                
                std::cout << "  Anchors: " << numDetections << ", Classes: " << numClasses << std::endl;
                
                // YOLOv5 postprocessing with NMS
                auto detections = yoloPostprocess(
                    hostData.data(), numDetections, numClasses,
                    letterbox, inputWidth, inputHeight, targetW, targetH,
                    confThreshold, iouThreshold);
                
                std::cout << "\n========================================" << std::endl;
                std::cout << "Final Detections: " << detections.size() << std::endl;
                std::cout << "========================================" << std::endl;
                
                const char* classNames[] = {"class0", "class1", "class2", "class3"};
                
                for (size_t i = 0; i < detections.size(); i++) {
                    const auto& d = detections[i];
                    const char* className = (d.classId < 4) ? classNames[d.classId] : "unknown";
                    std::cout << "  [" << i << "] " << className 
                              << " conf=" << std::fixed << std::setprecision(3) << d.confidence
                              << " box=(" << (int)d.x1 << ", " << (int)d.y1 
                              << ", " << (int)d.x2 << ", " << (int)d.y2 << ")"
                              << " size=" << (int)(d.x2-d.x1) << "x" << (int)(d.y2-d.y1)
                              << std::endl;
                }
            }
        }

        std::cout << "\nDone!" << std::endl;
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
