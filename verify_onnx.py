#!/usr/bin/env python3
"""
Simple script to verify ONNX inference results using onnxruntime.
Uses model.onnx and frame.raw (raw buffer of shape 3x1280x800, uint8).
"""

import numpy as np
import onnxruntime as ort
import cv2


def load_raw_image(filepath: str, shape: tuple, dtype=np.uint8) -> np.ndarray:
    """Load raw image buffer from file."""
    with open(filepath, 'rb') as f:
        data = f.read()
    image = np.frombuffer(data, dtype=dtype).reshape(shape)
    return image


def preprocess_image(image: np.ndarray, target_size: tuple = (640, 416)) -> tuple:
    """
    Preprocess the image for ONNX model inference using letterbox scaling.
    Converts BGR to RGB, resizes with uniform scale, pads to target size,
    converts uint8 [0-255] to float32 [0-1] and adds batch dimension.
    
    Args:
        image: Input image in CHW format (C, H, W), BGR order
        target_size: Target size as (height, width) to match model input
        
    Returns:
        Tuple of (preprocessed image tensor, scale, pad_left, pad_top)
    """
    target_h, target_w = target_size
    
    # Convert from CHW to HWC for cv2
    image_hwc = np.transpose(image, (1, 2, 0))
    src_h, src_w = image_hwc.shape[:2]
    
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image_hwc, cv2.COLOR_BGR2RGB)
    
    # Calculate uniform scale (letterbox approach)
    scale = min(target_w / src_w, target_h / src_h)
    new_w, new_h = int(src_w * scale), int(src_h * scale)
    
    # Calculate padding
    pad_left = (target_w - new_w) // 2
    pad_top = (target_h - new_h) // 2
    
    # Resize with uniform scale
    image_resized = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Create letterboxed image with gray padding (114 is common for YOLO)
    image_letterbox = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
    image_letterbox[pad_top:pad_top+new_h, pad_left:pad_left+new_w] = image_resized
    
    # Convert to CHW
    image_chw = np.transpose(image_letterbox, (2, 0, 1))
    
    # Convert to float32 and normalize to [0, 1]
    image_float = image_chw.astype(np.float32) / 255.0
    
    # Add batch dimension: (C, H, W) -> (1, C, H, W)
    image_batch = np.expand_dims(image_float, axis=0)
    return image_batch, scale, pad_left, pad_top


def run_inference(model_path: str, input_data: np.ndarray) -> list:
    """Run ONNX model inference."""
    # Create inference session
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    
    # Get input name
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    input_type = session.get_inputs()[0].type
    
    print(f"Model input name: {input_name}")
    print(f"Model input shape: {input_shape}")
    print(f"Model input type: {input_type}")
    print(f"Actual input shape: {input_data.shape}")
    print(f"Actual input dtype: {input_data.dtype}")
    
    # Get output info
    for i, output in enumerate(session.get_outputs()):
        print(f"Model output[{i}] name: {output.name}, shape: {output.shape}, type: {output.type}")
    
    # Run inference
    outputs = session.run(None, {input_name: input_data})
    
    return outputs


def xywh2xyxy(x):
    """Convert bounding box format from [x_center, y_center, w, h] to [x1, y1, x2, y2]."""
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # x1
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # y1
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # x2
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # y2
    return y


def nms(boxes, scores, iou_threshold=0.45):
    """Non-Maximum Suppression."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        if order.size == 1:
            break
            
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    
    return np.array(keep)


def postprocess_yolov5(output, conf_threshold=0.25, iou_threshold=0.45, num_classes=4):
    """
    Post-process YOLOv5 output to get final detections.
    
    Args:
        output: Raw model output of shape (1, num_detections, 5 + num_classes)
                Format: [x_center, y_center, width, height, objectness, class_scores...]
        conf_threshold: Confidence threshold for filtering detections
        iou_threshold: IoU threshold for NMS
        num_classes: Number of classes
    
    Returns:
        List of detections, each as [x1, y1, x2, y2, confidence, class_id]
    """
    # Remove batch dimension
    predictions = output[0]  # Shape: (num_detections, 5 + num_classes)
    
    # Get objectness scores
    objectness = predictions[:, 4]
    
    # Get class scores
    class_scores = predictions[:, 5:5 + num_classes]
    
    # Get max class score and class id for each detection
    max_class_scores = np.max(class_scores, axis=1)
    class_ids = np.argmax(class_scores, axis=1)
    
    # Combined confidence: objectness * class_score
    scores = objectness * max_class_scores
    
    print(f"  Objectness max: {objectness.max():.6f}")
    print(f"  Class scores max: {max_class_scores.max():.6f}")
    print(f"  Combined conf max: {scores.max():.6f}")
    
    # Filter by confidence threshold
    mask = scores > conf_threshold
    filtered_boxes = predictions[mask, :4]
    filtered_scores = scores[mask]
    filtered_class_ids = class_ids[mask]
    
    print(f"  Candidates after conf threshold ({conf_threshold}): {len(filtered_boxes)}")
    
    if len(filtered_boxes) == 0:
        return []
    
    # Convert from xywh to xyxy format
    boxes_xyxy = xywh2xyxy(filtered_boxes)
    
    # Apply NMS per class
    final_detections = []
    for cls_id in range(num_classes):
        cls_mask = filtered_class_ids == cls_id
        if not np.any(cls_mask):
            continue
            
        cls_boxes = boxes_xyxy[cls_mask]
        cls_scores = filtered_scores[cls_mask]
        
        keep_indices = nms(cls_boxes, cls_scores, iou_threshold)
        
        for idx in keep_indices:
            box = cls_boxes[idx]
            score = cls_scores[idx]
            final_detections.append([box[0], box[1], box[2], box[3], score, cls_id])
    
    # Sort by confidence
    final_detections.sort(key=lambda x: x[4], reverse=True)
    
    return final_detections


def main():
    model_path = "model.onnx"
    raw_image_path = "frame.raw"
    
    # Raw image shape: OpenCV stores as HWC (H=1280, W=800, C=3)
    raw_shape_hwc = (1280, 800, 3)  # HWC format (OpenCV native)
    src_h, src_w = 1280, 800
    
    print(f"Loading raw image from: {raw_image_path}")
    
    # Load raw image - HWC format since OpenCV uses HWC
    try:
        with open(raw_image_path, 'rb') as f:
            data = f.read()
        # Load as HWC (OpenCV native format)
        image_hwc = np.frombuffer(data, dtype=np.uint8).reshape(raw_shape_hwc)
        print(f"Loaded image shape (HWC): {image_hwc.shape}")
        print(f"Image dtype: {image_hwc.dtype}")
        print(f"Image value range: [{image_hwc.min()}, {image_hwc.max()}]")
        
        # Convert HWC to CHW for consistency
        image = np.transpose(image_hwc, (2, 0, 1))
        print(f"Converted to CHW: {image.shape}")
    except FileNotFoundError:
        print(f"Error: {raw_image_path} not found!")
        return
    except Exception as e:
        print(f"Error loading image: {e}")
        return
    
    # Preprocess - letterbox resize to model expected size (H=640, W=416)
    # Model expects [1, 3, 640, 416] = (batch, C, H, W)
    input_data, scale, pad_left, pad_top = preprocess_image(image, target_size=(640, 416))
    print(f"Preprocessed input shape: {input_data.shape}")
    print(f"Letterbox params: scale={scale}, pad_left={pad_left}, pad_top={pad_top}")
    
    # Run inference
    print("\nRunning inference...")
    try:
        outputs = run_inference(model_path, input_data)
        
        print("\n=== Raw Inference Results ===")
        for i, output in enumerate(outputs):
            print(f"\nOutput[{i}]:")
            print(f"  Shape: {output.shape}")
            print(f"  Dtype: {output.dtype}")
            print(f"  Min: {output.min():.6f}")
            print(f"  Max: {output.max():.6f}")
            print(f"  Mean: {output.mean():.6f}")
            print(f"  Std: {output.std():.6f}")
            
            # Print first few values
            flat = output.flatten()
            print(f"  First 10 values: {flat[:10]}")
            
            # Save output to file for comparison
            output_file = f"output_{i}.npy"
            np.save(output_file, output)
            print(f"  Saved to: {output_file}")
        
        # Post-process YOLOv5 output
        # Output shape is (1, 16380, 9) -> 9 = 4 (box) + 1 (objectness) + 4 (classes)
        print("\n=== YOLOv5 Post-Processing ===")
        detections = postprocess_yolov5(outputs[0], conf_threshold=0.25, iou_threshold=0.45, num_classes=4)
        
        print(f"\nFinal Detections after NMS: {len(detections)}")
        
        # Model input: [1, 3, 640, 416] = [batch, C, H, W]
        # H=640, W=416
        # Original image: H=1280, W=800
        # Coordinate transformation using letterbox params
        # scale, pad_left, pad_top are from preprocessing
        
        print("\nAll detections:")
        for i, det in enumerate(detections):
            x1, y1, x2, y2, conf, cls_id = det
            # The detection is in xyxy format (model space)
            model_cx = (x1 + x2) / 2
            model_cy = (y1 + y2) / 2
            model_w = x2 - x1
            model_h = y2 - y1
            
            # Convert to original image coordinates using letterbox inverse transform
            # Subtract padding first, then divide by scale
            x1_orig = (x1 - pad_left) / scale
            y1_orig = (y1 - pad_top) / scale
            x2_orig = (x2 - pad_left) / scale
            y2_orig = (y2 - pad_top) / scale
            
            # Clamp to image boundaries (W=800, H=1280)
            x1_orig = max(0, min(x1_orig, src_w))
            y1_orig = max(0, min(y1_orig, src_h))
            x2_orig = max(0, min(x2_orig, src_w))
            y2_orig = max(0, min(y2_orig, src_h))
            
            # Convert to xywh format (top-left corner)
            x_orig = x1_orig
            y_orig = y1_orig
            w_orig = x2_orig - x1_orig
            h_orig = y2_orig - y1_orig
            
            print(f"  Detection {i+1}: class={int(cls_id)}, conf={conf:.4f}")
            print(f"    Model space (raw): cx={model_cx:.2f}, cy={model_cy:.2f}, w={model_w:.2f}, h={model_h:.2f}")
            print(f"    Image space (xywh top-left): x={int(x_orig)}, y={int(y_orig)}, w={int(w_orig)}, h={int(h_orig)}")
            
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()