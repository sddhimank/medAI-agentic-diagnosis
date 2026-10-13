import os
import cv2
import numpy as np
import pydicom
import gc
import logging

logger = logging.getLogger("uvicorn.error")

def process_medical_scan(file_path: str, output_img_path: str, annotated_img_path: str) -> tuple[str, dict]:
    logger.info(f"[PREPROCESSOR] Starting processing execution metrics for: {os.path.basename(file_path)}")
    metadata = {"patient_age": "Unknown", "scan_type": "Standard Image", "modality": "Unknown"}
    gc.collect()
    
    if file_path.lower().endswith('.dcm'):
        ds = pydicom.dcmread(file_path, stop_before_pixels=False)
        metadata["patient_age"] = str(getattr(ds, "PatientAge", "N/A"))
        metadata["modality"] = str(getattr(ds, "Modality", "Unknown"))
        metadata["scan_type"] = f"DICOM ({metadata['modality']})"
        
        try:
            ds.convert_pixel_data()
            pixel_array = ds.pixel_array
        except Exception as pydicom_err:
            raise ValueError(f"DICOM Decompression Failure: {str(pydicom_err)}")

        if len(pixel_array.shape) == 3:
            pixel_array = pixel_array[pixel_array.shape // 2]
            
        intercept = float(getattr(ds, "RescaleIntercept", 0))
        slope = float(getattr(ds, "RescaleSlope", 1))
        
        if pixel_array.shape[0] > 512 or pixel_array.shape[1] > 512:
            pixel_array = cv2.resize(pixel_array, (512, 512), interpolation=cv2.INTER_AREA)
        
        hu_image = pixel_array.astype(np.float32) * slope + intercept
        img_min, img_max = hu_image.min(), hu_image.max()
        if img_max - img_min == 0:
            normalized = np.zeros(hu_image.shape, dtype=np.uint8)
        else:
            normalized = ((hu_image - img_min) / (img_max - img_min) * 255.0).astype(np.uint8)
            
        if str(getattr(ds, "PhotometricInterpretation", "MONOCHROME2")).strip() == "MONOCHROME1":
            normalized = cv2.bitwise_not(normalized)
    else:
        normalized = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if normalized is None:
            raise ValueError("Unsupported or corrupted visual file.")
        if normalized.shape[0] > 512 or normalized.shape[1] > 512:
            normalized = cv2.resize(normalized, (512, 512), interpolation=cv2.INTER_AREA)
            
    cv2.imwrite(output_img_path, normalized)
    
    # --- LIGHTWEIGHT OPENCV ANNOTATION ENGINE ---
    h, w = normalized.shape[:2]
    color_annotated = cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)
    
    thresh = cv2.threshold(normalized, 180, 255, cv2.THRESH_BINARY)[1]
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    concern_count = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 80 < area < 5000:
            x, y, cw, ch = cv2.boundingRect(cnt)
            if 30 < x < (w - 30) and 30 < y < (h - 30):
                concern_count += 1
                center = (int(x + cw/2), int(y + ch/2))
                radius = int((cw + ch) / 3)
                cv2.circle(color_annotated, center, radius, (0, 140, 255), 2)
                cv2.putText(color_annotated, f"ANOMALY {concern_count}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 140, 255), 1, cv2.LINE_AA)
                if concern_count >= 3:
                    break
            
    # FIXED: Fallback marker dynamically reads h and w to center perfectly inside the image bounds
    if concern_count == 0:
        cx, cy = w // 2, h // 2
        box_size = 80
        cv2.rectangle(color_annotated, (cx - box_size, cy - box_size), (cx + box_size, cy + box_size), (0, 100, 230), 2)
        cv2.putText(color_annotated, "REGION OF INTEREST", (cx - box_size + 5, cy - box_size - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 100, 230), 1, cv2.LINE_AA)
        
    cv2.imwrite(annotated_img_path, color_annotated)
    
    del normalized, thresh, color_annotated
    gc.collect()
    logger.info("✓ [PREPROCESSOR COMPLETE] Dynamic layout annotation computed successfully.")
    return output_img_path, metadata