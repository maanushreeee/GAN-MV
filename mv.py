from cellpose import models
import cv2
import numpy as np

def load_image(image_path):
    img_bgr = cv2.imread(image_path)
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def apply_cellpose(img_rgb):
    model = models.Cellpose(model_type='cyto')
    masks, _, _, _ = model.eval(img_rgb, channels=[0, 0])
    output = img_rgb.copy()

    for label in np.unique(masks):
        if label == 0:
            continue
        cell_mask = (masks == label).astype(np.uint8)
        cell_region = cv2.bitwise_and(output, output, mask=cell_mask)
        smoothed = cv2.GaussianBlur(cell_region, (5, 5), 0)

        hsv_cell = cv2.cvtColor(smoothed, cv2.COLOR_RGB2HSV)
        saturation = hsv_cell[:, :, 1]
        if np.mean(saturation[cell_mask == 1]) < 80:
            hsv_cell[:, :, 2] = np.clip(hsv_cell[:, :, 2] - 50, 0, 255)
            hsv_cell[:, :, 1] = np.clip(hsv_cell[:, :, 1] - 10, 0, 255)
            smoothed = cv2.cvtColor(hsv_cell, cv2.COLOR_HSV2RGB)

        for c in range(3):
            output[:, :, c] = np.where(cell_mask == 1, smoothed[:, :, c], output[:, :, c])

    return output

def apply_clahe(img_rgb):
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

def apply_median_blur(img_rgb):
    return cv2.medianBlur(img_rgb, 3)

# Optional: unsharp masking
def apply_unsharp_mask(img_rgb):
    blurred = cv2.GaussianBlur(img_rgb, (5, 5), 1)
    return cv2.addWeighted(img_rgb, 1.5, blurred, -0.5, 0)

def apply_color_boost(img_rgb):
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    hsv[:, :, 1] = cv2.add(hsv[:, :, 1], 10)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

