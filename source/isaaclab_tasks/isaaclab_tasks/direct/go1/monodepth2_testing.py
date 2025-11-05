import torch
import cv2
import os
from pathlib import Path
import numpy as np
from torchvision import transforms

# --- Settings ---
IMAGE_DIR = "/home/sripu715/Desktop/camera_images_demo"
OUTPUT_DIR = os.path.join(IMAGE_DIR, "depth_outs")
MODEL_TYPE = "MiDaS_small"
REGION_SIZE = 40  # pixels around each corner/center region to average

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Device ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {DEVICE}")

# --- Load MiDaS model ---
print(f"[INFO] Loading {MODEL_TYPE} model...")
midas = torch.hub.load("intel-isl/MiDaS", MODEL_TYPE)
midas.to(DEVICE)
midas.eval()

# --- Transforms ---
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform

# --- Image loop ---
image_paths = sorted(Path(IMAGE_DIR).glob("*.jpg")) + sorted(Path(IMAGE_DIR).glob("*.png"))
print(f"[INFO] Found {len(image_paths)} images to process.\n")

def region_mean(depth, cx, cy, region_size):
    """Return mean depth in a square region centered at (cx, cy)."""
    h, w = depth.shape
    x1, x2 = max(0, cx - region_size), min(w, cx + region_size)
    y1, y2 = max(0, cy - region_size), min(h, cy + region_size)
    return float(np.mean(depth[y1:y2, x1:x2]))

for img_path in image_paths:
    img = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    input_batch = transform(img_rgb).to(DEVICE)

    print('input_batch shape-->>',input_batch.shape)

    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img_rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth = prediction.cpu().numpy()

    h, w = depth.shape
    region_points = {
        "top_left": (int(w * 0.1), int(h * 0.1)),
        "top_right": (int(w * 0.9), int(h * 0.1)),
        "bottom_left": (int(w * 0.1), int(h * 0.9)),
        "bottom_right": (int(w * 0.9), int(h * 0.9)),
        "center": (w // 2, h // 2)
    }

    region_depths = {
        name: region_mean(depth, x, y, REGION_SIZE)
        for name, (x, y) in region_points.items()
    }

    # Normalize for display
    depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
    depth_uint8 = depth_norm.astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_PLASMA)

    # Save depth map
    out_path = os.path.join(OUTPUT_DIR, img_path.stem + "_depth.png")
    cv2.imwrite(out_path, depth_color)

    print(f"[{img_path.name}] avg regional depth → {region_depths}")

print("\n✅ Done. Averaged region depth values printed per frame.")
