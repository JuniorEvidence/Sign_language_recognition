import os
from PIL import Image
from tqdm import tqdm
import shutil

# Path to your dataset (change this if needed)
SOURCE_DIR = r"C:\Users\shash\OneDrive - dauniv.ac.in\Desktop\Assignments\signLang\dataset\asl_alphabet_train\asl_alphabet_train"
OUT_IMAGES = "dataset/images/train"
OUT_LABELS = "dataset/labels/train"

# Create output directories
os.makedirs(OUT_IMAGES, exist_ok=True)
os.makedirs(OUT_LABELS, exist_ok=True)

# List of 29 class folders (sorted for consistency)
classes = sorted([cls for cls in os.listdir(SOURCE_DIR) if os.path.isdir(os.path.join(SOURCE_DIR, cls))])
class_to_id = {cls: i for i, cls in enumerate(classes)}

print("Classes:", class_to_id)

# Loop over folders
for cls in tqdm(classes, desc="Converting"):
    cls_path = os.path.join(SOURCE_DIR, cls)
    for img_file in os.listdir(cls_path):
        if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img_path = os.path.join(cls_path, img_file)
        img = Image.open(img_path)
        w, h = img.size

        # Save image
        out_img_name = f"{cls}_{img_file}"
        out_img_path = os.path.join(OUT_IMAGES, out_img_name)
        shutil.copy(img_path, out_img_path)

        # Write YOLO label covering full image
        out_label_name = out_img_name.rsplit('.', 1)[0] + ".txt"
        out_label_path = os.path.join(OUT_LABELS, out_label_name)

        with open(out_label_path, "w") as f:
            f.write(f"{class_to_id[cls]} 0.5 0.5 1.0 1.0\n")
