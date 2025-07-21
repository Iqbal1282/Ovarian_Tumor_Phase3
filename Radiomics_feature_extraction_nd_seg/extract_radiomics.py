import os
import cv2
import numpy as np
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor

# --- Config ---
image_dir = 'data\\OTU_2d\\images'
mask_dir = 'data\\OTU_2d\\masks'
output_csv = 'radiomics_analysis\\radiomics_features.csv'
param_file = 'radiomics_analysis\\radiomics_config.yaml'

# --- Init extractor ---
extractor = featureextractor.RadiomicsFeatureExtractor(param_file)

results = []

for filename in sorted(os.listdir(image_dir)):
    if not filename.endswith(('.png', '.jpg', '.jpeg')):
        continue

    patient_id = os.path.splitext(filename)[0]
    image_path = os.path.join(image_dir, filename)
    mask_path = os.path.join(mask_dir, f"{patient_id}_binary.png")

    print(f"Processing {patient_id}...")

    # Load image and mask
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if image is None or mask is None:
        print(f"❌ Error loading {patient_id}. Skipping.")
        continue

    # Convert to SimpleITK
    image_sitk = sitk.GetImageFromArray(image.astype(np.float32))
    mask_sitk = sitk.GetImageFromArray((mask > 0).astype(np.uint8))

    # Extract features
    features = extractor.execute(image_sitk, mask_sitk)
    features = {k: v for k, v in features.items() if 'diagnostics' not in k}
    features['PatientID'] = patient_id
    results.append(features)

# Save to CSV
df = pd.DataFrame(results)
df.to_csv(output_csv, index=False)
print(f"\n✔ Saved radiomic features to {output_csv}")
