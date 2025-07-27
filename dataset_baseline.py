import os
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import torch 
import pandas as pd 
from pathlib import Path
import nrrd
import torch
from torch.utils.data import Dataset
import albumentations as A

#import albumentations.functional as F
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2 
import matplotlib.pyplot as plt 


import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2
import os
from PIL import Image

import albumentations as A
from albumentations.pytorch import ToTensorV2
import warnings
warnings.filterwarnings("ignore")
import os
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"



# train_transform = A.Compose([
#     # Resize with ratio range
#     A.RandomResizedCrop(size=(384, 384), scale=(0.9, 1.0), ratio=(0.9, 1.1), p=1.0),
#     A.ShiftScaleRotate(shift_limit=(-0.005,0.005), scale_limit=(-0.2, 0.005), rotate_limit=(-30,30), border_mode=0, value=0, p=0.6),

#     # Random cropping to fixed size
#     #A.RandomCrop(height=384, width=384, p=1.0),

#     # Horizontal flip
#     A.HorizontalFlip(p=0.5),

#     A.ElasticTransform(alpha = 10, sigma = 250, p=0.5),
#     A.GridDistortion(distort_limit=(-0.2,0.2), p=0.5),

#     # Photometric distortions
#     A.OneOf([
#         A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
#         A.CLAHE(clip_limit=2.0, p=0.3),
#         #A.HueSaturationValue(p=0.3),  # Doesn't apply to grayscale but included for RGB fallback
#     ], p=0.6),

#     # Normalize using paper settings (converted to single-channel equivalent)
#     A.Normalize(mean=(0.5,), std=(0.5,), max_pixel_value=255.0),  # Adapted for grayscale

#     # Ensure padding to crop size
#     A.PadIfNeeded(min_height=384, min_width=384, border_mode=0, value=0, p=1.0),

#     ToTensorV2()
# ])


# val_transform = A.Compose([
#     A.Resize(384, 384),
#     A.Normalize(mean=(0.5,), std=(0.5,), max_pixel_value=255.0),
#     ToTensorV2()
# ])

import albumentations as A
from albumentations.pytorch import ToTensorV2

train_transform = A.Compose([
    A.Resize(height=448, width=448, p = 1),  # matches img_scale=(448, 448)
    #A.ElasticTransform(alpha = 1, sigma = 250, p=0.5),
    A.GridDistortion(distort_limit=(-0.1,0.1), p=0.5),
    #A.ShiftScaleRotate(shift_limit=(-0.005,0.005), scale_limit=(-0.2, 0.005), rotate_limit=(-30,30), border_mode=0, value=0, p=0.6),

    A.ShiftScaleRotate(
    shift_limit=(-0.005, 0.005),
    scale_limit=(-0.2, 0.005),
    rotate_limit=(-30, 30),
    border_mode=0,
    p=0.6
),


    
    # Mimic Random Resize with ratio_range (0.5–2.0)
    A.RandomResizedCrop(size=(384, 384), scale=(0.8, 1.0), ratio=(0.75, 1.33), p=1.0),


    # Optional: limit max category ratio — we skip `cat_max_ratio` because it’s specific to segmentation class balance

    # Random horizontal flip (flip prob=0.5)
    A.HorizontalFlip(p=0.5),

    # Photometric distortion
    A.OneOf([
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.5),
    ], p=1.0),

    # Normalization (match mmseg style)
    A.Normalize(
        mean=[123.675 / 255.0, 116.28 / 255.0, 103.53 / 255.0],
        std=[58.395 / 255.0, 57.12 / 255.0, 57.375 / 255.0],
        max_pixel_value=255.0
    ),

    # Padding to final crop size (crop_size = 384x384)
    #A.PadIfNeeded(min_height=384, min_width=384, border_mode=0, value=0, mask_value=0),

    A.PadIfNeeded(
    min_height=384,
    min_width=384,
    border_mode=0,
    p=1.0  # Add a probability if needed
),

    ToTensorV2()
])


val_transform = A.Compose([
    A.Resize(height=448, width=448, p = 1),  # matches img_scale=(448, 448)
    
    # MultiScaleFlipAug is for test-time augmentation. We use a single scale for simplicity.
    A.HorizontalFlip(p=0.0),  # Flip = False

    A.Normalize(
        mean=[123.675 / 255.0, 116.28 / 255.0, 103.53 / 255.0],
        std=[58.395 / 255.0, 57.12 / 255.0, 57.375 / 255.0],
        max_pixel_value=255.0
    ),

    ToTensorV2()
])



class SegmentImageDataset(Dataset):
    def __init__(
        self,
        root_dir="data/OTU_2d",
        phase="train",
        img_transform=None,
        mask_transform=None,
        test_mode=False,
    ):
        self.root_dir = root_dir
        self.phase = phase
        self.img_transform = img_transform
        self.mask_transform = mask_transform
        self.test_mode = test_mode

        # Load image filenames
        unique_img_files = os.listdir(os.path.join(self.root_dir, "images"))
        num_lines = len(unique_img_files)
        separation_index = int(0.85 * num_lines)

        if self.phase == "train":
            self.image_files = unique_img_files[:separation_index]
        else:
            self.image_files = unique_img_files[separation_index:]

        # Store image-mask pairs
        self.data = [
            (os.path.join(self.root_dir, "images", file),
             os.path.join(self.root_dir, "masks", file.split(".")[0] + "_binary.PNG"))
            for file in self.image_files
        ]

    def __getitem__(self, index):
        image_path, mask_path = self.data[index]

        # Load image and mask as grayscale NumPy arrays
        image = np.array(Image.open(image_path).convert('L'))
        mask = np.array(Image.open(mask_path).convert('L'))

        # Apply the corresponding transformations
        if self.phase == 'train':
            transformed = train_transform(image=image, mask=mask)
        else:
            transformed = val_transform(image=image, mask=mask)

        slice_image = transformed['image']
        slice_mask = transformed['mask'].astype(np.float32)

        # Apply additional transformations if provided
        if self.img_transform:
            slice_image = self.img_transform(image=slice_image)['image'].to(torch.float32)
            slice_image = transforms.functional.normalize(slice_image, mean=58.42, std= 51.01)
        if self.mask_transform:
            slice_mask = self.mask_transform(image=slice_mask)['image']

        return slice_image, slice_mask

    def __len__(self):
        return len(self.data)

 # This dataset loader will be used for experiment with single image based  model , where model is encoder(narrow) + fc regression model  (8 classes) 
class MMotu_Classificaiton_Dataset(Dataset):
    def __init__(
        self,
        root_dir="data/OTU_2d",
        phase="train",
        img_transform=None,
        mask_transform=None,
        test_mode=False,
        num_classes = 8
    ):
        self.root_dir = root_dir
        self.phase = phase

        if self.phase == "train":
            self.img_transform = train_transform
            self.mask_transform = train_transform
        else: 
            self.img_transform = val_transform
            self.mask_transform = val_transform

        self.test_mode = test_mode
        self.num_classes = num_classes


        if self.phase == "train":
            label_file_path = r'data/OTU_2d/train_cls.txt'
            with open(label_file_path, "r") as f:
                data = [line.strip().split() for line in f]
            file_label_pairs = [(filename, int(label)) for filename, label in data]
        else:
            label_file_path = r'data/OTU_2d/val_cls.txt'
            with open(label_file_path, "r") as f:
                data = [line.strip().split() for line in f]
            file_label_pairs = [(filename, int(label)) for filename, label in data]

        # Store image-mask pairs
        self.data = [
            (os.path.join(self.root_dir, "images", file),
             os.path.join(self.root_dir, "masks", file.split(".")[0] + "_binary.PNG"), label)
            for file, label in file_label_pairs
        ]

    def __getitem__(self, index):
        image_path, mask_path, label = self.data[index]

        label = torch.tensor(label, dtype=torch.long)

        image = np.array(Image.open(image_path).convert('RGB'))
        label = torch.nn.functional.one_hot(label, num_classes= self.num_classes)
        # Apply additional transformations if provided
        if self.img_transform:
            transformed = self.img_transform(image=image)
            return transformed["image"], label 

    def __len__(self):
        return len(self.data)
    

    
if __name__ == '__main__':
    # train_dataset = SegmentImageDataset(img_transform=transform_img, mask_transform=transform_mask)
    # #test_dataset = SegmentImageDataset(transform=transform, target_transform=transform2, test_mode=True)

    # print(train_dataset[0][0].shape)
    # print(train_dataset[0][1].shape)
    # print(train_dataset[0][0].max())
    # print(train_dataset[0][1].min())
    # # print(len(train_dataset))
    # # from models import LitCNN 
    # # segment_model = LitCNN()

    
    
    # for i in range(10):
    #     img, mask = train_dataset[i]
    #     plt.figure()
    #     plt.subplot(1,3,1)
    #     plt.imshow(img[0], cmap= 'gray')

    #     plt.subplot(1,3,2)
    #     plt.imshow(mask[0]*255, cmap = 'gray')

    #     plt.subplot(1,3,3)
    #     plt.imshow((img[0]*mask[0])**1, cmap= 'gray')
    #     #print(((img[0]*mask[0])**2).min())
    #     plt.show()


    #Classificaiton_Dataset(phase = 'train', img_transform= transform_img)
    train_dataset = MMotu_Classificaiton_Dataset(phase = 'train') #, img_transform= transform_img)
    print(train_dataset[21][0].shape)
    print(train_dataset[21][1].shape)
    #print(train_dataset[21][2])
    print(train_dataset[21][0].max())
    print(train_dataset[21][1].max())
    print(len(train_dataset))

    for i in range(10):
        img, label = train_dataset[i]
        print(label)
        plt.figure()
        #plt.subplot(1,3,1)
        plt.imshow(img[0], cmap= 'gray')
        plt.show()


    # responder_count = 0 
    # no_responder_count = 0 

    # for i in range(len(train_dataset)):
    #     if train_dataset[i][2] == 0: 
    #         no_responder_count+=1 
    #     else: 
    #         responder_count+=1
    # print("response: ", responder_count)
    # print("no response: ", no_responder_count)
    # #pass 