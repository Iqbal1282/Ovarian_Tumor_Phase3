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
#import albumentations.augmentations.functional as F
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


# Training transformations
common_train_transform = A.Compose([
    A.PadIfNeeded(min_height=284, min_width=284, border_mode=0, value=0),  
    A.Resize(256, 256),  
    A.ShiftScaleRotate(shift_limit=0.25, scale_limit=0.25, rotate_limit= 20, p =0.5),
    A.CenterCrop(256, 256, p=0.5),
    #A.Normalize(mean=[0.5], std=[0.5]),  # Normalize to [0,1] range
    #ToTensorV2(),
])

# Validation transformations
common_val_transform = A.Compose([
    A.PadIfNeeded(min_height=284, min_width=284, border_mode=0, value=0),
    A.Resize(256, 256),
    #A.Normalize(mean=[0.5], std=[0.5]),
    #ToTensorV2(),
])

# Normalization for images
transform_img = A.Compose([
    #A.Normalize(mean=[58.42], std=[51.01]),
    ToTensorV2(),
])

# Transform mask: Convert to binary mask and tensor
transform_mask = A.Compose([
    #A.Lambda(image=lambda x: (x > 0).float()),  
    ToTensorV2(),
])

train_transform = A.Compose([
    A.PadIfNeeded(min_height=256, min_width=256, border_mode=0, value=0, mask_value=0),
    #A.RandomCrop(256, 256),
	# A.RandomResizedCrop(scale=(0.95, 1.0),
	# 					ratio=(0.95, 1.05),
	# 					size=(256, 256)), 
	A.Resize(256, 256),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
	A.GaussNoise(std_range=(0.002, 0.1), p=0.6),
	
	A.RandomBrightnessContrast(brightness_limit=(-0.01, 0.1), contrast_limit=(-0.01, 0.01), p=0.5),
    A.ElasticTransform(alpha = 10, sigma = 250, p=0.5),
    A.GridDistortion(distort_limit=(-0.2,0.2), p=0.5),
    # #A.CLAHE(clip_limit=.5, tile_grid_size=(8, 8), p=0.5),
	A.ShiftScaleRotate(shift_limit=(-0.005,0.005), scale_limit=(-0.2, 0.005), rotate_limit=(-30,30), border_mode=0, value=0, p=0.9), 
    A.Downscale(scale_range=(0.85,0.99), p=0.5),
    A.Normalize(mean=(0.5,), std=(0.5,)),
	ToTensorV2()
])



val_transform = A.Compose([
    A.PadIfNeeded(min_height=256, min_width=256, border_mode=0, value=0, mask_value=0),
    A.Resize(256, 256),
    A.Normalize(mean=(0.5,), std=(0.5,)),
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
            transformed = common_train_transform(image=image, mask=mask)
        else:
            transformed = common_val_transform(image=image, mask=mask)

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

        image = np.array(Image.open(image_path).convert('L'))
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
    train_dataset = MMotu_Classificaiton_Dataset(phase = 'train', img_transform= transform_img)
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