import os
import random
import torch
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as T
from PIL import Image, ImageOps
import numpy as np
import cv2

class DataSet(Dataset):

    def __init__(self, image_path, label_path, img_size=(512, 512), train=False, coarse=False, erode=25, imp=0.005, ignore_index=255):
        self.img_size = img_size
        self.train = train
        self.coarse = coarse
        self.erode = erode
        self.imp = imp
        self.ignore_index = ignore_index

        self.samples = []     
        for sample in os.listdir(label_path):
            sample_image_path = image_path + sample
            sampe_label_path = label_path + sample
            if os.path.isfile(sample_image_path) and os.path.isfile(sampe_label_path):
                self.samples.append((sample_image_path, sampe_label_path))

        #To ensure reproducibility
        self.samples.sort()


    def transform(self, image, label):
        #Training augmentation
        if self.train:
            #Random crop
            i, j, h, w = T.RandomCrop.get_params(image, output_size=self.img_size)
            image = T.functional.crop(image, i, j, h, w)
            label = T.functional.crop(label, i, j, h, w)

            #Random horizontal flipping
            if random.random() < 0.5:
                image = T.functional.hflip(image)
                label = T.functional.hflip(label)
        else:
            #Center crop
            image = T.CenterCrop(self.img_size)(image)
            label = T.CenterCrop(self.img_size)(label)

        #Transform to tensor
        image = T.ToTensor()(image)
        label = T.ToTensor()(label).squeeze()
        
        return image, label.to(torch.int64)


    def make_coarse(self, label):
        coarse_label = torch.ones_like(label) * self.ignore_index
        kernel = np.ones((self.erode, self.erode), np.uint8)
        classes = torch.unique(label[label!=self.ignore_index])

        for class_idx in classes:
            class_label = np.array(label==class_idx, dtype='uint8')
            class_label = cv2.erode(class_label, kernel, iterations=1)
            
            contours, _ = cv2.findContours(class_label, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            new_contours = []
            for count in contours:
                epsilon = self.imp * cv2.arcLength(count, True)
                approximations = cv2.approxPolyDP(count, epsilon, True)
                new_contours.append(approximations)

            class_label = np.zeros(class_label.shape)
            cv2.fillPoly(class_label, pts=new_contours, color=1)

            coarse_label[class_label.astype(dtype=bool)] = class_idx

        return coarse_label
    
    
    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):
        (image, label) = self.samples[idx]
        image = Image.open(image)
        label = Image.open(label)
        
        image, label = self.transform(image, label)
        if self.coarse:
            label = self.make_coarse(label)

        return image, label