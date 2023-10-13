import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from Functions.utils import rgb_to_cielab, get_xy, downsample, upsample

class CrossEntropy(nn.Module):
    def __init__(self, ignore_index=255):
        super(CrossEntropy, self).__init__()
        self.ignore_index = ignore_index


    def __str__(self):
        return 'cross_entropy'


    def forward(self, assignments, output, image, label):
        loss = F.cross_entropy(output, label, ignore_index=self.ignore_index)
        
        return loss
    

class RegularisedCrossEntropy(nn.Module):
    def __init__(self, lmbda=0.05, m=0, ignore_index=255):
        super(RegularisedCrossEntropy, self).__init__()
        self.lmbda = lmbda
        self.m = m
        self.ignore_index = ignore_index

    
    def __str__(self):
        if self.lmbda==0 and self.m==0:
            return 'cross_entropy'
        else:
            return f'reg_cross_entropy_{self.lmbda}_{self.m}'


    def forward(self, assignments, output, image, label):
        entropy_loss = F.cross_entropy(output, label, ignore_index=self.ignore_index)

        #Downsample with interpolation
        sorted_keys = np.sort(list(assignments.keys()))
        size = (int(image.size(-2) / sorted_keys[0]), int(image.size(-1) / sorted_keys[0]))
        image = F.interpolate(image, size=size, mode="bilinear", align_corners=False)

        #Transform image to CIELAB and add XY-coordinates
        image = rgb_to_cielab(image)
        image = torch.cat([image, get_xy(image.size(0), image.size(2), image.size(3))], dim=1)
        features = image.clone()

        #Downsample with superpixels
        for key in sorted_keys:
            features = downsample(features, assignments[key])

        #Upsample with superpixels
        for key in sorted_keys[::-1]:
            features = upsample(features, assignments[key])

        #SLIC loss
        slic_map = features[:,:-2,:,:] - image[:,:-2,:,:]
        pos_map = features[:,-2:,:,:] - image[:,-2:,:,:]
        slic_loss = torch.mean(torch.norm(slic_map, p=2, dim=1))
        pos_loss = torch.mean(torch.norm(pos_map, p=2, dim=1)) / (sorted_keys[-1]**2)

        slic_loss = slic_loss + self.m * pos_loss

        return entropy_loss + self.lmbda * slic_loss
    
    
class SLIC(nn.Module):
    def __init__(self, m=0):
        super(SLIC, self).__init__()
        self.m = m


    def forward(self, assignments, output, image, label):
        #Downsample with interpolation
        sorted_keys = np.sort(list(assignments.keys()))
        size = (int(image.size(-2) / sorted_keys[0]), int(image.size(-1) / sorted_keys[0]))
        image = F.interpolate(image, size=size, mode="bilinear", align_corners=False)

        #Transform image to CIELAB and add XY-coordinates
        image = rgb_to_cielab(image)
        image = torch.cat([image, get_xy(image.size(0), image.size(2), image.size(3))], dim=1)
        features = image.clone()

        #Downsample with superpixels
        for key in sorted_keys:
            features = downsample(features, assignments[key])

        #Upsample with superpixels
        for key in sorted_keys[::-1]:
            features = upsample(features, assignments[key])

        #SLIC loss
        slic_map = features[:,:-2,:,:] - image[:,:-2,:,:]
        pos_map = features[:,-2:,:,:] - image[:,-2:,:,:]
        slic_loss = torch.mean(torch.norm(slic_map, p=2, dim=1))
        pos_loss = torch.mean(torch.norm(pos_map, p=2, dim=1)) / (sorted_keys[-1]**2)

        slic_loss = slic_loss + self.m * pos_loss

        return slic_loss