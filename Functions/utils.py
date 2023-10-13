import os
import torch
import torch.nn.functional as F
import numpy as np
import ast
from Functions.model_blocks import *


def get_xy(b, h, w):
    """Generates XY pixel coordinates for the image shape"""
    h_coords = np.arange(0, h, 1)
    w_coords = np.arange(0, w, 1)
    grid = np.array(np.meshgrid(h_coords, w_coords, indexing='ij'))
    grid = np.flip(grid, axis=0)
    xy = torch.from_numpy(np.tile(grid, (b, 1, 1, 1)).astype(np.float32)).to('cuda' if torch.cuda.is_available() else 'cpu')

    return xy


def extract_superpixels(assignments):
    """Extracts the superpixels from the assignments (seeds)"""
    s = list(assignments)[0]
    w_feat = int(assignments[s].size(-2) / 2)
    h_feat = int(assignments[s].size(-1) / 2)
    seeds = torch.tensor(range(int(w_feat/4) * int(h_feat/4))).view(1, int(w_feat/4), int(h_feat/4))
    seeds = F.one_hot(seeds).permute(0, -1, 1, 2).float()
    seeds = seeds.tile(1, 1, 4, 4)

    for key in assignments:
        b, _, h, w = assignments[key].shape
        n_channels = seeds.size(1)

        candidate_clusters = F.unfold(seeds, kernel_size=3, padding=1).reshape(b, n_channels, 9, -1)
        assignment = F.unfold(assignments[key].cpu(), kernel_size=2, stride=2).reshape(b, 9, 4, -1)
        seeds = torch.einsum('bkcn,bcpn->bkpn', (candidate_clusters, assignment)).reshape(b, n_channels * 4, -1)
        seeds = F.fold(seeds, (h, w), kernel_size=2, stride=2)

    segments = F.interpolate(seeds, size=(w_feat*s*2, h_feat*s*2), mode="bilinear", align_corners=False)
    segments = torch.argmax(segments, dim=1).squeeze().numpy()

    return segments


def rgb_to_cielab(image):
    """Transforms a RGB image to the CIELAB color space"""
    mask = image > 0.04045
    image[mask] = torch.pow((image[mask] + 0.055) / 1.055, 2.4)
    image[~mask] /= 12.92

    xyz_from_rgb = torch.tensor([[0.412453, 0.357580, 0.180423],
                             [0.212671, 0.715160, 0.072169],
                             [0.019334, 0.119193, 0.950227]]).to('cuda' if torch.cuda.is_available() else 'cpu')
    rgb = image.permute(0,2,3,1)

    xyz_img = torch.matmul(rgb, xyz_from_rgb.transpose_(0, 1))
    xyz_ref_white = torch.tensor([0.95047, 1., 1.08883]).to('cuda' if torch.cuda.is_available() else 'cpu')

    # scale by CIE XYZ tristimulus values of the reference white point
    lab = xyz_img / xyz_ref_white

    # Nonlinear distortion and linear transformation
    mask = lab > 0.008856
    lab[mask] = torch.pow(lab[mask], 1. / 3.)
    lab[~mask] = 7.787 * lab[~mask] + 16. / 116.

    x, y, z = lab[..., 0:1], lab[..., 1:2], lab[..., 2:3]

    # Vector scaling
    L = (116. * y) - 16.
    a = 500.0 * (x - y)
    b = 200.0 * (y - z)

    return torch.cat([L, a, b], dim=-1).permute(0,3,1,2)


eps = 1e-8
def downsample(features, assignment):
    """Downsamples the features with the given assignment matrix"""
    b, nr_feat, h, w = features.shape

    #Unfold the features
    features = F.unfold(features, kernel_size=2, stride=2).reshape(b * nr_feat, 4, int(h/2), int(w/2))
    features = F.unfold(features, kernel_size=3, padding=1).reshape(b, nr_feat, 4, 9, int(h/2), int(w/2))

    #Unfold the assignment
    assignment = F.unfold(assignment, kernel_size=2, stride=2).reshape(b, 36, int(h/2), int(w/2))
    assignment = F.unfold(assignment, kernel_size=3, padding=1)
    assignment = assignment.reshape(b, 9, 4, 9, int(h/2), int(w/2)).permute(0, 1, 3, 2, 4, 5)

    #Flip to take the diagonal from right to left
    assignment = torch.flip(assignment, dims=[1])
    assignment = torch.diagonal(assignment, dim1=1, dim2=2).permute(0, 1, -1, 2, 3)
    assignment = assignment.view(b, 1, 4, 9, int(h/2), int(w/2)).repeat(1, nr_feat, 1, 1, 1, 1)

    #Downsample features
    down_features = torch.sum(features * assignment, dim=(2, 3))
    down_features = torch.div((down_features), (torch.sum(assignment, dim=(2, 3)) + eps))
    
    return down_features


def upsample(output, assignment):
    """Upsamples the output with the given assignment matrix"""
    b, _, h, w = assignment.shape
    n_channels = output.size(1)

    #Get 9 candidate clusters and corresponding assignments
    candidate_clusters = F.unfold(output, kernel_size=3, padding=1).reshape(b, n_channels, 9, -1)
    assignment = F.unfold(assignment, kernel_size=2, stride=2).reshape(b, 9, 4, -1)
    
    #Linear decoding
    output = torch.einsum('bkcn,bcpn->bkpn', (candidate_clusters, assignment)).reshape(b, n_channels * 4, -1)
    output = F.fold(output, (h, w), kernel_size=2, stride=2)

    return output


def save_results(results, model, loss_function, data_set, label_set, test_set, path='Results/'):
    full_path = path + str(model) + '/' + str(loss_function) + '/' + data_set + '/' + label_set + '/'
    
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    
    #Make text file and/or open it
    file_name = f'results_{test_set}.txt'
    with open(full_path + file_name, 'a') as f:
        #Write the results
        f.write(str(results) + '\n')
        
        
def load_results(model, loss_function, data_set, label_set, test_set, avg=True, path='Results/'):
    full_path = path + str(model) + '/' + str(loss_function) + '/' + data_set + '/' + label_set + '/' + f'results_{test_set}.txt'

    if os.path.isfile(full_path):
        results = dict()

        with open(full_path, "r") as f:
            lines = f.readlines()
            nr_samples = len(lines)
            for line in lines:
                line = ast.literal_eval(line)
                for key in line:
                    try:
                        results[key].append(line[key])
                    except:
                        results[key] = [line[key]]

        if avg:
            for key in results:
                mean, std = np.mean(results[key]), np.std(results[key])
                results[key] = (mean, std)

            results['nr_samples'] = nr_samples
        return results

    else:
        return None
    
    
def save_model(model, loss_function, data_set, label_set, path='Models/'):
    full_path = path + str(model) + '/' + str(loss_function) + '/' + data_set + '/' + label_set + '/'

    if not os.path.exists(full_path):
        os.makedirs(full_path)

    version = len(os.listdir(full_path)) + 1
    full_path = full_path + f'weights_{version}.pt'
    
    torch.save(model.state_dict(), full_path)
    
    
def load_model(version, model, loss_function, data_set, label_set, path='Models/'):
    full_path = path + str(model) + '/' + str(loss_function) + '/' + data_set + '/' + label_set + '/' + f'weights_{version}.pt'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if os.path.isfile(full_path):
        #Load model weights
        model.load_state_dict(torch.load(full_path, map_location=device))
    else:
        print(f"Could not find model weights at {full_path}")

    return model.to(device)