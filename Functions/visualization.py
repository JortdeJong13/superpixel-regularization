import random
import matplotlib.pyplot as plt
from PIL import Image, ImageOps, ImageEnhance
import torch
import torchvision.transforms as T
from skimage.segmentation import mark_boundaries
import seaborn as sns
import numpy as np
from Functions.utils import *

def add_img(axarr, idx, img, title):
    if hasattr(img, 'mode') and img.mode == 'L':
        axarr[idx].imshow(img, cmap='gray')
    else:
        axarr[idx].imshow(img)
    axarr[idx].set_title(title)
    axarr[idx].axis('off')
    
    
def overlay_label(img, label, classes):
    img = ImageEnhance.Brightness(ImageOps.grayscale(img)).enhance(1.3)
    palette = []
    for clas in classes[1:]:
        for channel in [*clas[2]]:
            palette.append(channel)
    label.putpalette(palette)
    img = img.convert("RGBA")
    label = label.convert("RGBA")
    label = Image.blend(img, label, 0.15)

    return label

def show_sample(dataset, classes, idx=None):
    if idx==None:
        idx = random.randint(0, len(dataset)-1)
    image, label = dataset[idx]

    f, axarr = plt.subplots(1, 2, figsize=(2*8, 8))
    image = T.functional.to_pil_image(image)
    add_img(axarr, 0, image, "Image")
    
    label = T.ToPILImage()(np.array(torch.squeeze(label), dtype='int32')).convert("L")
    label = overlay_label(image, label, classes)
    add_img(axarr, 1, label, "Label")
    plt.show()
    
    
def show_predict(model, dataset, classes, idx=None, crop=None):
    if idx==None:
        idx = random.randint(0, len(dataset)-1)
    image, label = dataset[idx]
    if crop:
        image, label = T.functional.center_crop(image, crop), T.functional.center_crop(label, crop)

    label = torch.unsqueeze(label.to('cuda' if torch.cuda.is_available() else 'cpu'), dim=0)

    model.eval()
    assignments, pred = model(torch.unsqueeze(image, dim=0).to('cuda' if torch.cuda.is_available() else 'cpu'))
    _, pred = torch.max(pred, 1)
    acc = float(torch.sum(pred.eq(label), dim=(1, 2)) / (torch.sum(label != 255, dim=(1, 2)) + 1e-6))

    f, axarr = plt.subplots(1, 3, figsize=(3*8, 8))
    image = T.functional.to_pil_image(image)
    add_img(axarr, 0, image, "Image")
    
    label = T.ToPILImage()(np.array(torch.squeeze(label.cpu()), dtype='int32')).convert("L")
    label = overlay_label(image, label, classes)
    add_img(axarr, 1, label, "Label")

    pred = T.ToPILImage()(np.array(torch.squeeze(pred.cpu()), dtype='int32')).convert("L")
    pred = overlay_label(image, pred, classes)
    add_img(axarr, 2, pred, f"Prediction, acc: {round(acc*100)}%")

    plt.show()
    
    
def show_superpixels(model, dataset, classes, idx=None):
    if idx==None:
        idx = random.randint(0, len(dataset)-1)
    image, label = dataset[idx]
    label = torch.unsqueeze(label, dim=0)

    model.eval()
    with torch.no_grad():
        assignments, pred = model(torch.unsqueeze(image, dim=0).to('cuda' if torch.cuda.is_available() else 'cpu'))
        
    _, pred = torch.max(pred.cpu(), 1)
    acc = float(torch.sum(pred.eq(label), dim=(1, 2)) / (torch.sum(label != 255, dim=(1, 2)) + 1e-6))

    f, axarr = plt.subplots(2, 2, figsize=(2*10, 2*5))
    image = T.functional.to_pil_image(image)
    add_img(axarr, (0, 0), image, "Image")

    label = T.ToPILImage()(np.array(torch.squeeze(label), dtype='int32')).convert("L")
    label = overlay_label(image, label, classes)
    add_img(axarr, (0, 1), label, "Label")

    segments = extract_superpixels(assignments)
    segments = mark_boundaries(image, segments)
    add_img(axarr, (1, 0), segments, "Extracted superpixels")

    pred = T.ToPILImage()(np.array(torch.squeeze(pred.cpu()), dtype='int32')).convert("L")
    pred = overlay_label(image, pred, classes)
    add_img(axarr, (1, 1), pred, f"Prediction, acc: {round(acc*100)}%")

    plt.show()
    
    
def plot_lmbda(lmbda_list, m, data_set, test_set, label_set):
    acc_list = []
    acc_std = []
    boundary_recall_list = []
    boundary_recall_std = []

    for lmbda in lmbda_list:
        loss_function = f'reg_cross_entropy_{lmbda}_{m}' if lmbda != 0 else 'cross_entropy'
        results = load_results('HCFCN', loss_function, data_set, label_set, test_set)
        if results:
            acc_list.append(results['acc'][0])
            acc_std.append(results['acc'][1])
            boundary_recall_list.append(results['boundary_recall'][0])
            boundary_recall_std.append(results['boundary_recall'][1])

    #Plotting
    fig, axes = plt.subplots(1, 2, figsize=(20, 5))
    axes[0].plot(lmbda_list, acc_list, 'o-', color='black')
    axes[0].fill_between(lmbda_list, np.array(acc_list) + np.array(acc_std), 
                         np.array(acc_list) - np.array(acc_std), alpha=0.5, color='gray')
    axes[0].tick_params(axis='both', which='major', labelsize=12)
    axes[0].set_xticks(lmbda_list)
    axes[0].set_xlabel("Lambda", fontsize=14)
    axes[0].set_ylabel('Pixel accuracy', fontsize=14)

    axes[1].plot(lmbda_list, boundary_recall_list, 'o-', color='black')
    axes[1].fill_between(lmbda_list, np.array(boundary_recall_list) + np.array(boundary_recall_std), 
                         np.array(boundary_recall_list) - np.array(boundary_recall_std), alpha=0.5, color='gray')
    axes[1].tick_params(axis='both', which='major', labelsize=12)
    axes[1].set_xticks(lmbda_list)
    axes[1].set_xlabel("Lambda", fontsize=14)
    axes[1].set_ylabel('Boundary recall', fontsize=14)
    plt.show()
    
    
def plot_m(m_list, lmbda, data_set, test_set, label_set):
    acc_list = []
    acc_std = []
    boundary_recall_list = []
    boundary_recall_std = []

    for m in m_list:
        results = load_results('HCFCN', f'reg_cross_entropy_{lmbda}_{m}', data_set, label_set, test_set)
        if results:
            acc_list.append(results['acc'][0])
            acc_std.append(results['acc'][1])
            boundary_recall_list.append(results['boundary_recall'][0])
            boundary_recall_std.append(results['boundary_recall'][1])

    #Plotting
    fig, axes = plt.subplots(1, 2, figsize=(20, 5))
    axes[0].plot(m_list, acc_list, 'o-', color='black')
    axes[0].fill_between(m_list, np.array(acc_list) + np.array(acc_std), 
                         np.array(acc_list) - np.array(acc_std), alpha=0.5, color='gray')
    axes[0].tick_params(axis='both', which='major', labelsize=12)
    axes[0].set_xticks(m_list)
    axes[0].set_xlabel("m", fontsize=14, fontstyle='italic')
    axes[0].set_ylabel('Pixel accuracy', fontsize=14)

    axes[1].plot(m_list, boundary_recall_list, 'o-', color='black')
    axes[1].fill_between(m_list, np.array(boundary_recall_list) + np.array(boundary_recall_std), 
                         np.array(boundary_recall_list) - np.array(boundary_recall_std), alpha=0.5, color='gray')
    axes[1].tick_params(axis='both', which='major', labelsize=12)
    axes[1].set_xticks(m_list)
    axes[1].set_xlabel("m", fontsize=14, fontstyle='italic')
    axes[1].set_ylabel('Boundary recall', fontsize=14)
    plt.show()
    
    
def plot_hyperparameters(lmbda_list, m_list, data_set, test_set):
    #Get results
    m_list = list(reversed(m_list))
    acc = np.ones((len(m_list), len(lmbda_list))) * np.NaN
    boundary_recall = np.ones((len(m_list), len(lmbda_list))) * np.NaN
    count = np.ones((len(m_list), len(lmbda_list))) * np.NaN

    for m_index, m in enumerate(m_list):
        for lmbda_index, lmbda in enumerate(lmbda_list):
            loss_function = 'cross_entropy' if lmbda==0 and m==0 else f'reg_cross_entropy_{lmbda}_{m}'
            results = load_results('HCFCN', loss_function, data_set, 'coarse', test_set)
            if results:
                count[m_index, lmbda_index] = results['nr_samples']
                acc[m_index, lmbda_index] = results['acc'][0]
                boundary_recall[m_index, lmbda_index] = results['boundary_recall'][0]

    #Plotting
    fig, axes = plt.subplots(1, 3, figsize=(20, 4))

    sns.heatmap(acc, xticklabels=lmbda_list, yticklabels=m_list, annot=True, cbar=False, fmt='.3f', cmap="YlGnBu", ax=axes[0])
    axes[0].set_xlabel('Lambda')
    axes[0].set_ylabel('m')
    axes[0].set_title(f"Accuracy heatmap")

    sns.heatmap(boundary_recall, xticklabels=lmbda_list, yticklabels=m_list, annot=True, cbar=False, fmt='.3f', cmap="YlGnBu", ax=axes[1])
    axes[1].set_xlabel('Lambda')
    axes[1].set_ylabel('m')
    axes[1].set_title(f"Boundary recall heatmap")

    sns.heatmap(count, xticklabels=lmbda_list, yticklabels=m_list, annot=True, cbar=False, cmap="YlGnBu", ax=axes[2])
    axes[2].set_xlabel('Lambda')
    axes[2].set_ylabel('m')
    axes[2].set_title(f"Number of samples")
    plt.show()

    
def plot_subset(full_size, nr_samples_list, data_set, test_set, label_set, lmbda, m):
    acc_list = []
    acc_std = []
    reg_acc_list = []
    reg_acc_std = []

    for nr_samples in [''] + [f'_{nr_samples}' for nr_samples in nr_samples_list]:
        results = load_results(data_set, test_set, label_set + nr_samples, True, lmbda=0, m=m)
        acc_list.append(results['acc'][0])
        acc_std.append(results['acc'][1])

        results = load_results(data_set, test_set, label_set + nr_samples, True, lmbda=lmbda, m=m)
        reg_acc_list.append(results['acc'][0])
        reg_acc_std.append(results['acc'][1])

    x = np.array([full_size] + nr_samples_list)
    x_labels = [f'{nr_samples}\n{round(100*nr_samples/full_size)}%' for nr_samples in x]

    #Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(x, acc_list, 'o-', label='HCFCN-16')
    plt.fill_between(x, np.array(acc_list) + np.array(acc_std), 
                         np.array(acc_list) - np.array(acc_std), alpha=0.5)
    plt.plot(x, reg_acc_list, 'o-', label='Regularized HCFCN-16')
    plt.fill_between(x, np.array(reg_acc_list) + np.array(reg_acc_std), 
                         np.array(reg_acc_list) - np.array(reg_acc_std), alpha=0.5)
    
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.xticks(x, x_labels)
    plt.gca().invert_xaxis()
    plt.xlabel("Number of training samples", fontsize=14)
    plt.ylabel('Pixel accuracy', fontsize=14)
    plt.legend(fontsize=12)
    plt.show()
    
    
def plot_coarseness(model_list, erode_list, data_set, test_set, full_c=25):
    model_acc = {model[0]:([], []) for model in model_list} 
    model_br = {model[0]:([], []) for model in model_list} 

    for erode in erode_list:
        if erode==1:
            tmp_label_set = 'full_fine'
        elif erode==full_c:
            tmp_label_set = 'full_coarse'
        else:
            tmp_label_set = 'full_coarse' + f'_{erode}'

        for model in model_list:
            results = load_results(model[1], model[2], data_set, tmp_label_set, test_set)
            model_acc[model[0]][0].append(results['acc'][0])
            model_acc[model[0]][1].append(results['acc'][1])
            model_br[model[0]][0].append(results['boundary_recall'][0])
            model_br[model[0]][1].append(results['boundary_recall'][1])
            
    #Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5.5), sharex=True)
    erode_list = (np.array(erode_list) - 1) / 2
    for model in model_list:
        ax1.plot(erode_list, model_acc[model[0]][0], 'o-', label=model[0])
        ax1.fill_between(erode_list, np.array(model_acc[model[0]][0]) + np.array(model_acc[model[0]][1]),
                        np.array(model_acc[model[0]][0]) - np.array(model_acc[model[0]][1]), alpha=0.5)
        
        ax2.plot(erode_list, model_br[model[0]][0], 'o-', label=model[0])
        ax2.fill_between(erode_list, np.array(model_br[model[0]][0]) + np.array(model_br[model[0]][1]),
                        np.array(model_br[model[0]][0]) - np.array(model_br[model[0]][1]), alpha=0.5)
    
    ax1.tick_params(axis='both', which='major', labelsize=14)
    ax2.tick_params(axis='both', which='major', labelsize=14)
    ax1.set_ylabel('Pixel accuracy', fontsize=14)
    ax2.set_xlabel("Number of eroded pixels", fontsize=14)
    ax2.set_ylabel('Boundary recall', fontsize=14)
    ax1.legend(fontsize=14, bbox_to_anchor=(0, 1, 1, 0), loc="lower left", mode="expand", ncol=2)
    plt.show()
    
    
def show_boundary_alignment(model, dataset, classes, idx, wx, wy, hx, hy):    
    image, label = dataset[idx]
    label = torch.unsqueeze(label, dim=0)

    model.eval()
    with torch.no_grad():
        assignments, pred = model(torch.unsqueeze(image, dim=0).to('cuda' if torch.cuda.is_available() else 'cpu'))

    _, pred = torch.max(pred.cpu(), 1)

    f, axarr = plt.subplots(1, 3, figsize=(12, 7))
    image = T.functional.to_pil_image(image[:, hx:hy, wx:wy])
    label = T.ToPILImage()(np.array(torch.squeeze(label[:, hx:hy, wx:wy]), dtype='int32')).convert("L")
    label = overlay_label(image, label, classes)
    add_img(axarr, 0, label, "Label")

    segments = extract_superpixels(assignments)[hx:hy, wx:wy]
    segments = mark_boundaries(np.array(image), segments)
    add_img(axarr, 1, segments, "Extracted superpixels")

    pred = T.ToPILImage()(np.array(torch.squeeze(pred[:, hx:hy, wx:wy].cpu()), dtype='int32')).convert("L")
    pred = overlay_label(image, pred, classes)
    add_img(axarr, 2, pred, f"Prediction")

    plt.show()
    