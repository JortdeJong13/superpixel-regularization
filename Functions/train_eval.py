import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from Functions.loss_functions import CrossEntropy, RegularisedCrossEntropy
from Functions.model_blocks import *
from Functions.utils import downsample, upsample

def train(model, optimizer, loss_function, train_loader, val_loader, epochs=50, verbose=True):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.train()
    max_val_acc = 0

    epochs = range(epochs) if verbose else tqdm(range(epochs))
    for epoch in epochs:
        tmp_loss = []
        tmp_acc = []
        for image, label in train_loader:
            image, label = image.to(device), label.to(device)

            assignments, output = model(image)
            loss = loss_function(assignments, output, image, label)
            tmp_loss.append(loss.item())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            _, pred = torch.max(output, 1)
            acc = torch.sum(pred.eq(label), dim=(1, 2)) / (torch.sum(label != 255, dim=(1, 2)) + 1e-6)
            tmp_acc.extend(acc.cpu().tolist())

        train_loss = sum(tmp_loss) / len(tmp_loss)
        train_acc = sum(tmp_acc) / len(tmp_acc)
        if verbose:
            print(f"Epoch: {epoch + 1}, \ttrain loss: {round(train_loss, 2):.2f}, \ttrain accuracy: {round(train_acc * 100, 2)}%")

        ###Validation###
        tmp_val_loss = []
        tmp_val_acc = []
        with torch.no_grad():
            for image, label in val_loader:
                image, label = image.to(device), label.to(device)

                assignments, output = model(image)
                loss = loss_function(assignments, output, image, label)
                tmp_val_loss.append(loss.item())

                _, pred = torch.max(output, 1)
                acc = torch.sum(pred.eq(label), dim=(1, 2)) / (torch.sum(label != 255, dim=(1, 2)) + 1e-6)
                tmp_val_acc.extend(acc.cpu().tolist())

        val_loss = sum(tmp_val_loss) / len(tmp_val_loss)
        val_acc = sum(tmp_val_acc) / len(tmp_val_acc)
        if verbose:
            print(f"Epoch: {epoch + 1}, \tval loss: {round(val_loss, 2):.2f}, \tval accuracy: {round(val_acc * 100, 2)}%")

        if val_acc > max_val_acc:
            max_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pt')

    #Load best model
    model.load_state_dict(torch.load('best_model.pt'))
    
    
def train_run(train_loader, val_loader, hc, lmbda=None, m=None, epochs=50, nr_classes=29):
    #Initialize model
    encoder = ResNet34Encoder()
    classifier = FCNHead(256, nr_classes)
    if hc:
        decoder = HCDecoder(tau=0.07, sim_feat=48)
        model = SegModel(encoder, classifier, decoder)
    else:
        model = SegModel(encoder, classifier, interpolate)       

    if lmbda==0 or lmbda==None:
        loss = CrossEntropy()
    else:
        loss = RegularisedCrossEntropy(lmbda=lmbda, m=m)

    #Train model
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    train(model.to('cuda' if torch.cuda.is_available() else 'cpu'), optimizer, loss, train_loader, val_loader, epochs=epochs, verbose=False)

    return model


def finetune(model, loss_function, train_loader, val_loader, train_val_loader, max_epochs=15, verbose=False):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #Get training loss
    model.eval()
    with torch.no_grad():
        prev_train_loss = []
        for image, label in train_loader:
            image, label = image.to(device), label.to(device)

            assignments, output = model(image)
            loss = loss_function(assignments, output, image, label)
            prev_train_loss.append(loss.item())

    prev_train_loss = sum(prev_train_loss) / len(prev_train_loss)
    if verbose:
        print(f'Previous train loss: {round(prev_train_loss, 2):.2f}')

    #Finetune
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    epoch = 0
    val_loss = prev_train_loss + 1

    while (epoch < max_epochs) and val_loss > prev_train_loss:
        tmp_loss = []
        tmp_acc = []
        for image, label in train_val_loader:
            image, label = image.to(device), label.to(device)

            assignments, output = model(image)
            loss = loss_function(assignments, output, image, label)
            tmp_loss.append(loss.item())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            _, pred = torch.max(output, 1)
            acc = torch.sum(pred.eq(label), dim=(1, 2)) / (torch.sum(label != 255, dim=(1, 2)) + 1e-6)
            tmp_acc.extend(acc.cpu().tolist())

        train_loss = sum(tmp_loss) / len(tmp_loss)
        train_acc = sum(tmp_acc) / len(tmp_acc)
        if verbose:
            print(f"Epoch: {epoch + 1}, \ttrain loss: {round(train_loss, 2):.2f}, \ttrain accuracy: {round(train_acc * 100, 2)}%")

        ###Validation###
        tmp_val_loss = []
        tmp_val_acc = []
        with torch.no_grad():
            for image, label in val_loader:
                image, label = image.to(device), label.to(device)

                assignments, output = model(image)
                loss = loss_function(assignments, output, image, label)
                tmp_val_loss.append(loss.item())

                _, pred = torch.max(output, 1)
                acc = torch.sum(pred.eq(label), dim=(1, 2)) / (torch.sum(label != 255, dim=(1, 2)) + 1e-6)
                tmp_val_acc.extend(acc.cpu().tolist())

        val_loss = sum(tmp_val_loss) / len(tmp_val_loss)
        val_acc = sum(tmp_val_acc) / len(tmp_val_acc)
        if verbose:
            print(f"Epoch: {epoch + 1}, \tval loss: {round(val_loss, 2):.2f}, \tval accuracy: {round(val_acc * 100, 2)}%")
        epoch += 1
        
        
def get_boundary_mask(labels, thickness, ignore_index=255):
    num_classes = torch.max(labels[labels!=ignore_index]) + 1
    
    #Make label a one-hot label
    hot_label = labels.clone()
    hot_label[hot_label==ignore_index] = num_classes
    hot_label = F.one_hot(hot_label, num_classes=num_classes+1).permute(-1, 0, 1, 2)
    hot_label[-1, :, :, :] = hot_label[-1, :, :, :] * -1 + 1          #Invert ignore_class
    hot_label = hot_label.reshape(-1, 1, hot_label.size(-2), hot_label.size(-1)) #Combine batch and classes
    hot_label = nn.ConstantPad2d(thickness-1, 1.0)(hot_label)

    #Construct edge detector kernel
    kernel_size = thickness*2-1
    kernel = torch.ones(size=(1, 1, kernel_size, kernel_size)) * -1
    kernel[0][0][thickness-1][thickness-1] = kernel_size**2 -1
    kernel = kernel.repeat((1, 1, 1, 1))

    output = F.conv2d(hot_label.float(), kernel.to(hot_label.device))

    #Construct boundary mask
    mask = torch.squeeze(output) > 0
    mask = mask.view(num_classes + 1, labels.size(0), mask.size(-2), mask.size(-1))
    ignore_mask = mask[-1]
    mask = mask[:num_classes]
    mask = torch.sum(mask, dim=0).bool()
    mask = mask * ~ignore_mask

    return mask.bool()


def get_boundary_recall(pred, label, ignore_index=255):
    b, h, w = label.shape
    r = np.round(np.sqrt(h**2 + w**2) * 0.0025) 

    #Get boundary maps and neighborhoods
    pred_boundary = get_boundary_mask(pred, 2, ignore_index=ignore_index)
    pred_hood = get_boundary_mask(pred, int(r+1), ignore_index=ignore_index)
    label_boundary = get_boundary_mask(label, 2, ignore_index=ignore_index)
    label_hood = get_boundary_mask(label, int(r+1), ignore_index=ignore_index)

    tp = torch.sum(pred_boundary * label_hood, dim=(1, 2))
    fn = torch.sum(~pred_hood * label_boundary, dim=(1, 2))

    return torch.div(tp, tp + fn + 0.00001)


def get_asa(assignments, label, ignore_index=255):
    num_classes = torch.max(label[label!=ignore_index]) + 1
    pooled_label = label.clone()

    #Make label one hot label
    pooled_label[label==ignore_index] = num_classes
    pooled_label = F.one_hot(pooled_label, num_classes=num_classes+1).permute(0, -1, 1, 2)
    pooled_label = pooled_label[:, :num_classes, :, :]

    #Downsample with interpolation
    sorted_keys = np.sort(list(assignments.keys()))
    size = (int(pooled_label.size(-2) / sorted_keys[0]), int(pooled_label.size(-1) / sorted_keys[0]))
    pooled_label = F.interpolate(pooled_label.to(torch.float32), size=size, mode="bilinear", align_corners=False)

    #Downsample with superpixels
    for key in sorted_keys:
        pooled_label = downsample(pooled_label, assignments[key])

    #Upsample with superpixels
    for key in sorted_keys[::-1]:
        pooled_label = upsample(pooled_label, assignments[key])

    #Upsample with interpolation
    pooled_label = F.interpolate(pooled_label, size=label.shape[-2:], mode="bilinear", align_corners=False)

    #Compute Achievable Segmentation Accuracy
    _, pooled_label = torch.max(pooled_label, dim=1)
    acc = torch.sum(pooled_label.eq(label), dim=(1, 2)) / (torch.sum(label!=ignore_index, dim=(1, 2)) + 1e-6)

    return acc


def eval_model(model, test_loader, ignore_index=255):
    acc_list = []
    boundary_recall_list = []
    asa_list = []

    model.eval()
    with torch.no_grad():
        for image, label in test_loader:
            image, label = image.to('cuda' if torch.cuda.is_available() else 'cpu'), label.to('cuda' if torch.cuda.is_available() else 'cpu')
            assignments, output = model(image)
            _, pred = torch.max(output, 1)

            #Accuracy
            acc = torch.sum(pred.eq(label), dim=(1, 2)) / (torch.sum(label!=ignore_index, dim=(1, 2)) + 1e-6)
            acc_list.extend(acc.cpu().tolist())

            #Boundary recall
            boundary_recall = get_boundary_recall(pred, label, ignore_index=ignore_index)
            boundary_recall_list.extend(boundary_recall.cpu().tolist())

            #Achievable Segmentation Accuracy
            if assignments:
                asa = get_asa(assignments, label)
                asa_list.extend(asa.cpu().tolist())

    results = {'acc': np.mean(acc_list),
               'boundary_recall': np.mean(boundary_recall_list)}

    if asa_list:
        results['asa'] = np.mean(asa_list)

    return results


def print_results(results):
    if results:
        asa = f"\tASA: {round(results['asa'][0], 3)} ± {round(results['asa'][1], 4)}" if 'asa' in results else ''
        print(f"Average over {results['nr_samples']} samples  \tAccuracy: {round(results['acc'][0], 3)} ± {round(results['acc'][1], 4)} \
        \tBoundary recall: {round(results['boundary_recall'][0], 3)} ± {round(results['boundary_recall'][1], 4)} \
        {asa}")
    else:
        print("No results found!")