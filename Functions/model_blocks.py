import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import math
import segmentation_models_pytorch as smp

class DeformableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DeformableConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride if type(stride) == tuple else (stride, stride)
        self.padding = padding
        
        # init weight and bias
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels))

        # offset conv
        self.conv_offset_mask = nn.Conv2d(in_channels, 
                                          3 * kernel_size * kernel_size,
                                          kernel_size=kernel_size, 
                                          stride=stride,
                                          padding=self.padding, 
                                          bias=True)
        
        # init        
        self.reset_parameters()
        self._init_weight()


    def reset_parameters(self):
        n = self.in_channels * (self.kernel_size**2)
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.zero_()


    def _init_weight(self):
        # init offset_mask conv
        nn.init.constant_(self.conv_offset_mask.weight, 0.)
        nn.init.constant_(self.conv_offset_mask.bias, 0.)


    def forward(self, x):
        out = self.conv_offset_mask(x)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)

        x = torchvision.ops.deform_conv2d(input=x, 
                                          offset=offset, 
                                          weight=self.weight, 
                                          bias=self.bias, 
                                          padding=self.padding,
                                          mask=mask,
                                          stride=self.stride)
        return x
    
    
class BasicBlock(nn.Module):
    def __init__(self, in_c, out_c, downsample=False):
        super().__init__()
        if downsample:
            self.conv1 = DeformableConv2d(in_c, out_c, kernel_size=3, stride=2, padding=1)
            self.downsample = nn.Sequential(
                DeformableConv2d(in_c, out_c, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_c))
        else:
            self.conv1 = nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False)
            self.downsample = nn.Identity()
        
        self.bn1 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
            

    def forward(self, x):
        identity = self.downsample(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out
    

class ResNet34Encoder(nn.Module):
    def __init__(self, input_channels=3):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        layer_1 = []
        for idx in range(0, 3):
            layer_1.append(BasicBlock(64, 64, downsample=False))
        self.layer_1 = nn.Sequential(*layer_1)

        layer_2 = []
        self.layer_2_down = BasicBlock(64, 128, downsample=True)
        for idx in range(1, 4):
            layer_2.append(BasicBlock(128, 128, downsample=False))
        self.layer_2 = nn.Sequential(*layer_2)

        layer_3 = []
        self.layer_3_down = BasicBlock(128, 256, downsample=True)
        for idx in range(1, 6):
            layer_3.append(BasicBlock(256, 256, downsample=False))
        self.layer_3 = nn.Sequential(*layer_3)


    def forward(self, x):
        features = dict()
        features['x'] = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer_1(x)

        features['x_4'] = x
        x = self.layer_2_down(x)
        features['s_8'] = x
        x = self.layer_2(x)

        features['x_8'] = x
        x = self.layer_3_down(x)
        features['s_16'] = x
        x = self.layer_3(x)

        return features, x
    
    
class FCNHead(nn.Sequential):
    def __init__(self, in_channels: int, nr_classes: int) -> None:
        inter_channels = in_channels // 4
        layers = [
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, nr_classes, 1)]

        super().__init__(*layers)
        
        
def interpolate(features, output):
    """Interpolate output to the original input size"""
    output = F.interpolate(output, size=features['x'].shape[-2:], mode="bilinear", align_corners=False)
    
    return None, output


class HCDecoder(nn.Module):
    def __init__(self, tau, sim_feat=64):
        super().__init__()
        self.tau = tau
        self.sim_feat = sim_feat
        self.strides = [4, 8]

        #Initialise weigths
        self.weights = nn.ModuleDict()
        self.weights['x_4'] = nn.Conv2d(64, sim_feat, kernel_size=1, bias=False)
        self.weights['s_8'] = nn.Conv2d(128, sim_feat, kernel_size=1, bias=False)
        self.weights['x_8'] = nn.Conv2d(128, sim_feat, kernel_size=1, bias=False)
        self.weights['s_16'] = nn.Conv2d(256, sim_feat, kernel_size=1, bias=False)


    def cluster(self, s, feat, downsampled_feat):
        """Computes the soft assignment matrix A^s"""
        b, _, h, w = feat.shape

        feat = self.weights[f'x_{s}'](feat)
        downsampled_feat = self.weights[f's_{s*2}'](downsampled_feat)

        candidate_clusters = F.unfold(downsampled_feat, kernel_size=3, padding=1).reshape(b, self.sim_feat, 9, -1)
        feat = F.unfold(feat, kernel_size=2, stride=2).reshape(b, self.sim_feat, 4, -1)
        similarities = torch.einsum('bkcn,bkpn->bcpn', (candidate_clusters, feat)).reshape(b * 9, 4, -1)
        similarities = F.fold(similarities, (h, w), kernel_size=2, stride=2).reshape(b, 9, h, w)
        assignment = (similarities / self.tau).softmax(1)

        return assignment


    def upsample(self, output, assignment):
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


    def forward(self, features, output):
        #Get Assignments (superpixels)
        assignments = dict()
        for s in reversed(self.strides):
            feat = features[f'x_{s}']
            downsampled_feat = features[f's_{s*2}']
            assignment = self.cluster(s, feat, downsampled_feat)
            assignments[s] = assignment

            #Upscale output with superpixels
            output = self.upsample(output, assignment)

        #Remaining is bilinear upsampling
        output = F.interpolate(output, size=features['x'].shape[-2:], mode="bilinear", align_corners=False)

        return assignments, output
    

class SegModel(nn.Module):
    def __init__(self, encoder, classifier, decoder):
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier
        self.decoder = decoder

    
    def __str__(self):
        if hasattr(self.decoder, 'tau'):
            return 'HCFCN'
        else:
            return 'FCN' 

    def forward(self, x):
        features, out_feat = self.encoder(x)
        output = self.classifier(out_feat)
        assignments, output = self.decoder(features, output)

        return assignments, output


class UNet(nn.Module):
    def __init__(self, nr_classes):
        super().__init__()
        self.model = smp.Unet(
            encoder_name="resnet34",
            encoder_depth=4,
            encoder_weights=None,  
            decoder_channels=(128, 64, 32, 16),
            in_channels=3,
            classes=nr_classes)
        
        self.model.encoder.layer2[0].conv1 = DeformableConv2d(self.model.encoder.layer2[0].conv1.in_channels,
                                                              self.model.encoder.layer2[0].conv1.out_channels,
                                                              kernel_size=3, stride=2, padding=1)

        self.model.encoder.layer2[0].downsample[0] = DeformableConv2d(self.model.encoder.layer2[0].downsample[0].in_channels,
                                                                      self.model.encoder.layer2[0].downsample[0].out_channels,
                                                                      kernel_size=3, stride=2, padding=1)

        self.model.encoder.layer3[0].conv1 = DeformableConv2d(self.model.encoder.layer3[0].conv1.in_channels, 
                                                              self.model.encoder.layer3[0].conv1.out_channels,
                                                              kernel_size=3, stride=2, padding=1)

        self.model.encoder.layer3[0].downsample[0] = DeformableConv2d(self.model.encoder.layer3[0].downsample[0].in_channels,
                                                                      self.model.encoder.layer3[0].downsample[0].out_channels,
                                                                      kernel_size=3, stride=2, padding=1)
        
    
    def __str__(self):
        return 'UNet'

    def forward(self, x):
        output = self.model(x)

        return None, output