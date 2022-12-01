# Improving segmentation boundary alignment when training on coarse labels

In this repository a PyTorch implementation of the superpixel downsampling operation is provided.
The downsampling operation can be found in the python file [downsample.py](https://github.com/JortdeJong13/superpixel_regularization/blob/main/downsample.py)

Equation 3 downsamples a full resolution feature map x with the assignment matrices A before upsampling with the same assignment matrices. A pixel in the resulting feature map contains the average features of its corresponding superpixels. This function is essential to the regularization term. Equation 3 consists of two distinct steps. Downsampling a full resolution feature map x with the assignment matrices A. And upsampling the resulting low resolution feature map with the same assignment matrices. In [Implicit Integration of Superpixel Segmentation into Fully Convolutional Networks](https://arxiv.org/pdf/2103.03435) T. Suzuki provides a PyTorch implementation for the latter step. In this repository we provide a PyTorch implementation to downsample a feature map with a given assignment matrix. Downsampling a feature map x with the assignment matrices A is done by recursively downsampling the feature map x with A(s′) for s′ ∈ {1, 2, 4, 8}. The provided downsample function is compatible with the PyTorch code provided.

contribution
