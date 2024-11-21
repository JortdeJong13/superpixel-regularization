eps = 1e-8

def downsample(features, assignment):
    """
    Downsamples the features using the assignment matrix.
    Args:
        features (Tensor): Input feature tensor of shape (B, C, H, W).
        assignment (Tensor): Assignment tensor of shape (B, 9, H, W).
    Returns:
        Tensor: Downsampled feature tensor of shape (B, C, H//2, W//2).
    """
    b, nr_feat, h, w = features.shape

    if h % 2 != 0 or w % 2 != 0:
        raise ValueError("Input height and width must be divisible by 2.")

    # Unfold the features
    features = F.unfold(features, kernel_size=2, stride=2).reshape(b * nr_feat, 4, h // 2, w // 2)
    features = F.unfold(features, kernel_size=3, padding=1).reshape(b, nr_feat, 4, 9, h // 2, w // 2)

    # Unfold the assignment
    assignment = F.unfold(assignment, kernel_size=2, stride=2).reshape(b, 36, h // 2, w // 2)
    assignment = F.unfold(assignment, kernel_size=3, padding=1)
    assignment = assignment.reshape(b, 9, 4, 9, h // 2, w // 2).permute(0, 1, 3, 2, 4, 5)

    # Flip to take the diagonal from right to left
    assignment = torch.flip(assignment, dims=[1])
    assignment = torch.diagonal(assignment, dim1=1, dim2=2).permute(0, 1, -1, 2, 3)
    assignment = assignment.view(b, 1, 4, 9, h // 2, w // 2).repeat(1, nr_feat, 1, 1, 1, 1)

    # Downsample features
    down_features = torch.sum(features * assignment, dim=(2, 3))
    down_features = torch.div(down_features, (torch.sum(assignment, dim=(2, 3)) + eps))
    
    return down_features
