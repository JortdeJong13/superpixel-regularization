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
