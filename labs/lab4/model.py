import torch
import torch.nn as nn
import torch.nn.functional as F


class _BNReluConv(nn.Sequential):
    def __init__(self, num_maps_in, num_maps_out, k=3, bias=True):
        super(_BNReluConv, self).__init__()
        # YOUR CODE HERE
        self.append(nn.BatchNorm2d(num_maps_in))
        self.append(nn.ReLU())
        self.append(nn.Conv2d(num_maps_in, num_maps_out, kernel_size=k, padding=k//2, bias=bias))

class SimpleMetricEmbedding(nn.Module):
    def __init__(self, input_channels, emb_size=32):
        super().__init__()
        self.emb_size = emb_size
        # YOUR CODE HERE
        self.conv1 = _BNReluConv(input_channels, emb_size, k=3)
        self.conv2 = _BNReluConv(emb_size, emb_size, k=3)
        self.conv3 = _BNReluConv(emb_size, emb_size, k=3)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)

    def get_features(self, img):
        # Returns tensor with dimensions BATCH_SIZE, EMB_SIZE
        # YOUR CODE HERE
        x = self.conv1(img)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = x.mean(dim=[-1, -2])
        return x

    def loss(self, anchor, positive, negative):
        a_x = self.get_features(anchor)
        p_x = self.get_features(positive)
        n_x = self.get_features(negative)
        # YOUR CODE HERE
        d_pos = F.pairwise_distance(a_x, p_x, keepdim=True)
        d_neg = F.pairwise_distance(a_x, n_x, keepdim=True)
        margin = 1.0
        loss = F.relu(d_pos - d_neg + margin).mean()
        return loss
    


class IdentityModel(nn.Module):
    def __init__(self):
        super(IdentityModel, self).__init__()

    def get_features(self, img):
        # YOUR CODE HERE
        feats = img.view(img.size(0), -1)
        return feats