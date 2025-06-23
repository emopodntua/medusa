import torch
import torch.nn as nn
import torch.nn.functional as F
from .pooling import *
from .leg import *


def mixup_features(x1, labels, alpha=0.2):

    batch_size, len_x1 = x1.size()
    _, len_labels = labels.size()
    
    # Create a Beta distribution and sample one lambda per batch element
    beta_dist = torch.distributions.Beta(alpha, alpha)
    lam = beta_dist.sample((batch_size,)).to(x1.device)  # Shape: (batch_size,)

    # Randomly shuffle the batch
    indices = torch.randperm(batch_size, device=x1.device)

    # Reshape lam to broadcast over features and labels
    # lam_features = lam.expand(batch_size, len_x1)  # Shape: (batch_size, 1) for broadcasting with x1
    # lam_labels = lam.expand(batch_size, len_labels)   # Shape: (batch_size, 1) for broadcasting with labels

    lam_features = lam.view(-1, 1)  # Shape: (batch_size, 1) for broadcasting with x1
    lam_labels = lam.view(-1, 1)   # Shape: (batch_size, 1) for broadcasting with labels

    # Perform mixup on inputs
    x1_mixed = lam_features * x1 + (1 - lam_features) * x1[indices]

    # Perform mixup on labels
    y_mixed = lam_labels * labels + (1 - lam_labels) * labels[indices]

    return x1_mixed, y_mixed
        


class DeepSER(nn.Module):
    def __init__(self, embed_dims, mlp_outs, mixup=0.0):
        super(DeepSER, self).__init__()
        self.mixup = mixup
        
        # Create a ModuleList to properly register the DeepLeg modules
        self.legs = nn.ModuleList()
        for i in range(len(embed_dims)):
            self.legs.append(DeepLeg(embed_dims[i], mlp_outs[i]))
            
        # Additional legs
        self.leg4 = DeepLeg(mlp_outs[-1], mlp_outs[-1])
        self.leg5 = DeepLeg(mlp_outs[-1], mlp_outs[-1])
        
        self.linear1 = nn.Linear(sum(mlp_outs), mlp_outs[-1])
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(mlp_outs[-1], 8)  # number of (categorical) emotions
        self.linear3 = nn.Linear(mlp_outs[-1], 3)  # number of emotional attributes
        # self.softmax = nn.Softmax(dim=-1)


    def forward(self, xs, labels=None, dev=False):
        h1 = [None] * len(xs)
        h2 = [None] * len(xs)
        h3 = [None] * len(xs)
        
        for i in range(len(xs)):
            h1[i], h2[i], h3[i] = self.legs[i](xs[i])
        # print(f1.shape, f2.shape, f3.shape)

        f1 = torch.cat(h1, dim = 1)
        h2 = torch.cat(h2, dim = 1) ##
        h3 = torch.cat(h3, dim = 1) ##

        # print(cf.shape)

        f1_1, f1_2, f1_3 = self.leg4(f1)

        f2 = torch.cat((h2, f1_2), dim = 1)

        f2_1, f2_2, f2_3 = self.leg5(f2)

        # print(x1.shape, x2.shape)

        x = torch.cat((h3, f2_3), dim=-1) # caution!

        x = self.linear1(x)

        p = torch.rand(size=(1,), device=x.device)

        if p < self.mixup and labels is not None and not dev:
            x, labels = mixup_features(x, labels)

        x = self.relu(x)
        y = self.linear3(x)

        x = self.linear2(x)
        y = 1 + 6*torch.sigmoid(y)
        # x = self.softmax(x)
        
        if self.mixup > 0.0 and labels is not None and not dev:
            return x, y, labels
        else:
            return x, y