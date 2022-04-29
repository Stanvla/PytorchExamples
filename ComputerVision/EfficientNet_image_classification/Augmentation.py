import torch
import torch.nn as nn
import torchvision.transforms as T


class MyRandomAdjustSharpness(nn.Module):
    def __init__(self, start, end, n, prob=0.5):
        super().__init__()
        self.n = n
        self.sharpness_factors = torch.linspace(start, end, n)
        self.adj_sharpness_transforms = PerformAtMostN(
            [T.RandomAdjustSharpness(sharpness_factor=f, p=1) for f in self.sharpness_factors],
            prob,
            n=1
        )

    def forward(self, batch):
        return self.adj_sharpness_transforms(batch)


class PerformAtMostN(nn.Module):
    def __init__(self, transforms, prob, n):
        super().__init__()
        self.n = n
        self.transforms = transforms
        self.prob = prob

    def forward(self, batch):
        if self.prob < torch.rand(1):
            return batch

        n_transforms = torch.randperm(self.n)[0]
        idxs = torch.randperm(len(self.transforms))[:n_transforms]
        for i in idxs:
            batch = self.transforms[i](batch)
        return batch
