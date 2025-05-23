import torch
import torch.nn as nn


class CNN_Classifier(nn.Module):
    def __init__(self, n_samples, n_counters):
        super(CNN_Classifier, self).__init__()

        n_kernels = 5

        self.conv11 = nn.ModuleList(
            [nn.Conv2d(1, 1, (1, n_counters)) for k in range(n_kernels)]
        )
        self.layer1 = nn.Linear(n_samples * n_kernels, 128)
        self.out = nn.Linear(128, 6)

        self.act_fn = nn.LeakyReLU()

    def forward(self, x):
        # add channel dimension
        x = x.unsqueeze(1)

        x = [self.act_fn(conv(x)).squeeze(3) for conv in self.conv11]
        x = torch.cat(x, dim=2)
        x = x.squeeze(1)

        x = self.layer1(x)
        x = self.act_fn(x)

        x = self.out(x)

        return x
