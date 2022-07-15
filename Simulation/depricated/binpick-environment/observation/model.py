import torch
import torch.nn as nn

import torchsummary

class ObservationModel(nn.Module):

    def __init__(self, **kwargs):
        super(ObservationModel, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(5000 * 3, 4096),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, 2048),
            nn.LeakyReLU(0.1),
            nn.Linear(2048, 5000)
        )

    def forward(self, x):
        x = self.layers(x)

        return x

            