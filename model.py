import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam
from torchmetrics import Precision, Recall, F1Score, Accuracy


class MultiHeadedCNN(nn.Module):
    def __init__(self):
        super(MultiHeadedCNN, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=(1, 5)),
            nn.ReLU(),

            nn.Conv2d(64, 256, kernel_size=(1, 4)),
            nn.ReLU(),

            nn.Conv2d(256, 1024, kernel_size=(1, 3)),
            nn.ReLU(),

            nn.Conv2d(1024, 2048, kernel_size=(1, 3)),
            nn.ReLU(),

            nn.Conv2d(2048, 512, kernel_size=(1, 2)),
            nn.ReLU(),
        )

        self.num_flatten = 1 * 6 * 512  # 3072 flatten neural

        self.fc_layers = nn.Sequential(
            nn.Linear(3072, 3072),
            nn.ReLU(),
            nn.Linear(3072, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 288),
            nn.ReLU(),
            nn.Linear(288, 288),
            nn.ReLU(),
        )

        self.head1 = nn.Linear(288, 4)
        self.head2 = nn.Linear(288, 3)
        self.head3 = nn.Linear(288, 4)
        self.head4 = nn.Linear(288, 6)

    #         # Apply Kaiming Initialization
    #         for m in self.modules():
    #             if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
    #                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    #                 if m.bias is not None:
    #                     nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)

        output1 = self.head1(x)
        output2 = self.head2(x)
        output3 = self.head3(x)
        output4 = self.head4(x)

        return [output1, output2, output3, output4]