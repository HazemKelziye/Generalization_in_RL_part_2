import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam
from torchmetrics import Precision, Recall, F1Score, Accuracy


class MultiHeadedCNN(nn.Module):
    def __init__(self, num_classes):
        super(MultiHeadedCNN, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(16, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=(1, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=(1, 3)),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=(1, 3)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.num_flatten = 1 * 6 * 256  # 1536 flatten neural

        self.fc_layers = nn.Sequential(
            nn.Linear(1536, 768),
            nn.ReLU(),
            nn.Linear(768, 384),
            nn.ReLU(),
            nn.Linear(384, 96),
            nn.ReLU(),
            nn.Linear(96, 48),
            nn.ReLU()
        )

        self.head1 = nn.Linear(48, 4)
        self.head2 = nn.Linear(48, 3)
        self.head3 = nn.Linear(48, 4)
        self.head4 = nn.Linear(48, 6)

        # Apply Kaiming Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv_layers(x)
        #         print("Conv output shape before flattening:", x.shape)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)

        output1 = self.head1(x)
        output2 = self.head2(x)
        output3 = self.head3(x)
        output4 = self.head4(x)

        return [output1, output2, output3, output4]