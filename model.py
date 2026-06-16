import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """A residual block with two conv layers and batch normalization.
    Preserves spatial dimensions via padding=1."""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual  # skip connection
        return F.relu(out)


class BomberlandCNN(nn.Module):
    def __init__(self, input_channels=9, num_actions=6):
        super(BomberlandCNN, self).__init__()

        # === Feature extraction backbone ===
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)

        self.res_block = ResidualBlock(128)

        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(128)

        # === FIXED: 1x1 Conv to reduce channels while PRESERVING spatial grid ===
        self.conv_reduce = nn.Conv2d(128, 32, kernel_size=1, bias=False)
        self.bn_reduce = nn.BatchNorm2d(32)
        
        # 32 channels * 13 width * 13 height = 5,408 dimensions
        self.flatten_dim = 32 * 13 * 13
        
        # === Shared FC trunk ===
        self.fc1 = nn.Linear(self.flatten_dim, 256)
        self.dropout = nn.Dropout(0.3)

        # === Dual heads ===
        self.policy_head = nn.Linear(256, num_actions)
        self.value_head = nn.Linear(256, 1)

    def forward(self, x):
        # Feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.res_block(x)
        x = F.relu(self.bn4(self.conv4(x)))

        # Reduce channels but preserve spatial alignment
        x = F.relu(self.bn_reduce(self.conv_reduce(x)))

        # Flatten explicitly to preserve coordinate positions
        x = x.view(x.size(0), -1)

        # Shared trunk
        x = self.dropout(F.relu(self.fc1(x)))

        # Dual heads
        policy_logits = self.policy_head(x)
        value = torch.tanh(self.value_head(x))

        return policy_logits, value