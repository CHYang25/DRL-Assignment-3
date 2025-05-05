import torch
import torch.nn as nn

class ConvNet(nn.Module):
    def __init__(self, in_channels=3, feature_dim=512):
        super(ConvNet, self).__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4), 
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, feature_dim),  
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv_net(x)
        x = self.fc(x)
        return x

class QNet(nn.Module):
    def __init__(self, in_channels, feature_dim, n_actions, n_hidden = 128):
        super(QNet, self).__init__()
        self.ft_extractor = ConvNet(in_channels, feature_dim)
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, n_actions)
        )

    def forward(self, x):
        y = self.ft_extractor(x)
        return self.fc(y)