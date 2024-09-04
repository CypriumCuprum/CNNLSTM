import unittest

import torch
import torch.nn as nn


class TestCNNModel(unittest.TestCase):
    def test_CNNModel128(self):
        num_classes = 10
        test_model = CNNModel128(num_classes=num_classes)
        inp = torch.rand(1, 128, 32, 32)
        output = test_model(inp)
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, (1, 10))
        self.assertEqual(output.shape[1], num_classes)


class CNNModel128(nn.Module):
    def __init__(self, num_classes=40):
        super(CNNModel128, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 8 * 8, 1024)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)

        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        # Convolutional layers
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        # Flatten the output
        x = x.view(-1, 128 * 8 * 8)

        # Fully connected layers
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.dropout(x)

        x = self.fc2(x)

        return x


class CNNModel128_4L(nn.Module):
    def __init__(self, num_classes=40):
        super(CNNModel128_4L, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear(256 * 4 * 4, 1024)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)

        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        # Convolutional layers
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        # Flatten the output
        x = x.view(-1, 128 * 8 * 8)

        # Fully connected layers
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.dropout(x)

        x = self.fc2(x)

        return x


if __name__ == "__main__":
    unittest.main()
    # Create an instance of the CNN model
    model = CNNModel128()

    # Print the model summary
    print(model)

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# class ResidualBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, stride=1):
#         super(ResidualBlock, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(out_channels)
#
#         # Shortcut connection
#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_channels != out_channels:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(out_channels)
#             )
#
#     def forward(self, x):
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out += self.shortcut(x)
#         out = self.relu(out)
#         return out


# class ResNetLike(nn.Module):
#     def __init__(self, num_classes=10):
#         super(ResNetLike, self).__init__()
#         self.in_channels = 64
#
#         self.conv1 = nn.Conv2d(128, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#
#         self.layer1 = self._make_layer(64, 2, stride=1)
#         self.layer2 = self._make_layer(128, 2, stride=2)
#         self.layer3 = self._make_layer(256, 2, stride=2)
#
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(256, num_classes)
#
#     def _make_layer(self, out_channels, blocks, stride):
#         layers = []
#         layers.append(ResidualBlock(self.in_channels, out_channels, stride))
#         self.in_channels = out_channels
#         for _ in range(1, blocks):
#             layers.append(ResidualBlock(self.in_channels, out_channels))
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#
#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.fc(x)
#
#         return x
#
#
# # Example usage
# model = ResNetLike(num_classes=10)
# print(model)
