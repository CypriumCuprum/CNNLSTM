import unittest

from torch import nn
import torch
import torch.nn.functional as F


class TestMethods(unittest.TestCase):
    def test_CNNModel128_4L(self):
        inp = torch.rand(5, 128, 32, 32)
        model = CNNModel128_4L()
        output = model(inp)
        self.assertIsInstance(model, nn.Module)
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, (5, 40))

    def test_CNN_LSTM(self):
        inp = torch.rand(5, 14, 128, 32, 32)
        model = CNN_LSTM()
        output = model(inp)
        self.assertIsInstance(model, nn.Module)
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, (5, 2))


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
        self.fc1 = nn.Linear(256 * 8 * 8, 1024)
        self.dropout = nn.Dropout(p=0.2)

        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        # Convolutional layers
        x = self.conv1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.relu(x)

        x = self.pool(x)

        # Flatten the output
        x = x.view(-1, 256 * 8 * 8)

        # Fully connected layers
        x = self.fc1(x)
        # x = self.relu(x)
        # x = self.dropout(x)
        #
        # x = self.fc2(x)

        return x


class CNN_LSTM(nn.Module):
    def __init__(self, num_classes=40):
        super(CNN_LSTM, self).__init__()
        self.cnn = CNNModel128_4L(num_classes=256)
        # self.lstm = nn.LSTM(input_size=256, hidden_size=256, num_layers=2, bidirectional=True)
        # self.fc1 = nn.Linear(512, 128)
        # self.fc2 = nn.Linear(128, num_classes)
        self.lstm = nn.LSTM(input_size=1024, hidden_size=256, num_layers=2, bidirectional=True)
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x_3d):
        out = None
        hidden = None

        for t in range(x_3d.size(1)):
            with torch.no_grad():
                x = self.cnn(x_3d[:, t, :, :, :])
            out, hidden = self.lstm(x.unsqueeze(0), hidden)

        x = self.fc1(out[-1, :, :])
        x = F.relu(x)
        x = self.fc2(x)
        return x
