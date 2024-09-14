import unittest

from torch import nn
import torch


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
        self.assertEqual(output.shape, (5, 40))


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


# class CNN_LSTM(nn.Module):
#     def __init__(self, num_classes=40):
#         super(CNN_LSTM, self).__init__()
#         self.cnn = CNNModel128_4L(num_classes=256)
#         # self.lstm = nn.LSTM(input_size=256, hidden_size=256, num_layers=2, bidirectional=True)
#
#         # self.lstm = nn.LSTM(input_size=1024, hidden_size=256, num_layers=2, bidirectional=True)
#         # self.fc1 = nn.Linear(512, 128)
#         # self.fc2 = nn.Linear(128, num_classes)
#
#         self.fc1 = nn.Linear(14 * 1024, 256)
#         self.fc2 = nn.Linear(256, num_classes)
#
#     def forward(self, x_3d):
#         out = torch.Tensor(()).to(x_3d.device)
#         # out = None
#         # hidden = None
#
#         for t in range(x_3d.size(1)):
#             with torch.no_grad():
#                 x = self.cnn(x_3d[:, t, :, :, :])
#             # out, hidden = self.lstm(x.unsqueeze(0), hidden)
#             out = torch.cat((out, x), dim=1)
#
#         # x = self.fc1(out[-1, :, :])
#         x = self.fc1(out)
#         x = F.relu(x)
#         x = self.fc2(x)
#         return x

class CNN_LSTM(nn.Module):
    def __init__(self, num_classes=40):
        super(CNN_LSTM, self).__init__()

        # CNN for processing each time step's image
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (64, 16, 16)

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (128, 8, 8)

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Output: (256, 4, 4)
        )

        # Flatten CNN output to feed into LSTM
        self.flatten = nn.Flatten()

        # LSTM to handle temporal dependencies
        self.lstm = nn.LSTM(input_size=256 * 4 * 4, hidden_size=512, num_layers=2, batch_first=True)

        # Fully connected layer for classification or output
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        batch_size, time_steps, channels, height, width = x.size()

        # Initialize hidden state for LSTM
        h0 = torch.zeros(2, batch_size, 512).to(x.device)  # num_layers=2, hidden_size=512
        c0 = torch.zeros(2, batch_size, 512).to(x.device)  # num_layers=2, hidden_size=512

        # CNN for each time step
        cnn_out = []
        for t in range(time_steps):
            cnn_feature = self.cnn(x[:, t, :, :, :])  # Apply CNN to each time step
            cnn_feature = self.flatten(cnn_feature)  # Flatten output for LSTM
            cnn_out.append(cnn_feature)

        cnn_out = torch.stack(cnn_out, dim=1)  # Shape: (batch_size, time_steps, cnn_output_dim)

        # LSTM for temporal sequence
        lstm_out, _ = self.lstm(cnn_out, (h0, c0))

        # Use the last output of LSTM for final classification
        final_output = lstm_out[:, -1, :]

        # Pass through fully connected layer
        output = self.fc(final_output)

        return output


class CNN4L_LSTM(nn.Module):
    def __init__(self, num_classes=40):
        super(CNN4L_LSTM, self).__init__()

        # CNN for processing each time step's image
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (128, 16, 16)

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Output: (512, 8, 8)
        )

        # Flatten CNN output to feed into LSTM
        self.flatten = nn.Flatten()

        # LSTM to handle temporal dependencies
        self.lstm = nn.LSTM(input_size=512 * 8 * 8, hidden_size=1024, num_layers=2, batch_first=True)

        # Fully connected layer for classification or output
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        batch_size, time_steps, channels, height, width = x.size()

        # Initialize hidden state for LSTM
        h0 = torch.zeros(2, batch_size, 1024).to(x.device)  # num_layers=2, hidden_size=1024
        c0 = torch.zeros(2, batch_size, 1024).to(x.device)  # num_layers=2, hidden_size=1024

        # CNN for each time step
        cnn_out = []
        for t in range(time_steps):
            cnn_feature = self.cnn(x[:, t, :, :, :])  # Apply CNN to each time step
            cnn_feature = self.flatten(cnn_feature)  # Flatten output for LSTM
            cnn_out.append(cnn_feature)

        cnn_out = torch.stack(cnn_out, dim=1)  # Shape: (batch_size, time_steps, cnn_output_dim)

        # LSTM for temporal sequence
        lstm_out, _ = self.lstm(cnn_out, (h0, c0))

        # Use the last output of LSTM for final classification
        final_output = lstm_out[:, -1, :]

        # Pass through fully connected layer
        output = self.fc(final_output)

        return output
