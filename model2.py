import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class CNN128Channels(nn.Module):
    def __init__(self, num_classes=2):
        super(CNN128Channels, self).__init__()
        self.resnet18 = resnet18(pretrained=False)
        self.resnet18.fc = nn.Sequential(nn.Linear(self.resnet18.fc.in_features, 256))
        self.fc1 = nn.Linear(256 * 128, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x1 = torch.tensor(())
        if torch.cuda.is_available():
            x1 = x1.cuda()
        for i in range(x.size(1)):
            out = self.resnet18(x[:, i, :, :, :])
            # out = out.unsqueeze(1)
            x1 = torch.cat((x1, out), 1)

        x = self.fc1(x1)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class CNNLSTM(nn.Module):
    def __init__(self, num_classes=2):
        super(CNNLSTM, self).__init__()
        self.resnet = resnet18(pretrained=True)
        self.resnet.fc = nn.Sequential(nn.Linear(self.resnet.fc.in_features, 256))
        self.lstm = nn.LSTM(input_size=256, hidden_size=256, num_layers=2, bidirectional=True)
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x_3d):
        out = None
        hidden = None

        for t in range(x_3d.size(1)):
            with torch.no_grad():
                x = self.resnet(x_3d[:, t, :, :, :])
                print(x)
            print(x.unsqueeze(0).shape)
            out, hidden = self.lstm(x.unsqueeze(0), hidden)

        x = self.fc1(out[-1, :, :])
        x = F.relu(x)
        x = self.fc2(x)
        return x
