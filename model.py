import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.models import resnet18


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


class CNNLSTM2(nn.Module):
    def __init__(self, num_classes=2, lstm_size=512, lstm_layers=2):
        super(CNNLSTM2, self).__init__()
        self.resnet = resnet18(pretrained=False)
        self.resnet.fc = nn.Sequential(nn.Linear(self.resnet.fc.in_features, 256))
        self.lstm = nn.LSTM(input_size=128, hidden_size=lstm_size, num_layers=lstm_layers, batch_first=True)
        self.fc1 = nn.Linear(lstm_size, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.lstm_size = lstm_size
        self.lstm_layers = lstm_layers

    def forward(self, x_3d):
        x_lstm = None

        for t in range(x_3d.size(1)):
            with torch.no_grad():
                x = self.resnet(x_3d[:, t, :, :, :])
                x_lstm = torch.cat((x_lstm, x.unsqueeze(1)), 1) if x_lstm is not None else x.unsqueeze(1)

        x_lstm = x_lstm.transpose(1, 2)
        # print("x_lstm: ", x_lstm.shape)
        if torch.cuda.is_available():
            x_lstm = x_lstm.cuda()
        batch_size = x_lstm.size(0)
        lstm_init = (torch.zeros(self.lstm_layers, batch_size, self.lstm_size),
                     torch.zeros(self.lstm_layers, batch_size, self.lstm_size))
        if x_lstm.is_cuda:
            lstm_init = (lstm_init[0].cuda(), lstm_init[0].cuda())
        lstm_init = (Variable(lstm_init[0], volatile=x_lstm.volatile), Variable(lstm_init[1], volatile=x_lstm.volatile))
        out = self.lstm(x_lstm, lstm_init)[0][:, -1, :]

        x = self.fc1(out)
        x = F.relu(x)
        x = self.fc2(x)
        return x
