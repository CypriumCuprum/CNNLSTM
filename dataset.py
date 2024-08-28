import os

import torch
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import Dataset
from torchvision import transforms


class FireDataset(Dataset):
    def __init__(self, root, transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])):
        self.root = root
        self.transform = transform
        self.all_class = os.listdir(root)

        self.x = []
        self.y = []

        for (aa, i) in enumerate(self.all_class):
            path = os.path.join(self.root, i)

            all_images = os.listdir(path)

            for j in range(0, len(all_images), 15):
                images = torch.tensor(())

                for k in all_images[j: j + 15]:
                    img = Image.open(os.path.join(path, k)).convert('RGB').resize((224, 224))
                    img = self.transform(img)
                    img = Variable(torch.unsqueeze(img, 0))

                    images = torch.cat((images, img), 0)

                self.x.append(images)
                self.y.append(aa)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        images = self.x[index]
        label = torch.tensor(self.y[index])

        return images, label


class EEG_ImageDataset(Dataset):
    def __init__(self, root, transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]), numclass=10):
        self.root = root
        self.transform = transform
        self.all_class = os.listdir(root)
        self.all_class = self.all_class[:numclass]

        self.x = []
        self.y = []
        for (i, label) in enumerate(self.all_class):
            path = os.path.join(self.root, label)

            data_list = os.listdir(path)
            for data in data_list:
                self.x.append(data)
                self.y.append(label)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        data = self.x[index]
        label = self.y[index]

        all_images = os.listdir(os.path.join(self.root, label, data))
        images = torch.tensor(())
        for img in all_images[:15]:
            img = Image.open(os.path.join(self.root, label, data, img)).convert('RGB').resize((224, 224))
            img = self.transform(img)
            img = Variable(torch.unsqueeze(img, 0))

            images = torch.cat((images, img), 0)

        label = torch.tensor(int(label))

        return images, label
