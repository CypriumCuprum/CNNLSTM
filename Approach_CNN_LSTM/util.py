import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import EEGDataset128Channel, Splitter
from model import CNN_LSTM, CNN4L_LSTM, Flex_Model, Flex_Model_noFC
import pandas as pd

parser = argparse.ArgumentParser(description="Test model")

parser.add_argument("--model_type", type=str, default="CNN_LSTM", help="Model type")
parser.add_argument("--filedata", type=str, default="../data/eeg_signals_32x32_128.pth", help="Path to data file")
parser.add_argument("--splits_path", type=str, default="data/block_splits_by_image_all.pth", help="Path to split file")
parser.add_argument("--start", type=int, default=0, help="Start index for dataset")
parser.add_argument("--end", type=int, default=-1, help="End index for dataset")
parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
parser.add_argument("--split_num", type=int, default=0,
                    help="Split number")  # should always set 0 to avoid error when make split if split is not exist
parser.add_argument("--model_path", type=str, default="", help="Path to save model")
parser.add_argument("--save_file", type=str, default="runtime_inference.txt", help="Path to save model")
parser.add_argument("--num_cnn_layers", type=int, default=3, help="Number of CNN layers")
parser.add_argument("--num_lstm_layers", type=int, default=2, help="Number of LSTM layers")
parser.add_argument("--is_resnet", type=bool, default=False, help="Use ResNet")
parser.add_argument("--is_bidirectional", type=bool, default=False, help="Use Bidirectional LSTM")

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EEGData = EEGDataset128Channel(args.filedata, args.start, args.end)

loader = {split: DataLoader(Splitter(EEGData, args.splits_path, args.split_num, split), batch_size=args.batch_size,
                            shuffle=True) for split in ["train", "val", "test"]}

model = CNN_LSTM()
if args.model_type == "CNN_LSTM":
    model = CNN_LSTM()
elif args.model_type == "CNN4L_LSTM":
    model = CNN4L_LSTM()
elif args.model_type == "Flex_Model":
    model = Flex_Model(args.num_cnn_layers, args.num_lstm_layers, args.is_resnet, args.is_bidirectional)
elif args.model_type == "Flex_Model_noFC":
    model = Flex_Model_noFC(args.num_cnn_layers, args.num_lstm_layers, args.is_resnet, args.is_bidirectional)

correct = 0
total = 0

checkpoint = torch.load(args.model_path, weights_only=True)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device).eval()
confusion_matrix = torch.zeros(40, 40, dtype=torch.int64)
with torch.no_grad():
    for i, (signal, label) in enumerate(loader["val"]):
        signal = signal.to(device)
        label = label.to(device, dtype=torch.int64)

        # Forward
        output = model(signal)

        # Accuracy
        _, predicted = output.max(1)
        for j in range(len(predicted)):
            confusion_matrix[label[j]][predicted[j]] += 1

sticks = ['sorrel ', 'Parachute', 'Iron', 'Anemone', 'Espresso maker', 'Coffee mug', 'Bike', 'Revolver', 'Panda',
          'Daisy', 'Canoe', 'Lycaenid', 'Dog', 'Running Shoe', 'Lantern', 'Cellular phone', 'Golf ball',
          'Computer', 'Broom', 'Pizza', 'Missile', 'Capuchin', 'Pool table', 'Mailbag', 'Convertible',
          'Folding chair', 'Pajama', 'Mitten', 'Electric guitar', 'Reflex camera', 'Piano', 'Mountain tent',
          'Banana', 'Bolete', 'Watch', 'Elephant', 'Airliner', 'Locomotive', 'Telescope', 'Egyptian cat']
confusion_matrix = confusion_matrix.numpy()

cm = confusion_matrix
# Save cm to csv
df_cm = pd.DataFrame(cm, index=sticks, columns=sticks)
df_cm.to_csv(args.save_file.replace('.csv', '_confusion_matrix.csv'), index=True)

# calculate precision, recall, f1 score
precision = torch.zeros(40)
recall = torch.zeros(40)
f1 = torch.zeros(40)

for i in range(40):
    precision[i] = confusion_matrix[i][i] / confusion_matrix[:, i].sum()
    recall[i] = confusion_matrix[i][i] / confusion_matrix[i].sum()
    f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i])

# create dataframes with 2 columns and 40 rows
df = pd.DataFrame(columns=['Label', 'Accuracy'])
df['Label'] = ['sorrel ', 'Parachute', 'Iron', 'Anemone', 'Espresso maker', 'Coffee mug', 'Bike', 'Revolver', 'Panda',
               'Daisy', 'Canoe', 'Lycaenid', 'Dog', 'Running Shoe', 'Lantern', 'Cellular phone', 'Golf ball',
               'Computer', 'Broom', 'Pizza', 'Missile', 'Capuchin', 'Pool table', 'Mailbag', 'Convertible',
               'Folding chair', 'Pajama', 'Mitten', 'Electric guitar', 'Reflex camera', 'Piano', 'Mountain tent',
               'Banana', 'Bolete', 'Watch', 'Elephant', 'Airliner', 'Locomotive', 'Telescope', 'Egyptian cat']
df['Accuracy'] = [(confusion_matrix[i][i] / confusion_matrix[i].sum()) * 100 for i in range(40)]
# save to csv
df.to_csv(args.save_file, index=True)

# Precision, Recall, F1 score
df2 = pd.DataFrame(columns=['Label', 'Precision', 'Recall', 'F1'])
df2['Label'] = ['Sorrel ', 'Parachute', 'Iron', 'Anemone', 'Espresso maker', 'Coffee mug', 'Bike', 'Revolver', 'Panda',
                'Daisy', 'Canoe', 'Lycaenid', 'Dog', 'Running Shoe', 'Lantern', 'Cellular phone', 'Golf ball',
                'Computer', 'Broom', 'Pizza', 'Missile', 'Capuchin', 'Pool table', 'Mailbag', 'Convertible',
                'Folding chair', 'Pajama', 'Mitten', 'Electric guitar', 'Reflex camera', 'Piano', 'Mountain tent',
                'Banana', 'Bolete', 'Watch', 'Elephant', 'Airliner', 'Locomotive', 'Telescope', 'Egyptian cat']
df2['Precision'] = np.round(precision.numpy() * 100, 2)

df2['Recall'] = np.round(recall.numpy() * 100, 2)
df2['F1'] = np.round(f1.numpy() * 100, 2)

df2.to_csv(args.save_file.replace('.csv', '_precision_recall_f1.csv'), index=False)

# sample run: python util.py --model_type Flex_Model_noFC --filedata ./data/eegsplit_bytime --splits_path ./data/block_splits_by_image_all.pth --model_path ./history/Flex_Model_noFC_015948_3_3_False_True/checkpoint/checkpoint_best.pth --save_file ./history/accuracy_by_label.csv --num_cnn_layers 3 --num_lstm_layers 3 --is_bidirectional 1
