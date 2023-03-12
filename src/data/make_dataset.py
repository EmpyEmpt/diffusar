import json
import cv2
import numpy as np

from torch.utils.data import Dataset

# TODO: dynamically create source images
# TODO: dynamically create prompt based on the artifacts used in source


class MyDataset(Dataset):
    def __init__(self, dataset_path: str, annotations_filename: str):
        self.data = []
        self.dataset_path = dataset_path
        self.annotations_filename = annotations_filename
        with open(self.dataset_path + self.annotations_filename, 'rt') as f:
            file = json.load(f)['tags']
            for dataline in file:
                self.data.append(dataline)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        source = cv2.imread(self.dataset_path + source_filename)
        target = cv2.imread(self.dataset_path + target_filename)

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)
