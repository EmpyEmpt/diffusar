import json
import cv2
import numpy as np
import artifacts as ar

from torch.utils.data import Dataset

random_artifact = [
    ar.apply_color_banding,
    ar.apply_compression_artifact,
    ar.apply_distortion,
    ar.apply_ghosting,
    ar.apply_moire_pattern,
    ar.apply_noise,
    ar.apply_rainbow_effect,
    ar.apply_screen_tearing
]


class MyDataset(Dataset):
    def __init__(self, dataset_path: str, annotations_path: str, use_prompts: bool = False, dynamic_source=True):
        self.data = []
        self.use_prompts = use_prompts
        self.dataset_path = dataset_path
        self.annotations_path = annotations_path
        self.dynamic_source = dynamic_source

        with open(self.dataset_path + self.annotations_path, 'rt') as f:
            file = json.load(f)['tags']
            for dataline in file:
                self.data.append(dataline)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # loading string values
        source_filename = item['source']
        target_filename = item['target']
        prompt = ''
        if self.use_prompts:
            prompt = item['prompt']

        # loading images
        source = cv2.imread(self.dataset_path + source_filename)
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)

        if not self.dynamic_source:
            target = cv2.imread(self.dataset_path + target_filename)
            target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        else:
            target = np.random.choice(
                random_artifact(source, random_args=True))

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)
