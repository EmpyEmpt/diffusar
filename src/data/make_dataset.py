import json
import cv2
import numpy as np
import artifacts as ar
from torchvision import transforms

from torch.utils.data import Dataset

random_artifact = [
    ar.apply_color_banding,
    ar.apply_compression_artifact,
    ar.apply_distortion,
    ar.apply_ghosting,
    ar.apply_moire_pattern,
    ar.apply_noise,
    ar.apply_rainbow_effect,
    # ar.apply_screen_tearing
]


class ArtifactDataset(Dataset):
    """
    Dataset for training Diffusar
    """
    def __init__(self, images_path: str, annotations_path: str, dynamic_source: bool = True, image_size: int = 512):
        self.data = []
        self.images_path = images_path
        self.annotations_path = annotations_path
        self.dynamic_source = dynamic_source
        self.image_size = image_size

        self.tfs = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((image_size, image_size)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        with open(self.annotations_path, 'r') as f:
            file = json.load(f)['tags']
            for dataline in file:
                self.data.append(dataline)

    def __getitem__(self, idx):
        item = self.data[idx]

        # loading string values
        source_filename = item['source']
        target_filename = item['target']

        # loading images
        target = cv2.imread(self.images_path + target_filename)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        
        if self.dynamic_source:
            source = np.random.choice(random_artifact)(target, random_args=True)
        else:
            source = cv2.imread(self.images_path + source_filename)
            source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)

        source = self.tfs(source)
        target = self.tfs(target)

        ret = {
            'target_image': target,
            'source_image': source,
            'path': source_filename
        }
        return ret
    
    def __len__(self):
        return len(self.data)
    