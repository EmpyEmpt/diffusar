import json
import cv2
import numpy as np
import artifacts as ar
from PIL import Image

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
    def __init__(self, images_path: str, annotations_path: str, use_prompts: bool = False, dynamic_source: bool = True, image_size: int = 512):
        self.data = []
        self.use_prompts = use_prompts
        self.images_path = images_path
        self.annotations_path = annotations_path
        self.dynamic_source = dynamic_source
        self.image_size = image_size

        with open(self.annotations_path, 'r') as f:
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
        target = cv2.imread(self.images_path + target_filename)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        target = self.__resize_with_pad(target, self.image_size)

        if self.dynamic_source:
            source = np.random.choice(random_artifact)(target, random_args=True)
        else:
            source = cv2.imread(self.images_path + source_filename)
            source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
            source = self.__resize_with_pad(source, self.image_size)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)

    def __resize_with_pad(self, image, desired_size):
        im = Image.fromarray(image)
        old_size = im.size

        ratio = float(desired_size)/max(old_size)
        new_size = tuple([int(x*ratio) for x in old_size])

        im = im.resize(new_size, Image.Resampling.LANCZOS)

        new_im = Image.new("RGB", (desired_size, desired_size))
        new_im.paste(im, ((desired_size-new_size[0])//2,
                          (desired_size-new_size[1])//2))

        image = np.array(new_im)
        return image
