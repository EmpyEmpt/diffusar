# I want to kill myself
import sys
sys.path.insert(0, './src/data')
from make_dataset import ArtifactDataset
import torch.utils.data as data
from torchvision import transforms
from torchvision.datasets import ImageFolder
from pytorch_lightning import LightningDataModule


class DiffusarData(LightningDataModule):
    def __init__(self, train_images_path, train_annotations_path, batch_size, dataloader_workers, val_images_path=None, val_annotations_path=None):
        super().__init__()

        self.batch_size = batch_size
        self.dataloader_workers = dataloader_workers

        self.train_images_path = train_images_path
        self.train_annotations_path = train_annotations_path

        self.val_images_path = None
        self.val_annotations_path = None

        if val_images_path is not None or val_annotations_path is not None:
            self.val_images_path = val_images_path
            self.val_annotations_path = val_annotations_path

    def setup(self, stage):
        self.train_dataset = ArtifactDataset(
            images_path=self.train_images_path,
            annotations_path=self.train_annotations_path,
        )

        if self.val_images_path is None or self.val_annotations_path is None:
            self.val_dataset = None
            return

        self.val_dataset = ArtifactDataset(
            images_path=self.val_images_path,
            annotations_path=self.val_annotations_path,
        )

    def train_dataloader(self):
        return data.DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.dataloader_workers)

    def val_dataloader(self):
        if self.val_dataset is None:
            return None
        return data.DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.dataloader_workers)

class InferenceData(LightningDataModule):
    def __init__(self, images_path, batch_size, dataloader_workers = 1):
        super().__init__()

        self.batch_size = batch_size
        self.dataloader_workers = dataloader_workers

        self.images_path = images_path

        self.tfs = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def setup(self, stage):
        self.dataset = ImageFolder(
            root=self.images_path,
            transform=self.tfs
        )

    def predict_dataloader(self):
        return data.DataLoader(self.dataset, batch_size=self.batch_size, num_workers=self.dataloader_workers)
