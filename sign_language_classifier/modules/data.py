import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import cv2
from typing import Optional, Any, Tuple


class SignLanguageMNISTDataset(Dataset):

    def __init__(self, csv_path: str, img_size: int = 224, train: bool = True):
        self.csv = pd.read_csv(csv_path)
        self.img_size = img_size
        self.train = train

        text = "pixel"
        self.images = torch.zeros((self.csv.shape[0], 1))

        for i in range(1, 785):
            temp_text = text + str(i)
            temp = self.csv[temp_text]
            temp = torch.FloatTensor(temp).unsqueeze(1)
            self.images = torch.cat((self.images, temp), 1)

        self.labels = self.csv['label']
        self.images = self.images[:, 1:]
        self.images = self.images.view(-1, 28, 28)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Any]:
        img = self.images[index]
        img = img.numpy()
        img = cv2.resize(img, (self.img_size, self.img_size))

        tensor_image = torch.FloatTensor(img)
        tensor_image = tensor_image.unsqueeze(0)
        tensor_image /= 255.

        print(f"Getting item {index}") 

        if self.train:
            return tensor_image, self.labels[index]
        else:
            return tensor_image

    def __len__(self) -> int:
        return self.images.shape[0]


class SignLanguageMNISTDataModule(pl.LightningDataModule):

    def __init__(
        self,
        train_csv_path: Optional[str] = None,
        val_csv_path: Optional[str] = None,
        test_csv_path: Optional[str] = None,
        predict_csv_path: Optional[str] = None,
        img_size: int = 224,
        train_batch_size: int = 128,
        predict_batch_size: int = 64,
        num_workers: int = 4,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.train_csv_path = train_csv_path
        self.val_csv_path = val_csv_path
        self.test_csv_path = test_csv_path
        self.predict_csv_path = predict_csv_path
        self.num_workers = num_workers

        self.img_size = img_size
        self.train_batch_size = train_batch_size
        self.predict_batch_size = predict_batch_size

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train_dataset = SignLanguageMNISTDataset(csv_path=self.train_csv_path, img_size=self.img_size, train=True)
            self.val_dataset = SignLanguageMNISTDataset(csv_path=self.val_csv_path, img_size=self.img_size, train=True)
        elif stage == "validate":
            self.val_dataset = SignLanguageMNISTDataset(csv_path=self.val_csv_path, img_size=self.img_size, train=True)
        elif stage == "test":
            self.test_dataset = SignLanguageMNISTDataset(csv_path=self.test_csv_path, img_size=self.img_size, train=True)
        elif stage == "predict":
            if self.predict_csv_path:
                self.predict_dataset = SignLanguageMNISTDataset(csv_path=self.predict_csv_path, img_size=self.img_size, train=False)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.train_dataset, num_workers=self.num_workers, batch_size=self.train_batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.val_dataset, num_workers=self.num_workers, batch_size=self.predict_batch_size, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.test_dataset, num_workers=self.num_workers, batch_size=self.predict_batch_size, shuffle=False)

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.predict_dataset, num_workers=self.num_workers, batch_size=self.predict_batch_size, shuffle=False)

    @property
    def num_classes(self) -> int:
        return 25

    @property
    def in_channels(self) -> int:
        return 1