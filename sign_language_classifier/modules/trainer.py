import pytorch_lightning as pl
import torch
import torch.nn as nn
import numpy as np
from typing import Any, Optional, List
from sign_language_classifier.modules.model import ConvClassifier
import torchmetrics


class SignLanguageCNNModule(pl.LightningModule):

    def __init__(self, lr: float = 1e-3, num_classes: int = 25, in_channels: int = 1):
        super().__init__()
        self.save_hyperparameters()

        self.model = ConvClassifier(num_classes, in_channels)
        self.criterion = nn.CrossEntropyLoss()

        self.lr = lr
        self.num_classes = num_classes

        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)
        self.f1 = torchmetrics.F1Score(task='multiclass', num_classes=num_classes, average='weighted')
        self.test_accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)
        self.test_f1 = torchmetrics.F1Score(task='multiclass', num_classes=num_classes, average='weighted')

        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, inputs):
        return self.model(inputs)

    def training_step(self, batch: Any, batch_idx: int):
        images, labels = batch
        outputs = self.forward(images)
        loss = self.criterion(outputs, labels)

        preds = torch.softmax(outputs, dim=1)
        self.f1(preds, labels) 

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, logger=True)
        self.log("train_f1", self.f1, prog_bar=True, on_step=True, on_epoch=True, logger=True)

        return loss

    def validation_step(self, batch: Any):
        """
        Собираем предсказания для всей валидационной выборки,
        как в функции validate() из оригинального кода.
        """
        images, labels = batch
        outputs = self.forward(images)
        loss = self.criterion(outputs, labels)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, logger=True)

        preds = torch.softmax(outputs, dim=1)
        self.accuracy(preds, labels)
        self.log("val_accuracy", self.accuracy, prog_bar=True, on_step=False, on_epoch=True, logger=True)
        self.f1(preds, labels)
        self.log("val_f1", self.f1, prog_bar=True, on_step=False, on_epoch=True, logger=True)

        #return preds

        # Сохраняем предсказания и метки для накопления
        #_, predicted = torch.max(preds, 1)

        # Сохраняем в хранилище (для v2.0+)
        #step_output = {
        #    'preds': predicted.cpu().numpy(),
        #    'labels': labels.cpu().numpy()
        #}
        #self.validation_step_outputs.append(step_output)
        #return step_output

    # def on_validation_epoch_end(self):
    #     """
    #     Вычисляем accuracy и weighted F1 на всей валидационной выборке.
    #     Сохраняем точную логику оригинального validate().
    #     Используем self.validation_step_outputs вместо параметра.
    #     """
    #     # --- Оригинальная логика из validate() ---
    #     test_labels = [0]
    #     test_pred = [0]

    #     for outputs in self.validation_step_outputs:
    #         test_pred.extend(list(outputs['preds']))
    #         test_labels.extend(list(outputs['labels']))

    #     test_pred = np.array(test_pred[1:])
    #     test_labels = np.array(test_labels[1:])

    #     correct = (test_pred == test_labels).sum()
    #     accuracy_original = correct / len(test_labels)
    #     # -----------------------------------------

    #     # Получаем метрики из torchmetrics
    #     accuracy = self.accuracy.compute()
    #     f1 = self.f1.compute()

    #     # Логируем метрики
    #     self.log("val_accuracy", accuracy, prog_bar=True, on_epoch=True, logger=True)
    #     self.log("val_f1", f1, prog_bar=True, on_epoch=True, logger=True)
    #     self.log("val_accuracy_original", accuracy_original, prog_bar=True, on_epoch=True, logger=True)
    #     self.log("val_loss", 1 - accuracy, prog_bar=True, on_epoch=True)

    #     # Сбрасываем метрики и очищаем хранилище (важно для памяти!)
    #     self.accuracy.reset()
    #     self.f1.reset()
    #     self.validation_step_outputs.clear()

    #     return {"val_accuracy": accuracy, "val_f1": f1}

    def test_step(self, batch: Any, batch_idx: int):
        """Тестирование на одном батче"""
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)

        preds = torch.softmax(outputs, dim=1)

        self.test_accuracy(preds, labels)
        self.test_f1(preds, labels)

        self.log("test_loss", loss, prog_bar=True, on_step=False, on_epoch=True, logger=True)
        self.log("test_accuracy", self.test_accuracy, prog_bar=True, on_step=False, on_epoch=True, logger=True)
        self.log("test_f1", self.test_f1, prog_bar=True, on_step=False, on_epoch=True, logger=True)

    #     # Сохраняем предсказания и метки для накопления
    #     _, predicted = torch.max(preds, 1)

    #     step_output = {
    #         'preds': predicted.cpu().numpy(),
    #         'labels': labels.cpu().numpy()
    #     }
    #     self.test_step_outputs.append(step_output)

    #     return step_output

    # def on_test_epoch_end(self):
    #     """Вычисляем метрики на тестовой выборке"""
    #     # --- Оригинальная логика ---
    #     test_labels = [0]
    #     test_pred = [0]

    #     for outputs in self.test_step_outputs:
    #         test_pred.extend(list(outputs['preds']))
    #         test_labels.extend(list(outputs['labels']))

    #     test_pred = np.array(test_pred[1:])
    #     test_labels = np.array(test_labels[1:])

    #     correct = (test_pred == test_labels).sum()
    #     accuracy_original = correct / len(test_labels)
    #     # ---------------------------

    #     # Получаем метрики из torchmetrics
    #     accuracy = self.accuracy.compute()
    #     f1 = self.f1.compute()

    #     self.log("test_accuracy", accuracy, prog_bar=True, logger=True)
    #     self.log("test_f1", f1, prog_bar=True, logger=True)
    #     self.log("test_accuracy_original", accuracy_original, prog_bar=False, logger=True)

    #     # Сбрасываем метрики и очищаем хранилище
    #     self.accuracy.reset()
    #     self.f1.reset()
    #     self.test_step_outputs.clear()

    #     return {"test_accuracy": accuracy, "test_f1": f1}

    # def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
    #     """Шаг предсказания для инференса"""
    #     # Обрабатываем случай с лейблами и без
    #     if isinstance(batch, (list, tuple)) and len(batch) == 2:
    #         images, _ = batch
    #     else:
    #         images = batch

    #     outputs = self(images)
    #     predicted = torch.softmax(outputs, dim=1)
    #     _, predicted = torch.max(predicted, 1)

    #     return predicted


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    # def load_checkpoint(self, checkpoint_path: str):
    #     """Загрузка чекпоинта как в оригинале"""
    #     if checkpoint_path:
    #         checkpoint = torch.load(checkpoint_path)
    #         self.load_state_dict(checkpoint['state_dict'])
    #         self.start_epoch = checkpoint['epoch']
    #         self.checkpoint = checkpoint_path
    #         print(f"Loaded checkpoint from epoch {self.start_epoch}")
