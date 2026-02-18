import fire
import pandas as pd
import pytorch_lightning as pl
from pathlib import Path

from sign_language_classifier.modules.data import SignLanguageMNISTDataModule
from sign_language_classifier.modules.trainer import SignLanguageCNNModule


def main(
    test_csv_path: str,
    checkpoint_path: str,
    batch_size: int = 64,
    num_workers: int = 0,
    img_size: int = 224
) -> None:
    """
    Run inference on test dataset using a trained checkpoint.
    
    Args:
        test_csv_path: Path to test CSV file
        checkpoint_path: Path to model checkpoint (.ckpt)
        batch_size: Batch size for inference
        num_workers: Number of workers for dataloader
        img_size: Image size for resizing
    """
    # Create test datamodule
    dm = SignLanguageMNISTDataModule(
        train_csv_path=None,  # not needed for test
        val_csv_path=None,    # not needed for test
        test_csv_path=test_csv_path,
        img_size=img_size,
        train_batch_size=batch_size,  # will be used as predict_batch_size
        predict_batch_size=batch_size,
        num_workers=num_workers
    )
    
    # Load model from checkpoint
    module = SignLanguageCNNModule.load_from_checkpoint(checkpoint_path)
    module.eval()
    
    # Setup trainer
    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        log_every_n_steps=1,
    )
    
    # Run test
    test_results = trainer.test(module, datamodule=dm)
    
    print("\n" + "="*50)
    print(f"Test Results:")
    print(f"  Accuracy: {test_results[0]['test_accuracy']:.4f}")
    print(f"  F1 Score: {test_results[0]['test_f1']:.4f}")
    print("="*50)


if __name__ == "__main__":
    fire.Fire(main)
