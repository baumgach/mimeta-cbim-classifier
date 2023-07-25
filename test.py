from mimeta import MIMeta
from model import ResNet18Classifier
from torchvision import transforms
import os
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import argparse


def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script for evaluating trained models."
    )
    parser.add_argument(
        "--checkpoints_dir",
        type=dir_path,
        help="Path to directory containing the checkpoint files. The most recent checkpoint will be loaded.",
        required=True,
    )
    parser.add_argument(
        "--checkpoint_identifier",
        type=str,
        help="Filter by checkpoint identifier such aus 'auc' or 'loss'",
        default=None,
    )

    args = parser.parse_args()

    checkpoint_dir = args.checkpoints_dir
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".ckpt")]
    if args.checkpoint_identifier is not None:
        checkpoints = [f for f in checkpoints if args.checkpoint_identifier in f]

    # Sort the checkpoints by epoch number
    checkpoints.sort(key=lambda x: int(x.split("=")[1].split("-")[0]))

    # Get the most recent checkpoint
    latest_checkpoint = os.path.join(checkpoint_dir, checkpoints[-1])

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    test_dataset = MIMeta(
        "data",
        "Mammography (Masses)",
        "pathology",
        original_split="test",
        transform=transform,
    )

    test_loader = DataLoader(test_dataset, batch_size=32)

    model = ResNet18Classifier.load_from_checkpoint(
        num_classes=2,
        checkpoint_path=latest_checkpoint,
    )

    trainer = pl.Trainer()

    # test the model
    trainer.test(model, dataloaders=test_loader)
