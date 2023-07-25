import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import transforms
from mimeta import MIMeta
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from model import ResNet18Classifier
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for training model.")
    parser.add_argument(
        "--experiment_name",
        type=str,
        help="Base name for experiment.",
        required=True,
    )
    parser.add_argument(
        "--use_data_augmentation",
        action="store_true",
        help="Training uses data augmentation if enabled.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        help="Strength of weight-decay term; 0 means no weight decay.",
        default=0.0,
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        help="Learning rate for ADAM optimizer.",
        default=1e-4,
    )

    args = parser.parse_args()

    transform_list = []
    if args.use_data_augmentation:
        random_crop_and_resize = transforms.Compose(
            [
                transforms.RandomCrop(200),
                transforms.Resize(224),
            ]
        )

        transform_list = [
            transforms.RandomHorizontalFlip(p=0.25),
            transforms.RandomVerticalFlip(p=0.25),
            transforms.RandomRotation(
                10, interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.RandomApply([random_crop_and_resize], p=0.25),
        ]
    transform_list += [
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    # Define the data loaders
    transform = transforms.Compose(transform_list)

    dataset = MIMeta(
        "data",
        "Mammography (Masses)",
        "pathology",
        original_split="train",
        transform=transform,
    )

    # Compute the lengths of the splits
    train_length = int(0.8 * len(dataset))
    valid_length = len(dataset) - train_length

    # Perform the random split
    train_dataset, valid_dataset = torch.utils.data.random_split(
        dataset, [train_length, valid_length]
    )

    print(f"Training dataset length: {len(train_dataset)}")
    print(f"Validation dataset length: {len(valid_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=True)

    experiment_name = args.experiment_name
    if args.use_data_augmentation:
        experiment_name += "-with-aug"
    experiment_name += f"-LR{str(args.learning_rate)}"
    experiment_name += f"-WD{str(args.weight_decay)}"

    logger = TensorBoardLogger(
        save_dir="./runs", name=experiment_name, default_hp_metric=False
    )

    checkpoint_callbacks = [
        ModelCheckpoint(
            monitor="valid/auc_epoch",
            filename="best-auc-{epoch}-{step}",
            mode="max",
        ),
        ModelCheckpoint(
            monitor="valid/loss_epoch",
            filename="best-loss-{epoch}-{step}",
            mode="min",
        ),
    ]

    # Create the model and trainer
    num_classes = dataset.task_target.value
    model = ResNet18Classifier(
        num_classes=num_classes,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    trainer = pl.Trainer(
        max_epochs=1000,
        log_every_n_steps=10,
        val_check_interval=0.25,
        logger=logger,
        callbacks=checkpoint_callbacks,
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
