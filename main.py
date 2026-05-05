import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from utils.data_utils import (
    prepare_resized_cache_dataset,
    load_wikiart_dataset,
    print_dataset_summary,
    split_dataset,
    print_split_summary,
    create_dataloaders,
)

from models.transfer_model import create_resnet_model
from train import train_model
from test import test_model


def set_seed(seed):
    """
    Fixem llavors per fer l'experiment més reproduïble.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_checkpoint(model, checkpoint_path, device):
    """
    Carrega els pesos del millor model guardat.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"Checkpoint carregat: {checkpoint_path}")
    print(f"Epoch checkpoint: {checkpoint['epoch'] + 1}")
    print(f"Validation accuracy checkpoint: {checkpoint['val_accuracy']:.4f}")

    return model


def main():
    config = {
        "experiment_name": "exp1_resnet18_all_classes_no_aug_no_class_weights",
        "dataset_root": "/home/datasets/wikiart/",  # CANVIA AIXÒ SI CAL
        "model_name": "resnet18",
        "feature_extraction": True,
        "epochs": 10,
        "batch_size": 128,
        "learning_rate": 1e-3,
        "image_size": 224,
        "val_size": 0.15,
        "test_size": 0.15,
        "random_seed": 42,
        "num_workers": 8,
        "remove_duplicates": False,
        "check_corrupted": False,
        "use_resized_cache": True,
        "resized_cache_root": "/tmp/wikiart_224",
        "force_rebuild_cache": False,
        "cache_num_workers": 12,
        "use_class_weights": False,
        "use_augmentation": False,
    }

    set_seed(config["random_seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device utilitzat: {device}")

    checkpoint_path = os.path.join(
        "results",
        "checkpoints",
        f"{config['experiment_name']}_best.pth",
    )

    wandb.init(
        project="wikiart-classification",
        name=config["experiment_name"],
        config=config,
    )

    dataset_root = config["dataset_root"]
    resize_images_in_dataloader = True

    if config["use_resized_cache"]:
        dataset_root = prepare_resized_cache_dataset(
            source_root=config["dataset_root"],
            cache_root=config["resized_cache_root"],
            image_size=config["image_size"],
            force_rebuild=config["force_rebuild_cache"],
            num_workers=config["cache_num_workers"],
        )
        resize_images_in_dataloader = False

    # 1. Carregar dataset complet
    image_paths, labels, class_to_idx, idx_to_class, stats = load_wikiart_dataset(
        root_dir=dataset_root,
        remove_duplicates=config["remove_duplicates"],
        check_corrupted=config["check_corrupted"],
    )

    print_dataset_summary(
        image_paths=image_paths,
        labels=labels,
        class_to_idx=class_to_idx,
        idx_to_class=idx_to_class,
        stats=stats,
    )

    num_classes = len(class_to_idx)
    print(f"Nombre de classes utilitzades: {num_classes}")

    wandb.config.update({
        "num_classes": num_classes,
        "num_images": len(image_paths),
    })

    # 2. Split train / validation / test
    train_paths, val_paths, test_paths, train_labels, val_labels, test_labels = split_dataset(
        image_paths=image_paths,
        labels=labels,
        val_size=config["val_size"],
        test_size=config["test_size"],
        random_state=config["random_seed"],
    )

    print_split_summary(train_labels, val_labels, test_labels)

    # 3. Crear DataLoaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_paths=train_paths,
        val_paths=val_paths,
        test_paths=test_paths,
        train_labels=train_labels,
        val_labels=val_labels,
        test_labels=test_labels,
        batch_size=config["batch_size"],
        image_size=config["image_size"],
        num_workers=config["num_workers"],
        resize_images=resize_images_in_dataloader,
    )

    # 4. Crear model ResNet
    model = create_resnet_model(
        num_classes=num_classes,
        model_name=config["model_name"],
        feature_extraction=config["feature_extraction"],
    )

    model = model.to(device)

    # 5. Loss sense pesos de classe
    criterion = nn.CrossEntropyLoss()

    # Només entrenem paràmetres amb requires_grad=True.
    # En feature extraction això serà principalment la capa fc final.
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())

    optimizer = optim.Adam(
        trainable_params,
        lr=config["learning_rate"],
    )

    # 6. Entrenament amb checkpoint
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        config=config,
        device=device,
        checkpoint_path=checkpoint_path,
        class_to_idx=class_to_idx,
        idx_to_class=idx_to_class,
    )

    # 7. Carregar millor checkpoint abans del test
    model = load_checkpoint(
        model=model,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    # 8. Test final
    test_model(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        device=device,
        idx_to_class=idx_to_class,
        save_dir="results/figures",
    )
    wandb.finish()


if __name__ == "__main__":
    main()
