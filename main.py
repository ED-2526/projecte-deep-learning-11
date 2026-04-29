import os
import random
import wandb

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from train import *
from test import *
from utils.utils import *
from tqdm.auto import tqdm
from utils.data_utils import (
    load_wikiart_dataset,
    print_dataset_summary,
    split_dataset,
    print_split_summary,
)

# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
random.seed(hash("setting random seeds") % 2**32 - 1)
np.random.seed(hash("improves reproducibility") % 2**32 - 1)
torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# remove slow mirror from list of MNIST mirrors
torchvision.datasets.MNIST.mirrors = [mirror for mirror in torchvision.datasets.MNIST.mirrors
                                      if not mirror.startswith("http://yann.lecun.com")]




def model_pipeline(cfg:dict) -> None:
    # tell wandb to get started
    with wandb.init(project="pytorch-demo", config=cfg):
      # access all HPs through wandb.config, so logging matches execution!
      config = wandb.config

      # make the model, data, and optimization problem
      model, train_loader, test_loader, criterion, optimizer = make(config,device=device)

      # and use them to train the model
      train(model, train_loader, criterion, optimizer, config,device=device)

      # and test its final performance
      test(model, test_loader,device=device)

    return model

if __name__ == "__main__":
    root_dir = "/home/edxnG11/dataset_wikiart/raw/"

    image_paths, labels, class_to_idx, idx_to_class, stats = load_wikiart_dataset(
        root_dir=root_dir,
        remove_duplicates=True,
        check_corrupted=True,
    )

    print_dataset_summary(
        image_paths=image_paths,
        labels=labels,
        class_to_idx=class_to_idx,
        idx_to_class=idx_to_class,
        stats=stats,
    )

    train_paths, val_paths, test_paths, train_labels, val_labels, test_labels = split_dataset(
        image_paths=image_paths,
        labels=labels,
        val_size=0.15,
        test_size=0.15,
        random_state=42,
    )

    print_split_summary(train_labels, val_labels, test_labels)

    print("Example image path:", train_paths[0])
    print("Example label:", train_labels[0])
    print("Example class name:", idx_to_class[train_labels[0]])
