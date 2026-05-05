import torch
import wandb
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay


def test_model(model, test_loader, criterion, device, idx_to_class=None, save_dir="results/figures"):
    """
    Avalua el model sobre test.

    Important:
    Test només s'hauria d'utilitzar al final, amb el millor checkpoint carregat.
    """

    model.eval()

    running_loss = 0.0
    running_correct = 0
    running_total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            _, preds = torch.max(outputs, dim=1)

            running_correct += (preds == labels).sum().item()
            running_total += labels.size(0)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    test_loss = running_loss / running_total
    test_acc = running_correct / running_total
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted")

    print("\n========== TEST RESULTS ==========")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Acc:  {test_acc:.4f}")
    print(f"Macro F1:  {macro_f1:.4f}")
    print(f"Weighted F1:  {weighted_f1:.4f}")
    print("==================================\n")

    wandb.log({
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    })
    #confusion matrix
    os.makedirs(save_dir, exist_ok=True)

    cm = confusion_matrix(all_labels, all_preds)

    if idx_to_class is not None:
        class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    else:
        class_names = None

    fig, ax = plt.subplots(figsize=(14, 14))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names,
    )

    disp.plot(
        ax=ax,
        xticks_rotation=90,
        colorbar=True,
    )

    plt.title("Confusion Matrix - Test Set")
    plt.tight_layout()

    confusion_matrix_path = os.path.join(save_dir, "confusion_matrix_test.png")
    plt.savefig(confusion_matrix_path, dpi=300)
    plt.close(fig)

    print(f"Confusion matrix guardada a: {confusion_matrix_path}")

    wandb.log({
        "confusion_matrix": wandb.Image(confusion_matrix_path)
    })

    return test_loss, test_acc, macro_f1, weighted_f1, all_preds, all_labels
