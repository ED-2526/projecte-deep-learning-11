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

    running_loss = torch.zeros((), device=device, dtype=torch.float64)
    running_correct = torch.zeros((), device=device, dtype=torch.int64)
    running_total = 0

    all_preds = []
    all_labels = []

    with torch.inference_mode():
        for images, labels in tqdm(test_loader, desc="Testing", mininterval=1.0):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.double() * images.size(0)

            preds = outputs.argmax(dim=1)

            running_correct += (preds == labels).sum()
            running_total += labels.size(0)

            all_preds.append(preds)
            all_labels.append(labels)

    test_loss = (running_loss / running_total).item()
    test_acc = (running_correct.double() / running_total).item()
    all_preds = torch.cat(all_preds).cpu().tolist()
    all_labels = torch.cat(all_labels).cpu().tolist()
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
