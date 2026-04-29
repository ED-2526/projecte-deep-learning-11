import os
import hashlib
from collections import Counter

from PIL import Image
from sklearn.model_selection import train_test_split


VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def is_image_file(filename):
    """
    Comprova si un fitxer té extensió d'imatge vàlida.
    """
    return filename.lower().endswith(VALID_EXTENSIONS)


def check_image_is_valid(image_path):
    """
    Intenta obrir una imatge per comprovar que no està corrupta.
    No carrega la imatge completa per entrenar, només valida el fitxer.
    """
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception:
        return False


def compute_file_hash(file_path):
    """
    Calcula un hash del fitxer per detectar duplicats exactes.
    Això ens ajuda a evitar fuga de dades entre train i test.
    """
    hasher = hashlib.md5()

    with open(file_path, "rb") as f:
        hasher.update(f.read())

    return hasher.hexdigest()


def load_wikiart_dataset(root_dir, remove_duplicates=True, check_corrupted=True):
    """
    Llegeix un dataset organitzat en carpetes per classe.

    Estructura esperada:
        root_dir/
            Impressionism/
                img1.jpg
                img2.jpg
            Cubism/
                img3.jpg
            Realism/
                img4.jpg

    Retorna:
        image_paths: llista de paths d'imatges
        labels: llista d'etiquetes numèriques
        class_to_idx: diccionari classe -> índex
        idx_to_class: diccionari índex -> classe
        stats: informació de neteja
    """

    image_paths = []
    labels = []
    class_to_idx = {}

    stats = {
        "total_files_seen": 0,
        "valid_images": 0,
        "skipped_non_images": 0,
        "skipped_corrupted": 0,
        "skipped_duplicates": 0,
    }

    seen_hashes = set()

    class_names = sorted([
        class_name
        for class_name in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, class_name))
    ])

    for class_idx, class_name in enumerate(class_names):
        class_to_idx[class_name] = class_idx
        class_dir = os.path.join(root_dir, class_name)

        for filename in sorted(os.listdir(class_dir)):
            stats["total_files_seen"] += 1

            if not is_image_file(filename):
                stats["skipped_non_images"] += 1
                continue

            image_path = os.path.join(class_dir, filename)

            if check_corrupted and not check_image_is_valid(image_path):
                stats["skipped_corrupted"] += 1
                continue

            if remove_duplicates:
                file_hash = compute_file_hash(image_path)

                if file_hash in seen_hashes:
                    stats["skipped_duplicates"] += 1
                    continue

                seen_hashes.add(file_hash)

            image_paths.append(image_path)
            labels.append(class_idx)
            stats["valid_images"] += 1

    idx_to_class = {idx: class_name for class_name, idx in class_to_idx.items()}

    return image_paths, labels, class_to_idx, idx_to_class, stats


def get_class_distribution(labels, idx_to_class):
    """
    Calcula quantes imatges hi ha per classe.
    Retorna una llista ordenada de major a menor nombre d'imatges.
    """
    counts = Counter(labels)

    distribution = []
    for class_idx, count in counts.items():
        class_name = idx_to_class[class_idx]
        distribution.append((class_name, count))

    distribution = sorted(distribution, key=lambda x: x[1], reverse=True)

    return distribution


def print_dataset_summary(image_paths, labels, class_to_idx, idx_to_class, stats):
    """
    Imprimeix un resum senzill del dataset.
    Ens serveix per veure ràpidament mida, classes i desbalanceig.
    """
    distribution = get_class_distribution(labels, idx_to_class)

    print("\n========== DATASET SUMMARY ==========")
    print(f"Total files seen:        {stats['total_files_seen']}")
    print(f"Valid images:            {stats['valid_images']}")
    print(f"Skipped non-images:      {stats['skipped_non_images']}")
    print(f"Skipped corrupted:       {stats['skipped_corrupted']}")
    print(f"Skipped duplicates:      {stats['skipped_duplicates']}")
    print(f"Number of classes:       {len(class_to_idx)}")
    print(f"Number of image paths:   {len(image_paths)}")
    print(f"Number of labels:        {len(labels)}")

    print("\nTop 10 largest classes:")
    for class_name, count in distribution[:10]:
        print(f"  {class_name}: {count}")

    print("\nTop 10 smallest classes:")
    for class_name, count in distribution[-10:]:
        print(f"  {class_name}: {count}")

    print("=====================================\n")


def split_dataset(image_paths, labels, val_size=0.15, test_size=0.15, random_state=42):
    """
    Divideix el dataset en train, validation i test.

    Important:
    Fem split estratificat perquè el dataset està desbalancejat.
    Això intenta mantenir proporcions semblants de classes a cada partició.
    """

    # Primer separem test
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        image_paths,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    # Ara separem validation dins del que queda
    val_relative_size = val_size / (1.0 - test_size)

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths,
        train_val_labels,
        test_size=val_relative_size,
        random_state=random_state,
        stratify=train_val_labels,
    )

    return train_paths, val_paths, test_paths, train_labels, val_labels, test_labels


def print_split_summary(train_labels, val_labels, test_labels):
    """
    Imprimeix quantes imatges hi ha a train, validation i test.
    """
    print("\n========== SPLIT SUMMARY ==========")
    print(f"Train images: {len(train_labels)}")
    print(f"Val images:   {len(val_labels)}")
    print(f"Test images:  {len(test_labels)}")
    print("===================================\n")