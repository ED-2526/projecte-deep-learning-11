"""
la base per carregar i servir les imatges del dataset a PyTorch durant l'entrenament.
Bàsicament, diu a PyTorch on són les imatges, quina etiqueta té cadascuna i
com manipular-les abans de donar-les al model.
"""


from torch.utils.data import Dataset
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None): #rep les imatges
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths) #retorna nombre total d'imatges

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label = self.labels[index]

        # Load and preprocess the image --> s'integra amb DataLoaders
        image = self.load_image(image_path) #obre la imatge del disc i la converteix en un tensor
        image = self.preprocess_image(image) #aplica redimensionament, normalització, data augmentation, etc.

        return image, label

    def load_image(self, image_path):
        with Image.open(image_path) as image:
            return image.convert("RGB")

    def preprocess_image(self, image):
        if self.transform is not None:
            return self.transform(image)

        return image
