"""
la base per carregar i servir les imatges del dataset a PyTorch durant l'entrenament.
Bàsicament, diu a PyTorch on són les imatges, quina etiqueta té cadascuna i
com manipular-les abans de donar-les al model.
"""


import torch
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    def __init__(self, image_paths, labels): #rep les imatges
        self.image_paths = image_paths
        self.labels = labels

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

        
        # Implement your image loading logic here
        # For example, you can use PIL or OpenCV to load the image
        # and return it as a tensor
        image = torch.tensor(...)  # Replace ... with your image loading code
        raise NotImplementedError

    def preprocess_image(self, image):

        #AQUI: redimensionar, normalitzar amb mitjanes o sd, data augmentation... 
        # Implement your image preprocessing logic here
        # For example, you can apply transformations such as resizing,
        # normalization, or data augmentation
        preprocessed_image = image  # Replace with your preprocessing code
        raise NotImplementedError