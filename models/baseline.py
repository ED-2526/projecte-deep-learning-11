""""
Model senzill de CNN. Haurem d'implementar després ResNet, EfficientNet...
"""


import torch

import torch.nn as nn

class Baseline(nn.Module): #hereda nn.Module, classe base per xarxes
    def __init__(self, num_classes): #num_classes seran el nombre d'estils a classificar
        super(Baseline, self).__init__()
        
        #primera capa convolucional q extreu patrons visuals (16 filtres de 3x3)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU() #afegeix no-linealitat
        #redueix mida a la meitat (més robust i eficient)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        #fa una mitjana de tot el mapa de característiques, reduint qualsevol mida espacial a 1×1.
        #obtenim vector de 32 característiques
        self.maxpool2 = nn.AdaptiveAvgPool2d(1)
        
        
        #ara aplanem el tensor i expandeix els 32 valors a 128 neurones
        self.fc1 = nn.Linear(32, 128)
        self.relu3 = nn.ReLU()
        #capa final que projecta les 128 neurones a les prediccions
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x): #flux complet
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        
        x = x.view(x.size(0), -1)
        
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        
        return x
    
    
    
if __name__ == "__main__":
    
    # Create an instance of the Baseline model
    model = Baseline(num_classes=10)
    
    # Create a random input tensor
    input_tensor = torch.randn(1, 3, 32, 32) #HAUREM DE CANVIAR MIDA D'ENTRADA A 224 X 224 PER EXEMPLE
    
    # Forward pass through the model
    output = model(input_tensor)
    
    # Print the output
    print(output.shape)