############################## IMPORTS ##############################


import torch
import torchvision


############################## MODÈLE ##############################


# CNN basique, avec 3 convolutions et 2 fully connected layers.
class Cnn(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Max pooling 2x2, qu'on appliquera après chaque convolution.
        # Concrètement, la taille des images sera divisée par 2, ce qui permettra de mieux résister aux légères différences.
        self.pool = torch.nn.MaxPool2d(2, 2)

        # Convolutions
        # On se retrouvera avec une feature map de 4x4x64
        self.conv1 = torch.nn.Conv2d(3, 32, 3) # 3 inputs (RGB), 32 outputs, kernels 3x3
        self.conv2 = torch.nn.Conv2d(32, 64, 3) # 32 inputs, 64 outputs, kernels 3x3
        self.conv3 = torch.nn.Conv2d(64, 64, 3) # 64 inputs, 64 outputs, kernels 3x3

        # Fully connected layers
        # On passera de notre feature map à nos 10 classes
        self.fc1 = torch.nn.Linear(64 * 4 * 4, 64) # 1024 inputs, 64 outputs
        self.fc2 = torch.nn.Linear(64, 10) # 64 inputs, 10 outputs

    def forward(self, x):
        # Matrice initiale: 32x32x3
        x = torch.nn.functional.relu(self.conv1(x)) # -> 30x30x32 (ReLU passe les valeurs négatives à 0)
        x = self.pool(x) # -> 15x15x32
        x = torch.nn.functional.relu(self.conv2(x)) # -> 13x13x64
        x = self.pool(x) # -> 6x6x64
        x = torch.nn.functional.relu(self.conv3(x)) # -> 4x4x64
        x = torch.flatten(x, 1) # -> 1024x1 (nécessaire pour les FCL qui attendent un vecteur en entrée)
        x = torch.nn.functional.relu(self.fc1(x)) # -> 64x1
        x = self.fc2(x) # -> 10x1
        return x


################################### ENTRAÎNEMENT ##############################


# On utilise CUDA si disponible, le CPU sinon.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("\033[92m✓ CUDA actif.\033[0m" if device.type == 'cuda' else "\033[91m⚠ CUDA indisponible. Exécution sur le CPU.\033[0m")

# Hyperparamètres
EPOCHS = 10 # Nombre d'itérations pour l'entraînement.
            # Plus grand -> plus précis.
            # Plus petit -> plus rapide, moins de risque d'overfitting.

LEARNING_RATE = 0.001 # Vitesse d'apprentissage ("taille des pas").
                      # Plus grand -> plus rapide, sort des minima locaux plus facilement.
                      # Plus petit -> plus précis.

BATCH_SIZE = 32 # Nombres d'images traitées à chaque itération.
                # Plus grand -> plus rapide, calcul de gradient plus "stable".
                # Plus petit -> moins de risque de blocage dans un minimum local, moins de mémoire utilisée.

# Transformations à appliquer aux images.
# On a (x - mean) / std tel que pour x = 0, on a (0 - 0,5) / 0,5 = -1 et pour x = 1 (initialement 255), on a (1 - 0,5) / 0,5 = 1.
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),  # Transforme les images en tenseurs Pytorch (et transforme les valeurs de 0 à 255 en valeurs entre 0 et 1)
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Transforme les valeurs de 0 à 1 en valeurs entre -1 et 1
    ])

# Import de CIFAR10 (60000 images de 32x32, 10 classes)
training_dataset = torchvision.datasets.CIFAR10('./data', True, transform, None, True)
testing_dataset = torchvision.datasets.CIFAR10('./data', True, transform, None, True)

# Création des DataLoaders
training_loader = torch.utils.data.DataLoader(training_dataset, BATCH_SIZE, True)
testing_loader = torch.utils.data.DataLoader(testing_dataset, BATCH_SIZE, True)

# Définition des classes
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Instantiation du modèle
model = Cnn().to(device)

# Fonction de perte, qui calcule l'écart entre la sortie du modèle et la sortie attendue.
# La cross-entropy punit plus sévèrement les erreurs qui ont un haut taux de certitude.
loss_function = torch.nn.CrossEntropyLoss() 

# Optimiseur, qui va modifier les paramètres du modèle pour minimiser la fonction de perte.
# Le Adam peut ajuster la taille des pas en mémorisant les modifs précédentes, en et affectant plus les paramètres qui changent rarement.
# Il est donc plus efficace que le SGD, qui se contente d'avancer vers le minimum global selon une taille de pas fixe.
optimizer = torch.optim.Adam(model.parameters(), LEARNING_RATE)

# Entraînement
print("\033[93m🛈 Début de l'entraînement...\033[0m")
total_steps = len(training_loader) # On s'en servira pour calculer la perte moyenne par batch.
for epoch in range(EPOCHS): 
    total_loss = 0.0

    # On parcourt les batches
    for i, (images, labels) in enumerate(training_loader):
        # Passage sur le GPU si disponible
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images) # On passe les images au modèle pour obtenir les prédictions
        loss = loss_function(outputs, labels) # On calcule la perte entre les prédictions et les labels réels

        # Backward pass
        loss.backward() # Backpropagation ; on transmet les gradients calculés par la fonction de perte.
                        # Les gradients représentent la dérivée de la fonction de perte par rapport aux paramètres du modèle.
                        # Chaque paramètre aura un gradient associé, qui indiquera à l'optimiseur comment modifier le paramètre pour réduire la perte.

        # Ajustement des paramètres
        optimizer.step() # On ajuste les paramètres du modèle en fonction des gradients calculés.
        optimizer.zero_grad() # On réinitialise les gradients pour le batch suivant.
        total_loss += loss.item() # On ajoute la perte du batch à la perte totale.

    print(f'Epoch: {epoch + 1}/{EPOCHS}, Perte: {total_loss / total_steps:.4f}')

print("\033[92m✓ Entraînement terminé.\033[0m")

# Sauvegarde du modèle
print("\033[93m🛈 Sauvegarde du modèle...\033[0m")
torch.save(model.state_dict(), 'model.pth')
print("\033[92m✓ Modèle sauvegardé.\033[0m")



################################### TEST ##############################

print("\033[93m🛈 Début du test...\033[0m")
model.eval() # On met le modèle en mode évaluation.

with torch.no_grad(): # On désactive le calcul des gradients pour accélérer les calculs.
    correct = 0
    samples = len(testing_loader.dataset)

    for images, labels in testing_loader:
        # Passage sur le GPU si disponible
        images = images.to(device)
        labels = labels.to(device)

        # Prédictions
        outputs = model(images) 
        _, predicted = torch.max(outputs.data, 1) # On prend la classe avec la plus grande probabilité.
        correct += (predicted == labels).sum().item()

print("\033[92m✓ Test terminé.\033[0m")
print(f'Précision: {100 * correct / samples:.2f}%')

