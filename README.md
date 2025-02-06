# CNN avec PyTorch

Implémentation basique d'un réseau neuronal convolutif avec PyTorch et entraîné sur [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html).
3 couches convolutives, 2 couches linéaires, 1 couche de pooling.

Réalisé dans le cadre d'un projet universitaire.

## Usage

Vérifier que *torch* et *torchvision* sont installés, puis :
```bash
git clone https://github.com/bakraw/pytorch-cnn
cd pytorch-cnn
python3 cnn.py
```

Alternativement, copier-coller le script dans un notebook Jupyter (Colab, Kaggle, etc.).

Le script utilise automatiquement CUDA si disponible, mais il est aussi possible de le faire tourner sur CPU (non recommandé, passez par Colab si vous n'avez pas de [GPU Nvidia](https://youtu.be/XDpDesU_0zo?si=aCmgeyb0Auf7egqH) à disposition).

## Résultats

Jusqu'à 94% de précision avec 25 itérations. Il s'agit d'un réseau neuronal convolutif très basique, avec peu de couches, qui serait objectivement plus adapté à des datasets plus simples (Fashion-MNIST, Kuzushiji-MNIST, etc.) et qui rame donc un peu sur CIFAR10. On obtiendrait de bien meilleurs résultats avec des modèles plus avancés, tels que ResNet.

Malgré tout, le modèle se trompe généralement sur les images les plus piégeuses (avions, camions, etc.), et reste assez performant sur les images plus standards.

![Image](https://github.com/user-attachments/assets/a89be846-149d-481a-a98e-1ddf6624440c)
![Image](https://github.com/user-attachments/assets/68442d8e-85f1-42b7-a918-752cd53d3da8)
![Image](https://github.com/user-attachments/assets/22610fdd-c365-42ec-90a2-a651d4682f77)
![Image](https://github.com/user-attachments/assets/3f64fecb-a6b2-4e17-987f-a4f62453d96e)