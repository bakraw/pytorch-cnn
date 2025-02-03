# CNN avec PyTorch

Implémentation basique d'un réseau neuronal convolutif avec PyTorch et entraîné sur [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html).
3 couches convolutives, 2 couches linéaires, 1 couche de pooling.

Réalisé dans le cadre d'un projet universitaire.

## Usage

Vérifier que les *torch* et *torchvision* sont installés, puis :
```bash
git clone https://github.com/bakraw/pytorch-cnn
cd pytorch-cnn
python3 main.py
```

Alternativement, copier-coller le script dans un notebook Jupyter (Colab, Kaggle, etc.).

## Résultats

On a 80~85% de précision en moyenne. Il s'agit d'un réseau neuronal convolutif très basique, avec peu de couches, qui serait objectivement plus adapté à des datasets plus simples (Fashion-MNIST, Kuzushiji-MNIST, etc.) et qui rame donc un peu sur CIFAR10. On obtiendrait de bien meilleurs résultats avec des modèles plus avancés, tels que ResNet.

Malgré tout, le modèle se trompe généralement sur les images les plus piégeuses (avions, camions, etc.), et reste assez performant sur les images plus standards.

![Image 1](https://github.com/user-attachments/assets/157b061d-9acb-4aab-ae42-8d502f31153a)
![Image 2](https://github.com/user-attachments/assets/34d060ff-8c30-4384-8090-8b765201ee0c)
![Image 3](https://github.com/user-attachments/assets/07477780-79a0-447f-bb62-beb382d1c4c3)