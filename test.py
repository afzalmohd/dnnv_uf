import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import math


# Load CIFAR-100 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()

# Combine train and test sets to form the full dataset (optional, if needed)
# x_data = np.concatenate([x_train, x_test])
# y_data = np.concatenate([y_train, y_test])
x_data = x_test
y_data = y_test

# Extract CIFAR-100 (first 100 images)
base_idx = 7100
cifar_100_images = x_data[base_idx:base_idx + 100]
cifar_100_labels = y_data[base_idx:base_idx + 100]
fine_labels = [
    'apple',  # id 0
    'aquarium_fish',
    'baby',
    'bear',
    'beaver',
    'bed',
    'bee',
    'beetle',
    'bicycle',
    'bottle',
    'bowl',
    'boy',
    'bridge',
    'bus',
    'butterfly',
    'camel',
    'can',
    'castle',
    'caterpillar',
    'cattle',
    'chair',
    'chimpanzee',
    'clock',
    'cloud',
    'cockroach',
    'couch',
    'crab',
    'crocodile',
    'cup',
    'dinosaur',
    'dolphin',
    'elephant',
    'flatfish',
    'forest',
    'fox',
    'girl',
    'hamster',
    'house',
    'kangaroo',
    'computer_keyboard',
    'lamp',
    'lawn_mower',
    'leopard',
    'lion',
    'lizard',
    'lobster',
    'man',
    'maple_tree',
    'motorcycle',
    'mountain',
    'mouse',
    'mushroom',
    'oak_tree',
    'orange',
    'orchid',
    'otter',
    'palm_tree',
    'pear',
    'pickup_truck',
    'pine_tree',
    'plain',
    'plate',
    'poppy',
    'porcupine',
    'possum',
    'rabbit',
    'raccoon',
    'ray',
    'road',
    'rocket',
    'rose',
    'sea',
    'seal',
    'shark',
    'shrew',
    'skunk',
    'skyscraper',
    'snail',
    'snake',
    'spider',
    'squirrel',
    'streetcar',
    'sunflower',
    'sweet_pepper',
    'table',
    'tank',
    'telephone',
    'television',
    'tiger',
    'tractor',
    'train',
    'trout',
    'tulip',
    'turtle',
    'wardrobe',
    'whale',
    'willow_tree',
    'wolf',
    'woman',
    'worm',
]
for i,name in enumerate(fine_labels):
    print(i,name,end=',')
def visualize_cifar(images, labels, rows=10, cols=10, save_path="cifar100_grid.png"):
    """Visualize a grid of CIFAR-100 images and save to a file."""
    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
    fig.suptitle("CIFAR-100 Dataset", fontsize=16)
    
    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j 
            axes[i, j].imshow(images[idx])
            axes[i, j].set_title(f"{base_idx+idx, labels[idx][0]}")
            # axes[i, j].set_title(f"{fine_labels[labels[idx][0]]}")
            axes[i, j].axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)  # Adjust top for title
    plt.savefig(save_path)
    print(f"Figure saved to {save_path}")

i = 2680
y = y_test
print()
print(f"Image idx: {i}, Prev-label: {y[i-1][0]}, label: {y[i][0]}, next_label: {y[i+1][0]}")
# Visualize CIFAR-100
def main():
    visualize_cifar(cifar_100_images, cifar_100_labels)

if __name__ == "__main__":
    main()
