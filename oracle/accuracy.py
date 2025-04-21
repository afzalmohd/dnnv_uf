import os
import numpy as np
import math
import onnxruntime as ort
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import torch
import random

transform = transforms.Compose([
    transforms.ToTensor(), 
    # transforms.Normalize((0.1307,), (0.3081,)),
    transforms.Lambda(lambda x: x.view(-1))
])

is_training_dataset = True
batch_size = 1
dataset = MNIST(root='data', train=is_training_dataset, download=True, transform=transform)
dataset_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

def print_as_grid(indeces):
    # images, labels = next(iter(dataset_loader))
    images = []
    for idx in indeces:
        im, lb = dataset[idx]
        images.append(im)
    images = torch.stack(images)
    print(images.shape)
    images = images.view(len(indeces), 1, 28, 28)
    print(images.shape)
    image_grid = make_grid(images, nrow=25, padding=2)
    npimg = image_grid.numpy()

    plt.figure(figsize=(30, 30))
    plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap='gray')
    plt.axis('off')
    plt.title('MNIST Grid')


    plt.show()


# def print_as_grid():
#     images, labels = next(iter(test_loader))
#     images = images.view(batch_size, 1, 28, 28)
#     # Set grid size (e.g., 5 rows x 10 columns for batch_size=50)
#     nrow = 10
#     ncol = math.ceil(batch_size/nrow)
#     fig, axes = plt.subplots(nrow, ncol, figsize=(ncol*2, nrow*2))
#     axes = axes.flatten()

#     # 4. Plot each image with its label
#     for idx, ax in enumerate(axes):
#         if idx >= batch_size:
#             break
#         img = images[idx].squeeze(0)    # from [1,28,28] to [28,28]
#         ax.imshow(img, cmap='gray')
#         ax.set_title(f"Label: {labels[idx].item()}")
#         ax.axis('off')

#     plt.tight_layout()
#     plt.show()


def softmax(x):
    e_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
    return e_x / e_x.sum(axis=-1, keepdims=True)

def get_accuracy(net_path):
    session = ort.InferenceSession(net_path)
    input_name = session.get_inputs()[0].name
    total = 0
    correct_label = 0
    max_conf = -math.inf
    min_conf = math.inf

    counter = 0
    indeces = []
    index = 0
    high_conf = 0
    for image, label in dataset_loader:
        image_np = image.numpy().reshape(1,784,1).astype(np.float32)
        output = session.run(None, {input_name: image_np})
        pred = np.argmax(output[0][0])
        softmax_output= softmax(output[0][0])
        pred_conf = softmax_output[pred]
        pred_conf1 = round(pred_conf * 100, 1)
        print(pred, pred_conf1)
        if pred == label.item():
            correct_label += 1
            if pred_conf1 <= 20.0:
                counter += 1
                indeces.append(index) 
            elif pred_conf1 >= 24.0 and high_conf <= 500:
                high_conf += 1
                counter += 1
                indeces.append(index)
        
        total += 1

        if max_conf < pred_conf1:
            max_conf = pred_conf1
        
        if min_conf > pred_conf1:
            min_conf = pred_conf1

        index += 1

    accuracy = correct_label / total
    print(f"Accuracy: {accuracy * 100: .2f}%")
    print(f"Max conf: {max_conf}, Min conf: {min_conf}")
    print(f"Number of selected images: {counter}")

    return indeces





net_dir = '/home/u1411251/tools/vnncomp_benchmarks/mnist_fc/onnx'
net_name = 'mnist-net_256x2.onnx'

net_path = os.path.join(net_dir, net_name)
indeces =  get_accuracy(net_path)
indeces = random.sample(indeces, 1000)
# print_as_grid(indeces=indeces[:1000])
# indeces = random.sample(range(0,60000), 1000)
indeces.sort()

with open("indices.txt", "w") as f:
    f.write(",".join(map(str, indeces[:1000])))