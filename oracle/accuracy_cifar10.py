import os
import numpy as np
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import onnxruntime as ort
import random
from generate_benchmarks.simulate_network import get_cifar10_train_data

# 1) Configuration
MODEL_PATH = "/home/u1411251/tools/vnncomp_benchmarks/cifar10/oval21/nets/cifar_deep_kw.onnx"  # path to your ONNX model
BATCH_SIZE = 64                  # inference batch size
THRESHOLD = 0.30                 # confidence threshold
NUM_TO_DISPLAY = 16              # how many low-conf images to show

# 2) Prepare the ONNX runtime session
assert os.path.isfile(MODEL_PATH), f"Cannot find ONNX model at {MODEL_PATH}"
session = ort.InferenceSession(MODEL_PATH)
input_name = session.get_inputs()[0].name

# 3) Load CIFAR-10 test set
transform = transforms.Compose([
    transforms.ToTensor(),  # convert to [0,1]
    # add Normalize if your model was trained with it:
    transforms.Normalize((0.4914,0.4822,0.4465),
                         (0.2023,0.1994,0.2010))
])
testset = torchvision.datasets.CIFAR10(
    root="../datasets/data_cifar10", train=True, download=True, transform=transform
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
)

# 4) Run inference and collect low-confidence cases
low_conf_images = []
low_conf_scores = []
low_conf_preds = []
softmax = torch.nn.Softmax(dim=1)

correct = 0
total = 0
high_thr = 80.0
high_conf_images = []
high_conf_limit = 400
index = 0
for images, labels in testloader:
    for img, label in zip(images, labels):
        inp = img.unsqueeze(0).numpy()
        print(inp.shape)
        out = session.run(None, {input_name: inp})[0]  # shape [1,10]
        logits = torch.from_numpy(out)
        probs = softmax(logits)
        score, pred = torch.max(probs, dim=1)  # Get predicted class index
        correct += (pred.item() == label.item())
        total += 1
        if pred.item() == label.item():
            if score.item() < THRESHOLD:
                low_conf_images.append(index)
                low_conf_scores.append(score.item())
                low_conf_preds.append(pred.item())
            elif score.item() >= high_thr and len(high_conf_images) < high_conf_limit:
                high_conf_images.append(index)
        
        index += 1


accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")


print(f"Selected low conf: {len(low_conf_images)}")
selected_idxs = low_conf_images+high_conf_images

random.shuffle(selected_idxs)

selected_idxs = selected_idxs[:1000]
# count = 0
# images, labels = get_cifar10_train_data()
# images = np.transpose(images, (0, 3, 1, 2))
# softmax = torch.nn.Softmax(dim=1)
# for idx in selected_idxs:
#     im = images[idx]
#     lb = labels[idx]
#     mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
#     std = np.array([0.2023, 0.1994, 0.2010], dtype=np.float32)
#     mean = mean.reshape(1,-1,1,1)
#     std = std.reshape(1,-1,1,1)
#     inp = np.expand_dims(im, axis=0)
#     inp = (inp-mean)/std
#     out = session.run(None, {input_name: inp})[0]  # shape [1,10]
#     logits = torch.from_numpy(out)
#     probs = softmax(logits)
#     _, pred = torch.max(probs, dim=1)  # Get predicted class index
#     print(pred.item(), int(lb))
#     if  (pred.item() != int(lb)):
#         print(f"something wrong")
#         count += 1

# print(f"wrong classification: {count}")

print(selected_idxs)

selected_idxs.sort()

with open("indices_cifar10.txt", "w") as f:
    f.write(",".join(map(str, selected_idxs[:1000])))

# # Pick up to NUM_TO_DISPLAY images
# to_show = low_conf_images[:NUM_TO_DISPLAY]
# scores_to_show = low_conf_scores[:NUM_TO_DISPLAY]
# preds_to_show = low_conf_preds[:NUM_TO_DISPLAY]

# # Build a grid: nrow=4 → 4×4 grid for 16 images
# grid = torchvision.utils.make_grid(to_show, nrow=4, padding=2)

# # Helper to show the grid
# def imshow(img_grid: torch.Tensor, scores, preds):
#     # img_grid: [3, H, W]
#     img = img_grid.permute(1, 2, 0).cpu().numpy()  # -> [H, W, 3]
#     img = np.clip(img, 0, 1)

#     plt.figure(figsize=(6, 6))
#     plt.imshow(img)
#     plt.axis('off')

#     # Annotate each cell with score & pred
#     for idx, (s, p) in enumerate(zip(scores, preds)):
#         row, col = divmod(idx, 4)
#         x = col * (32 + 2) + 2
#         y = row * (32 + 2) + 10
#         plt.text(x, y, f"{p} ({s:.2f})",
#                  color='white', fontsize=8,
#                  bbox=dict(facecolor='black', alpha=0.6, pad=1))
#     plt.show()

# Finally display\ n
# imshow(grid, scores_to_show, preds_to_show)
