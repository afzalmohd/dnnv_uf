import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import onnxruntime as ort

# 1) Configuration
MODEL_PATH = "/home/u1411251/tools/vnncomp_benchmarks/cifar10/cifar2020/onnx/cifar10_2_255.onnx"  # path to your ONNX model
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
    # transforms.Normalize((0.4914,0.4822,0.4465),
    #                      (0.2470,0.2435,0.2616))
])
testset = torchvision.datasets.CIFAR10(
    root="../datasets/data_cifar10", train=False, download=True, transform=transform
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
)

# 4) Run inference and collect low-confidence cases
low_conf_images = []
low_conf_scores = []
low_conf_preds = []
softmax = torch.nn.Softmax(dim=1)

for images, _ in testloader:
    # images: [BATCH_SIZE, 3, 32, 32]
    for img in images:
        # make it a batch of 1
        inp = img.unsqueeze(0).numpy()
        out = session.run(None, {input_name: inp})[0]  # shape [1,10]
        logits = torch.from_numpy(out)
        probs = softmax(logits)
        score, pred = torch.max(probs, dim=1)  # each is shape [1]
        if score.item() < THRESHOLD:
            low_conf_images.append(img)
            low_conf_scores.append(score.item())
            low_conf_preds.append(pred.item())

# 5) If any low-conf images were found, stack and display first few
if not low_conf_images:
    print(f"No images with confidence below {THRESHOLD}")
    exit()

# Stack images into a single tensor: shape [N, 3, 32, 32]
low_conf_images = torch.stack(low_conf_images, dim=0)
N = low_conf_images.size(0)
print(f"Found {N} low-confidence images (confidence < {THRESHOLD})")

# Pick up to NUM_TO_DISPLAY images
to_show = low_conf_images[:NUM_TO_DISPLAY]
scores_to_show = low_conf_scores[:NUM_TO_DISPLAY]
preds_to_show = low_conf_preds[:NUM_TO_DISPLAY]

# Build a grid: nrow=4 → 4×4 grid for 16 images
grid = torchvision.utils.make_grid(to_show, nrow=4, padding=2)

# Helper to show the grid
def imshow(img_grid: torch.Tensor, scores, preds):
    # img_grid: [3, H, W]
    img = img_grid.permute(1, 2, 0).cpu().numpy()  # -> [H, W, 3]
    img = np.clip(img, 0, 1)

    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.axis('off')

    # Annotate each cell with score & pred
    for idx, (s, p) in enumerate(zip(scores, preds)):
        row, col = divmod(idx, 4)
        x = col * (32 + 2) + 2
        y = row * (32 + 2) + 10
        plt.text(x, y, f"{p} ({s:.2f})",
                 color='white', fontsize=8,
                 bbox=dict(facecolor='black', alpha=0.6, pad=1))
    plt.show()

# Finally display\ n
imshow(grid, scores_to_show, preds_to_show)
