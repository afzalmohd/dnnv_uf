import sys
import os
import random
import numpy as np
cwd = os.getcwd()
sys.path.append(os.path.join(cwd, '..'))
print(sys.path)
from inference import load_mnist_data, preprocess_images



def create_input_bounds_tf(img, ep, mean = np.array([0.0]), std = np.array([1.0]), dataset = 'MNIST'):
    img = np.array(img)
    mean = mean.reshape(1,-1,1,1)
    std = std.reshape(1,-1,1,1)
    if dataset == 'MNIST':
        img = img.reshape(2,1,28,28)
    elif dataset == 'CIFAR10':
        img = img.reshape(2,3,32,32)
    lb = np.clip(img-ep, 0, 1)
    lb = ((lb-mean)/std).reshape(-1)
    ub = np.clip(img+ep, 0, 1)
    ub = ((ub-mean)/std).reshape(-1)
    return list(lb), list(ub)

def write_vnnlib(lb, ub, label1=0, label2=0, spec_path='prp.vnnlib', dataset='MNIST', total_output_class=10, eta=-1e-4):
    with open(spec_path, "w") as f:
        # if dataset == 'MNIST':
        #     f.write(f"; Mnist property with labels: {label1}, {label2}.\n")
        # else:
        #     f.write(f"; Cifar10 property with label: {label1}, {label2}.\n")

        if dataset == 'MNIST':
            f.write(f"; Mnist global robustness property.\n")
        else:
            f.write(f"; Cifar10 global robustness property.\n")

        # Declare input variables.
        f.write("\n")
        for i in range(0, len(lb)):
            f.write(f"(declare-const X_{i} Real)\n")
        f.write("\n")

        # Declare output variables.
        f.write("\n")
        for i in range(total_output_class-1):
            f.write(f"(declare-const Y_{i} Real)\n")
        f.write("\n")

        # Define input constraints.
        f.write(f"; Input constraints:\n")
        for i in range(0,len(lb)):
            f.write(f"(assert (<= X_{i} {ub[i]}))\n")
            f.write(f"(assert (>= X_{i} {lb[i]}))\n")
            f.write("\n")
        f.write("\n")

        f.write(f"; Output constraints:\n")
        f.write("(assert (or\n")
        # prop_output = int(total_output_class/2)
        # for i in range(total_output_class):
        #     if i != label1 and i < prop_output:
        #         f.write(f"    (and (>= Y_{i} Y_{label1}))\n")
        #     elif i != label2 and i >= prop_output:
        #         f.write(f"    (and (>= Y_{i} Y_{label2}))\n")

        for i in range(total_output_class-1):
            f.write(f"    (and (>= Y_{i} {eta}))\n")

        f.write("))")


def generate_prop():
    # mnist = load_mnist_data()
    input_dim = 2

    # Select two random images
    # idx1, idx2 = random.sample(range(len(mnist)), 2)
    # idx1 = 6515
    # idx2=6146
    # img1, label1 = mnist[idx1]
    # img2, label2 = mnist[idx2]

    # print(f"Using digits: {idx1}:{label1} and {idx2}:{label2}")

    # # Preprocess images
    # input_data = preprocess_images(img1, img2)
    # print(input_data.shape)

    # lb, ub = create_input_bounds_tf(input_data, ep=0.03)
    lb = [0.0]*2*input_dim
    ub = [1.0]*2*input_dim
    write_vnnlib(lb, ub, eta=0.0, total_output_class=2) # eta < 1e-4

if __name__ == '__main__':
    generate_prop()

