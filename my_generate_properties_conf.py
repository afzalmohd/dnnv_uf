
import os
import sys
from typing import List
from simulate_network import read_images_from_dataset
from simulate_network import get_selected_images


# import torch
# import torchvision.datasets as dset
# import torchvision.transforms as trans
# from torch.utils.data import DataLoader
# from torch.utils.data import sampler
import numpy as np
# import onnxruntime as onnxrun

    


def gen_props_specific(spec_dir):
    eps = [0.04, 0.06]
    net_dirs = '/home/afzal/tools/networks/conf_final/eran_mod' 
    nets = ['mnist_relu_3_50.onnx', 'mnist_relu_3_100.onnx', 'mnist_relu_5_100.onnx', 'mnist_relu_6_100.onnx', 'mnist_relu_6_200.onnx', 'mnist_relu_9_100.onnx', 'mnist_relu_9_200.onnx']
    nets += ['mnist_relu_4_1024.onnx', 'ffnnRELU__Point_6_500.onnx', 'ffnnRELU__PGDK_w_0.1_6_500.onnx', 'ffnnRELU__PGDK_w_0.3_6_500.onnx']

    nets = ['mnist_relu_5_100.onnx']
    
    counter = 0
    for net in nets:
        # selected_images, selected_labels, selected_idxs = read_images_from_dataset(os.path.join(net_dirs, net))
        selected_images, selected_labels, selected_idxs = get_selected_images()
        for i in range(len(selected_images)):
            image = selected_images[i]
            image = image.reshape(784)
            label = selected_labels[i]
            idx = selected_idxs[i]
            for ep in eps:
                lb,ub = create_input_bounds_tf(image, ep)
                spec_path = f"prop_{idx}_{ep}.vnnlib"
                spec_path = os.path.join(spec_dir, spec_path)
                save_vnnlib_tf_1(lb, ub, label, spec_path, dataset)
                counter += 1

    print(f"Total number of props: {counter}")


    
def gen_props_standard(spec_dir):
    eps = [0.04, 0.06]

    counter = 0
    selected_images, selected_labels, selected_idxs = get_selected_images()
    for i in range(len(selected_images)):
        image = selected_images[i]
        image = image.reshape(784)
        label = selected_labels[i]
        idx = selected_idxs[i]
        for ep in eps:
            lb,ub = create_input_bounds_tf(image, ep)
            spec_path = f"prop_{idx}_{ep}.vnnlib"
            spec_path = os.path.join(spec_dir, spec_path)
            save_vnnlib_tf_standard(lb, ub, label, spec_path, dataset)
            counter += 1

    print(f"Total number of props: {counter}")

def read_images_from_file(filepath):
    labels = []
    images = []
    f = open(filepath, 'r')
    Lines = f.readlines()
    for line in Lines:
        line = line.strip()
        line = list(eval(line))
        label = line[0]
        im = line[1:]
        labels.append(label)
        images.append(im)

    return labels, images

def normalize_image(im, dataset):
    if dataset == 'MNIST':
        im = np.array(im)/255
        im = im.astype(np.float32)
        im = list(im)
    elif dataset == 'CIFAR10':
        im = np.array(im)/255
        # im = im.astype(np.float32)
        im = list(im)
        means = [0.4914, 0.4822, 0.4465]
        stds = [0.2023, 0.1994, 0.2010]
        # stds = [0.2470, 0.2435, 0.2616]
        count = 0
        temp = 3072*[0]
        for i in range(0,1024):
            temp[count] = (im[count] - means[0])/stds[0]
            count += 1
            temp[count] = (im[count] - means[1])/stds[1]
            count += 1
            temp[count] = (im[count] - means[2])/stds[2]
            count += 1

        count = 0
        for i in range(0,1024):
            im[i] = temp[count]
            count += 1
            im[i+1024] = temp[count]
            count += 1
            im[i+2048] = temp[count]
            count += 1
    
        im = np.array(im)
        im = im.astype(np.float32)
        im = list(im)
    else:
        sys.stderr.write("Dataset must be either MNIST or CIFAR10\n")

    return im




def load_data_tf(net_path: str, images, labels, dataset):
    selected_images, selected_lables, selected_indexes, wrong_classified = [], [], [], []
    index = 0
    for(lb, image) in zip(labels, images):
        image = normalize_image(image, dataset)
        selected_indexes.append(index)
        selected_images.append(image)
        selected_lables.append(lb)
        index += 1

    return selected_images, selected_lables, selected_indexes, wrong_classified


def create_input_bounds_tf(img, ep):
    img = np.array(img)
    lb = np.clip(img-ep, 0, 1)
    ub = np.clip(img+ep, 0, 1)
    return list(lb), list(ub)


def save_vnnlib_tf_standard(lb, ub, label: int, spec_path: str, dataset, total_output_class: int = 10):
     with open(spec_path, "w") as f:
        if dataset == 'MNIST':
            f.write(f"; Mnist property with label: {label}.\n")
        else:
            f.write(f"; Cifar10 property with label: {label}.\n")

        # Declare input variables.
        f.write("\n")
        for i in range(0, len(lb)):
            f.write(f"(declare-const X_{i} Real)\n")
        f.write("\n")

        # Declare output variables.
        f.write("\n")
        for i in range(total_output_class):
            f.write(f"(declare-const Y_{i} Real)\n")
        f.write("\n")

        # Define input constraints.
        f.write(f"; Input constraints:\n")
        for i in range(0,len(lb)):
            f.write(f"(assert (<= X_{i} {ub[i]}))\n")
            f.write(f"(assert (>= X_{i} {lb[i]}))\n")
            f.write("\n")
        f.write("\n")

        # Define output constraints.
        f.write(f"; Output constraints:\n")
        f.write("(assert (or\n")
        for i in range(total_output_class):
            if i != label:
                f.write(f"    (and (>= Y_{i} Y_{label}))\n")

        # for i in range(9):
        #     and_str = "     (and "
        #     for j in range(9):
        #         and_str = f"{and_str} (>= Y_{9*i+j} 0.0)"
        #     and_str = f"{and_str})\n"
        #     f.write(and_str)

        f.write("))")
    
def save_vnnlib_tf(lb, ub, label: int, spec_path: str, dataset, total_output_class: int = 81):
     with open(spec_path, "w") as f:
        if dataset == 'MNIST':
            f.write(f"; Mnist property with label: {label}.\n")
        else:
            f.write(f"; Cifar10 property with label: {label}.\n")

        # Declare input variables.
        f.write("\n")
        for i in range(0, len(lb)):
            f.write(f"(declare-const X_{i} Real)\n")
        f.write("\n")

        # Declare output variables.
        f.write("\n")
        for i in range(total_output_class):
            f.write(f"(declare-const Y_{i} Real)\n")
        f.write("\n")

        # Define input constraints.
        f.write(f"; Input constraints:\n")
        for i in range(0,len(lb)):
            f.write(f"(assert (<= X_{i} {ub[i]}))\n")
            f.write(f"(assert (>= X_{i} {lb[i]}))\n")
            f.write("\n")
        f.write("\n")

        # Define output constraints.
        f.write(f"; Output constraints:\n")
        f.write("(assert (or\n")
        # for i in range(total_output_class):
        #     f.write(f"    (and (>= Y_{i} Y_{label}))\n")

        for i in range(9):
            and_str = "     (and "
            for j in range(9):
                and_str = f"{and_str} (>= Y_{9*i+j} 0.0)"
            and_str = f"{and_str})\n"
            f.write(and_str)

        f.write("))")


def save_vnnlib_tf_1(lb, ub, label: int, spec_path: str, dataset, total_output_class: int = 9):
     tolerance_param = -1e-3
     with open(spec_path, "w") as f:
        if dataset == 'MNIST':
            f.write(f"; Mnist property with label: {label}.\n")
        else:
            f.write(f"; Cifar10 property with label: {label}.\n")

        # Declare input variables.
        f.write("\n")
        for i in range(0, len(lb)):
            f.write(f"(declare-const X_{i} Real)\n")
        f.write("\n")

        # Declare output variables.
        f.write("\n")
        for i in range(total_output_class):
            f.write(f"(declare-const Y_{i} Real)\n")
        f.write("\n")

        # Define input constraints.
        f.write(f"; Input constraints:\n")
        for i in range(0,len(lb)):
            f.write(f"(assert (<= X_{i} {ub[i]}))\n")
            f.write(f"(assert (>= X_{i} {lb[i]}))\n")
            f.write("\n")
        f.write("\n")

        # Define output constraints.
        f.write(f"; Output constraints:\n")
        f.write("(assert (or\n")
        for i in range(total_output_class):
            f.write(f"    (and (>= Y_{i} {tolerance_param}))\n")

        

        f.write("))")




def create_instances_csv(nets: List, eps: List, wrong_classified, num_props: int = 15, path: str = "mnistfc_instances.csv"):

    """
    Creates the instances_csv file.

    Args:
        num_props:
            The number of properties.
        path:
            The path of the csv file.
    """
    props = []
    for net in nets:
        file_name = os.path.splitext(net)[0]
        file_name = file_name.replace("_","-")
        for ep in eps:
            for i in range(num_props):
                if i not in wrong_classified:
                    prop = f"prop_{i}_{ep}.vnnlib"
                    props.append(prop)

    with open(path, "w") as f:

        for net in nets:
            timeout = 2000 # if net == "mnist-net_256x2.onnx" else 300
            for prop in props:
                if net == nets[-1] and prop == props[-1]:
                    f.write(f"{net},{prop},{timeout}")
                else:
                    f.write(f"{net},{prop},{timeout}\n")




    







if __name__ == '__main__':
    epsilons = [0.04, 0.06]
    dataset = 'MNIST'
    num_images = 2
    net_format = 'tf' #onnx/tf
    num_props = len(epsilons)*num_images
    # get_selected_images()
    if len(sys.argv) == 5:
        net_path_onnx = str(sys.argv[1])
        net_path_tf = str(sys.argv[2])
        dataset_path = str(sys.argv[3])
        spec_dir = str(sys.argv[4])
    else:
        print("Please provide the network,dataset path and spec dir")
        sys.exit(0)

    gen_props_specific(spec_dir)
    exit(0)
    
    labels, images = read_images_from_file(dataset_path)
    selected_images, selected_labels, selected_indexes_tf, wrong_classified_tf = load_data_tf(net_path_tf, images, labels, dataset)
    wrong_classified = wrong_classified_tf


    print(f"Number of props: {len(selected_images)}")

    # exit(0)
    net_name = os.path.basename(net_path_onnx)
    # file_name = os.path.splitext(net_name)[0]
    # file_name = file_name.replace("_","-")
    # spec_dir = os.path.join(spec_dir, file_name)
    if not os.path.exists(spec_dir):
        os.makedirs(spec_dir)
  
    for eps in epsilons:
        for ind, image, label in zip(selected_indexes_tf, selected_images, selected_labels):
            lb,ub = create_input_bounds_tf(image, eps)
            spec_path = f"prop_{ind}_{eps}.vnnlib"
            spec_path = os.path.join(spec_dir, spec_path)
            save_vnnlib_tf_1(lb, ub, label, spec_path, dataset)


    # ins_path = os.path.join(spec_dir, "mnist_instances.csv")
    # create_instances_csv([net_name], epsilons, wrong_classified, num_props=num_images, path=ins_path)

    print(f"Properties written to: {spec_dir}")





# noinspection PyShadowingNames
# def load_data1(netpath: str, dataset_path: str, num_imgs: int=100) -> tuple:

#     """
#     Loads the mnist data.

#     Args:
#         data_dir:
#             The directory to store the full MNIST dataset.
#         num_imgs:
#             The number of images to extract from the test-set
#         random:
#             If true, random image indices are used, otherwise the first images
#             are used.
#     Returns:
#         A tuple of tensors (images, labels).
#     """
#     selected_images, selected_labels, indexes, wrong_classified = [], [], [], []
#     file = open(dataset_path, 'r')
#     csvreader = csv.reader(file)
#     header = next(csvreader)
#     label = int(header[0])
#     im = np.array(header[1:]).reshape(1, 1,28, 28)
#     im = im.astype(np.float32)/255

#     im1 = np.array(header[1:]).reshape(1, 784, 1)
#     im1 = im1.astype(np.float32)/255
#     im1 = torch.from_numpy(im1)
#     if(is_correctly_classified_onnx(label, im, netpath, index=0)):
#         selected_images.append(im1)
#         selected_labels.append(label)
#         indexes.append(0)
#     else:
#         print(f"Image: {label}, index: 0, not classified correctly")
#         wrong_classified.append(0)

#     count = 1
#     for row in csvreader:
#         if count >= num_imgs:
#             break
#         label = int(row[0])
#         im = np.array(row[1:]).reshape(1, 1, 28,28)
#         im = im.astype(np.float32)/255

#         im1 = np.array(row[1:]).reshape(1, 784, 1)
#         im1 = im1.astype(np.float32)/255
#         im1 = torch.from_numpy(im1)
#         if(is_correctly_classified_onnx(label, im, netpath, index=count)):
#             selected_images.append(im1)
#             selected_labels.append(label)
#             indexes.append(count)
#         else:
#             wrong_classified.append(count)
#         count += 1
#     return selected_images, selected_labels, indexes, wrong_classified




    # exit(0)
    # data_dir = "./tmp.txt"
    # random = False
    # trns_norm = trans.ToTensor()
    # mnist_test = dset.MNIST(data_dir, train=False, download=True, transform=trns_norm)

    # if random:
    #     loader_test = DataLoader(mnist_test, batch_size=10000,
    #                              sampler=sampler.SubsetRandomSampler(range(10000)))
    # else:
    #     loader_test = DataLoader(mnist_test, batch_size=10000)

    # x, labels = next(iter(loader_test))

    # num_selected = 0
    # selected_images, selected_labels = [], []
    # num_imgs=1
    # i = 0
    # while i < num_imgs:
    #     i += 1
    #     im = images[i].numpy().reshape(1, 784, 1)
    #     print("Labels: ")
    #     print(labels[i])
    #     print("Image: ")
    #     print(im)
    #     selected_images.append(im)
    #     selected_labels.append(labels[i])


    #exit(0)
    # sess1 = onnxrun.InferenceSession("./mnist-net_256x2.onnx")
    # sess2 = onnxrun.InferenceSession("./mnist-net_256x4.onnx")
    # sess3 = onnxrun.InferenceSession("./mnist-net_256x6.onnx")
    # sessions = [sess1, sess2, sess3]

    # i = -1
    # while num_selected < num_imgs:

    #     i += 1
    #     correctly_classified = True

    #     for sess in sessions:
    #         input_name = sess.get_inputs()[0].name
    #         result = np.argmax(sess.run(None, {input_name: images[i].numpy().reshape(1, 784, 1)})[0])

    #         if result != labels[i]:
    #             correctly_classified = False
    #             break

    #     if not correctly_classified:
    #         continue

    #     num_selected += 1
    #     selected_images.append(images[i])
    #     selected_labels.append(labels[i])

    # return selected_images, selected_labels