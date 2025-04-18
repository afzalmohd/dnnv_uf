import os
import numpy as np
import shutil
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from setup import clean_directory, get_output_dims, set_images_labels, select_images_with_labels
from generate_properties import create_input_bounds_tf, save_vnnlib_tf_standard

# transform = transforms.Compose([
#     transforms.ToTensor(), 
#     # transforms.Normalize((0.1307,), (0.3081,)),
#     transforms.Lambda(lambda x: x.view(-1))
# ])

# is_training_dataset = True
# batch_size = 1
# dataset = MNIST(root='../data', train=is_training_dataset, download=True, transform=transform)
# dataset_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

def gen_props_standard(spec_dir, selected_images, selected_labels, selected_idxs, eps):
    counter = 0
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

def setup_on_standard(netnames, im_idxs_file, dataset, timeout, epsilons, target_benchmarks_dir, vnncomp_benchmarks_dir):
    print(dataset, timeout, epsilons, target_benchmarks_dir, vnncomp_benchmarks_dir)
    clean_directory(target_benchmarks_dir)
    os.makedirs(target_benchmarks_dir, exist_ok=True)
    # net_dir = os.path.join(target_benchmarks_dir, 'nets')
    # prop_dir = os.path.join(target_benchmarks_dir, 'props')
    instances_file = os.path.join(target_benchmarks_dir, 'instances.csv')
    if os.path.isfile(instances_file):
        os.remove(instances_file)
    # create_empty_dirs(net_dir, prop_dir)

    net_path = os.path.join(target_benchmarks_dir, 'onnx')
    prp_path = os.path.join(target_benchmarks_dir, 'vnnlib')
    os.makedirs(net_path, exist_ok=True)
    os.makedirs(prp_path, exist_ok=True)

    for net in netnames:
        orig_net_path = os.path.join(vnncomp_benchmarks_dir, 'onnx', net)
        target_net_path = os.path.join(net_path, net)
        shutil.copy(orig_net_path, target_net_path)

    images, labels, indexes = select_images_with_labels(dataset_idxs_file=im_idxs_file, max_num_indexs=1000)
    labels = labels.reshape(-1)
    
    gen_props_standard(prp_path, images, labels, indexes, epsilons)

    instance_lines = []
    for net in netnames:
        for idx in indexes:
            for ep in epsilons:
                net_path1 = os.path.join(net_path, net)
                prp_name = f"prop_{idx}_{ep}.vnnlib"
                prp_path1 = os.path.join(prp_path, prp_name)
                ins_line = f"{net_path1},{prp_path1},{timeout}\n"
                instance_lines.append(ins_line)

    with open(instances_file, 'w') as f:
        f.writelines(instance_lines)


if __name__ == '__main__':
    is_test_dataset = False
    netnames = ['mnist-net_256x2.onnx']
    im_idxs_file = '/home/u1411251/tools/my_scripts/oracle/indices.txt'
    dataset = 'MNIST'
    timeout = 120
    epsilons = [0.06]
    target_benchmarks_dir = '/home/u1411251/tools/fp'
    vnncomp_benchmarks_dir = '/home/u1411251/tools/vnncomp_benchmarks/mnist_fc'
    set_images_labels(dataset=dataset, is_test_data=is_test_dataset)

    setup_on_standard(netnames, im_idxs_file, dataset, timeout, epsilons, target_benchmarks_dir, vnncomp_benchmarks_dir)