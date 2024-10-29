import os
import shutil
from PIL import Image
import csv
import numpy as np
from modify_onnx import append_layers
from generate_properties import gen_props
from generate_instance_file import gen_instances_file
from generate_instance_file import gen_instances_file_top_k
from simulate_network import get_mnist_test_data
from simulate_network import get_mnist_train_data
from simulate_network import run_network_mnist_test
from simulate_network import select_images_top_k
from simulate_network import get_selected_images_gans
from simulate_network import get_cifar10_test_data
from simulate_network import get_cifar10_train_data
from simulate_network import run_network_cifar10
from modify_onnx_top_k import append_layers_top_k



is_test_data = True
mnist_dataset = 'MNIST'
cifar10_dataset = 'CIFAR10'
cifar10_mean = np.array([0.49140000, 0.48219999, 0.44650000], dtype=np.float32)
cifar10_std = np.array([0.20229999, 0.19939999, 0.20100000], dtype=np.float32)

IMAGES_MNIST, LABELS_MNIST = get_mnist_train_data()
if is_test_data:
    IMAGES_MNIST, LABELS_MNIST = get_mnist_test_data()

IMAGES_CIFAR10, LABELS_CIFAR10 = get_cifar10_train_data()
if is_test_data:
    IMAGES_CIFAR10, LABELS_CIFAR10 = get_cifar10_test_data()

print(IMAGES_MNIST.shape)

def get_images_labels_idxs(model_path, conf):
    conf = conf / 100.0
    _, _, low_confs_idx, _, high_conf_idx, _  = run_network_mnist_test(model_path, conf_th=conf)
    return low_confs_idx, high_conf_idx

def create_empty_dirs(net_dir, prop_dir):
    if not os.path.isdir(net_dir):
        os.makedirs(net_dir)
    
    if not os.path.isdir(prop_dir):
        os.makedirs(prop_dir)

def clean_directory(directory_path):
    if os.path.exists(directory_path):
        # Loop over the files and subdirectories in the directory
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            try:
                # If it's a file, remove it
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.remove(file_path)
                # If it's a directory, use shutil to remove it
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')

def get_images_csv_gans():
    image_csv_path = '/home/afzal/tools/my_scripts/gans/images_gan.csv'
    with open(image_csv_path, 'r') as f:
        selected_images, selected_labels, selected_indexes = [], [], []
        csv_readers = csv.reader(f, delimiter=',')
        idx = 0
        for row in csv_readers:
            label = int(row[0])
            image = np.array(row[1:])
            image = image.reshape(1,784,1)
            image = image.astype(np.float32)
            selected_images.append(image)
            selected_labels.append(label)
            selected_indexes.append(idx)
            idx += 1

    return selected_images, selected_labels, selected_indexes

def get_images():
    images_dir = '/home/afzal/tools/my_scripts/gans/images'
    selected_images, selected_labels, selected_indexes = [], [], []
    for im_filename in os.listdir(images_dir):
        image_path = os.path.join(images_dir, im_filename)
        file_name_list = im_filename[:-4].split('_')
        im_idx = int(file_name_list[1])
        im_label = int(file_name_list[2])
        img = Image.open(image_path).convert('L')
        img_arr = np.array(img, dtype=np.float32)
        img_arr = img_arr.reshape(1,784,1)
        selected_images.append(img_arr)
        selected_labels.append(im_label)
        selected_indexes.append(im_idx)

    return selected_images, selected_labels, selected_indexes

def get_eran_images(eran_images_file):
    images, labels = [], []
    with open(eran_images_file, 'r') as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            labels.append(int(row[0]))
            im = np.array(row[1:], dtype=np.float32)/255.0
            images.append(im)

    images = np.array(images)
    images = images.reshape(images.shape[0], 1, 784, 1)
    labels = np.array([labels])
    labels = labels.reshape(images.shape[0],)
    return images, labels


def setup_modified_props_gans():
    orig_net_dir = '/home/afzal/tools/networks/conf_final/eran_mod'
    nets = ['mnist_relu_3_50.onnx', 'mnist_relu_3_100.onnx', 'mnist_relu_5_100.onnx', 'mnist_relu_6_100.onnx']
    nets += ['mnist_relu_6_200.onnx', 'mnist_relu_9_100.onnx', 'mnist_relu_9_200.onnx']
    nets = ['mnist_relu_5_100.onnx']
    confs = [60, 70, 80, 90, 95]
    epsilons = [0.04]
    setup_dir = '/home/afzal/tools/networks/mod_props'
    clean_directory(setup_dir)
    net_dir = os.path.join(setup_dir, 'nets')
    prop_dir = os.path.join(setup_dir, 'props')
    instances_file = os.path.join(setup_dir, 'instances.csv')
    if os.path.isfile(instances_file):
        os.remove(instances_file)
    create_empty_dirs(net_dir, prop_dir)

    g_images, g_labels, g_indexes = get_images_csv_gans()
    g_images = np.array(g_images)
    g_labels = np.array(g_labels)
    for net in nets:
        for conf in confs:
            high_confs_idx, low_confs_idx = get_selected_images_gans(os.path.join(orig_net_dir, net), g_images, g_indexes, conf)
            print(f"net: {net},conf:{conf},low conf images: {len(low_confs_idx)}")
            selected_images = g_images[low_confs_idx]
            selected_labels = g_labels[low_confs_idx]
            append_layers([net], orig_net_dir, net_dir, selected_images, selected_labels, low_confs_idx, is_softmax=True, confs=[conf], is_high_conf=False)
            gen_props(prop_dir, selected_images, selected_labels, low_confs_idx, epsilons) 
            gen_instances_file(net_dir, [net], prop_dir, low_confs_idx, [conf], epsilons, instances_file)

            print(f"net: {net},conf:{conf},high conf images: {len(high_confs_idx)}")
            # print(high_conf_idx)
            selected_images = g_images[high_confs_idx]
            selected_labels = g_labels[high_confs_idx]
            append_layers([net], orig_net_dir, net_dir, selected_images, selected_labels, high_confs_idx, is_softmax=True, confs=[conf], is_high_conf=True)
            gen_props(prop_dir, selected_images, selected_labels, high_confs_idx, epsilons)
            gen_instances_file(net_dir, [net], prop_dir, high_confs_idx, [conf], epsilons, instances_file)            

        print(f"Number of images for standard prop: {len(g_indexes)}")
        append_layers([net], orig_net_dir, net_dir, g_images, g_labels, g_indexes, is_softmax=True, confs=[0], is_high_conf=False)
        prop_dir_normal = os.path.join(prop_dir, 'standard')
        if not os.path.isdir(prop_dir_normal):
            os.makedirs(prop_dir_normal)
        gen_props(prop_dir_normal, g_images, g_labels, g_indexes, epsilons, is_standard_prop=True) 
        gen_instances_file(net_dir, [net], prop_dir_normal, g_indexes, [0], epsilons, instances_file)
        


def setup_modified_props_old():
    orig_net_dir = '/home/afzal/tools/networks/conf_final/eran_mod'
    nets = ['mnist_relu_3_50.onnx', 'mnist_relu_3_100.onnx', 'mnist_relu_5_100.onnx', 'mnist_relu_6_100.onnx']
    nets += ['mnist_relu_6_200.onnx', 'mnist_relu_9_100.onnx', 'mnist_relu_9_200.onnx']
    # nets = ['mnist_relu_3_50.onnx', 'mnist_relu_3_100.onnx', 'mnist_relu_6_100.onnx', 'mnist_relu_9_100.onnx', 'mnist_relu_9_200.onnx']
    confs = [0, 60, 70, 80, 90, 95]
    epsilons = [0.04]
    max_low_conf_images = 100
    max_high_conf_images = 20
    setup_dir = '/home/afzal/tools/networks/mod_props'
    clean_directory(setup_dir)
    net_dir = os.path.join(setup_dir, 'nets')
    prop_dir = os.path.join(setup_dir, 'props')
    instances_file = os.path.join(setup_dir, 'instances.csv')
    if os.path.isfile(instances_file):
        os.remove(instances_file)
    create_empty_dirs(net_dir, prop_dir)

    for net in nets:
        for conf in confs:
            if conf != 0:
                _, _, low_confs_idx, _, high_conf_idx, _  = run_network_mnist_test(os.path.join(orig_net_dir, net), conf_th=conf)
                low_confs_idx = low_confs_idx[:max_low_conf_images]
                print(f"net: {net},conf:{conf},low conf images: {len(low_confs_idx)}")
                print(low_confs_idx)
                selected_images = IMAGES_MNIST[low_confs_idx]
                selected_labels = LABELS_MNIST[low_confs_idx]
                append_layers([net], orig_net_dir, net_dir, selected_images, selected_labels, low_confs_idx, is_softmax=True, confs=[conf], is_high_conf=False)
                gen_props(prop_dir, selected_images, selected_labels, low_confs_idx, epsilons) 
                gen_instances_file(net_dir, [net], prop_dir, low_confs_idx, [conf], epsilons, instances_file)

                high_conf_idx = high_conf_idx[:max_high_conf_images]
                print(f"net: {net},conf:{conf},high conf images: {len(high_conf_idx)}")
                print(high_conf_idx)
                selected_images = IMAGES_MNIST[high_conf_idx]
                selected_labels = LABELS_MNIST[high_conf_idx]
                append_layers([net], orig_net_dir, net_dir, selected_images, selected_labels, high_conf_idx, is_softmax=True, confs=[conf], is_high_conf=True)
                gen_props(prop_dir, selected_images, selected_labels, high_conf_idx, epsilons)
                gen_instances_file(net_dir, [net], prop_dir, high_conf_idx, [conf], epsilons, instances_file)
            else:
                _, _, low_confs_idx, _, high_conf_idx, _  = run_network_mnist_test(os.path.join(orig_net_dir, net), conf_th=100)
                low_confs_idx = low_confs_idx[:max_low_conf_images]
                print(f"net: {net},conf:{conf},low conf images: {len(low_confs_idx)}")
                print(low_confs_idx)
                selected_images = IMAGES_MNIST[low_confs_idx]
                selected_labels = LABELS_MNIST[low_confs_idx]
                append_layers([net], orig_net_dir, net_dir, selected_images, selected_labels, low_confs_idx, is_softmax=True, confs=[conf], is_high_conf=False)
                prop_dir_normal = os.path.join(prop_dir, 'standard')
                if not os.path.isdir(prop_dir_normal):
                    os.makedirs(prop_dir_normal)
                gen_props(prop_dir_normal, selected_images, selected_labels, low_confs_idx, epsilons, is_standard_prop=True) 
                gen_instances_file(net_dir, [net], prop_dir_normal, low_confs_idx, [conf], epsilons, instances_file)


def setup_modified_props(dataset = mnist_dataset):
    net_root_dir = '/home/afzal/Documents/tools/networks/conf_final'
    is_softmax = True
    is_eran_images = False
    max_num_images = 100
    max_low_conf_images = int(0.9*max_num_images)
    max_high_conf_images = max_num_images - max_low_conf_images
    timeout = 2000
    if is_softmax:
        confs = [60, 80, 90, 95]
        confs = [60]
    else:
        confs = [40, 60, 80]
    

    if dataset == mnist_dataset:
        dataset_idxs_file = os.path.join(net_root_dir, 'mnist', 'selected_idxs_mnist.txt')
        orig_net_dir = os.path.join(net_root_dir, 'mnist', 'vnncomp')
        # orig_net_dir = '/home/afzal/tools/networks/conf_final/mnist/eran_mod'
        nets = ['mnist-net_256x2.onnx', 'mnist-net_256x4.onnx', 'mnist-net_256x6.onnx']
        # nets = ['mnist-net_256x6.onnx']
        mean = np.array([0.0], dtype=np.float32)
        std = np.array([1.0], dtype=np.float32)
        epsilons = [0.06]
        filter_images = run_network_mnist_test
        is_cnn = False
        IMAGES = IMAGES_MNIST
        LABELS = LABELS_MNIST
    else:
        dataset_idxs_file = os.path.join(net_root_dir, 'cifar10', 'selected_idxs_cifar10.txt')
        orig_net_dir = os.path.join(net_root_dir, 'cifar10', 'vnncomp')
        nets = []
        # nets += ['cifar10_2_255_simplified.onnx']
        # nets += ['cifar10_8_255_simplified.onnx']
        # nets += ['convBigRELU__PGD.onnx']
        nets += ['resnet_2b.onnx']
        # nets += ['resnet_4b.onnx']
        mean = cifar10_mean
        std = cifar10_std
        epsilons = [0.01]
        filter_images = run_network_cifar10
        is_cnn = True
        IMAGES = IMAGES_CIFAR10
        LABELS = LABELS_CIFAR10


    setup_dir = os.path.join(net_root_dir, 'benchmarks')
    clean_directory(setup_dir)
    net_dir = os.path.join(setup_dir, 'nets')
    prop_dir = os.path.join(setup_dir, 'props')
    instances_file = os.path.join(setup_dir, 'instances.csv')
    if os.path.isfile(instances_file):
        os.remove(instances_file)
    create_empty_dirs(net_dir, prop_dir)

    for net in nets:
        low_plus_high_conf_images_idxs = []
        for conf in confs:
            _, _, low_confs_idx, _, high_conf_idx, _  = filter_images(os.path.join(orig_net_dir, net), conf_th=conf, is_cnn=is_cnn, mean=mean, std=std)
            low_confs_idx = low_confs_idx[:max_low_conf_images]
            selected_idxs = low_confs_idx
            low_plus_high_conf_images_idxs += selected_idxs
            selected_images = IMAGES[selected_idxs]
            selected_labels = LABELS[selected_idxs]
            print(f"net: {net},conf:{conf},low conf images: {len(selected_idxs)}")
            # print(low_confs_idx)
            
            append_layers([net], orig_net_dir, net_dir, selected_images, selected_labels, selected_idxs, is_softmax=is_softmax, confs=[conf], is_high_conf=False)
            gen_props(prop_dir, selected_images, selected_labels, selected_idxs, epsilons, mean=mean, std=std, dataset=dataset)
            gen_instances_file(net_dir, [net], prop_dir, selected_idxs, [conf], epsilons, instances_file, timeout=timeout)
                
            high_conf_idx = high_conf_idx[:max_high_conf_images]
            print(f"net: {net},conf:{conf},high conf images: {len(high_conf_idx)}")
            low_plus_high_conf_images_idxs += high_conf_idx
            # print(high_conf_idx)
            selected_idxs = high_conf_idx
            selected_images = IMAGES[high_conf_idx]
            selected_labels = LABELS[high_conf_idx]
            append_layers([net], orig_net_dir, net_dir, selected_images, selected_labels, selected_idxs, is_softmax=is_softmax, confs=[conf], is_high_conf=True)
            gen_props(prop_dir, selected_images, selected_labels, selected_idxs, epsilons, mean=mean, std=std, dataset=dataset)
            gen_instances_file(net_dir, [net], prop_dir, selected_idxs, [conf], epsilons, instances_file, timeout=timeout)

        low_plus_high_conf_images_idxs = list(set(low_plus_high_conf_images_idxs))
        print(f"Number of images for standard prop: {len(low_plus_high_conf_images_idxs)}")
        selected_images = IMAGES[low_plus_high_conf_images_idxs]
        selected_labels = LABELS[low_plus_high_conf_images_idxs]
        append_layers([net], orig_net_dir, net_dir, selected_images, selected_labels, low_plus_high_conf_images_idxs, is_softmax=is_softmax, confs=[0], is_high_conf=False)
        prop_dir_normal = os.path.join(prop_dir, 'standard')
        if not os.path.isdir(prop_dir_normal):
            os.makedirs(prop_dir_normal)
        gen_props(prop_dir_normal, selected_images, selected_labels, low_plus_high_conf_images_idxs, epsilons, is_standard_prop=True, mean=mean, std=std, dataset=dataset) 
        gen_instances_file(net_dir, [net], prop_dir_normal, low_plus_high_conf_images_idxs, [0], epsilons, instances_file, timeout=timeout)




def set_up_top_k():
    num_top_k = 2
    orig_net_dir = '/home/afzal/tools/networks/conf_final/eran_mod'
    nets = ['mnist_relu_3_50.onnx', 'mnist_relu_3_100.onnx', 'mnist_relu_5_100.onnx', 'mnist_relu_6_100.onnx']
    nets += ['mnist_relu_6_200.onnx', 'mnist_relu_9_100.onnx', 'mnist_relu_9_200.onnx']
    # nets = ['mnist_relu_3_50.onnx']
    epsilons = [0.04]
    max_images = 50
    conf_th = 40
    setup_dir = '/home/afzal/tools/networks/mod_props'
    clean_directory(setup_dir)
    net_dir = os.path.join(setup_dir, 'nets')
    prop_dir = os.path.join(setup_dir, 'props')
    instances_file = os.path.join(setup_dir, 'instances.csv')
    if os.path.isfile(instances_file):
        os.remove(instances_file)
    create_empty_dirs(net_dir, prop_dir)

    for net in nets:
        net_path = os.path.join(orig_net_dir, net)
        selected_idexs, selected_top_k = select_images_top_k(model_path=net_path, num_top_k=num_top_k, conf_th=conf_th)
        selected_idexs = selected_idexs[:max_images]
        selected_top_k = selected_top_k[:max_images]
        print(f"{net} , {len(selected_top_k)}")
        selected_images = IMAGES_MNIST[selected_idexs]
        append_layers_top_k([net], orig_net_dir, net_dir, selected_images, selected_top_k, selected_idexs, is_top_k_robust_paper=False)
        append_layers_top_k([net], orig_net_dir, net_dir, selected_images, selected_top_k, selected_idexs, is_standard_prop=True, is_top_k_robust_paper=False)

        prop_selected_labels = [l[0] for l in selected_top_k]
        final_out_dims = 10 - num_top_k
        gen_props(prop_dir, selected_images, prop_selected_labels, selected_idexs, epsilons, net_out_dims=final_out_dims)
        gen_instances_file_top_k(net_dir, [net], prop_dir, selected_idexs, epsilons, instances_file)

        prop_dir_normal = os.path.join(prop_dir, 'standard')
        if not os.path.isdir(prop_dir_normal):
            os.makedirs(prop_dir_normal)
        gen_props(prop_dir_normal, selected_images, prop_selected_labels, selected_idexs, epsilons, net_out_dims=final_out_dims, is_standard_prop=True)
        gen_instances_file_top_k(net_dir, [net], prop_dir_normal, selected_idexs, epsilons, instances_file, is_standard_prop=True)

def set_up_top_k_robust_paper():
    num_top_k = 2
    orig_net_dir = '/home/afzal/tools/networks/conf_final/eran_mod'
    nets = ['mnist_relu_3_50.onnx', 'mnist_relu_3_100.onnx', 'mnist_relu_5_100.onnx', 'mnist_relu_6_100.onnx']
    nets += ['mnist_relu_6_200.onnx', 'mnist_relu_9_100.onnx', 'mnist_relu_9_200.onnx']
    # nets = ['mnist_relu_3_50.onnx']
    epsilons = [0.04]
    max_images = 50
    setup_dir = '/home/afzal/tools/networks/mod_props'
    clean_directory(setup_dir)
    net_dir = os.path.join(setup_dir, 'nets')
    prop_dir = os.path.join(setup_dir, 'props')
    instances_file = os.path.join(setup_dir, 'instances.csv')
    if os.path.isfile(instances_file):
        os.remove(instances_file)
    create_empty_dirs(net_dir, prop_dir)

    for net in nets:
        net_path = os.path.join(orig_net_dir, net)
        selected_idexs, selected_top_k = select_images_top_k(model_path=net_path, num_top_k=num_top_k, conf_th=0)
        selected_idexs = selected_idexs[:max_images]
        selected_top_k = selected_top_k[:max_images]
        print(f"{net} , {len(selected_top_k)}")
        selected_images = IMAGES_MNIST[selected_idexs]
        append_layers_top_k([net], orig_net_dir, net_dir, selected_images, selected_top_k, selected_idexs)
        append_layers_top_k([net], orig_net_dir, net_dir, selected_images, selected_top_k, selected_idexs, is_standard_prop=True)

        prop_selected_labels = [l[0] for l in selected_top_k]
        final_out_dims = 1
        gen_props(prop_dir, selected_images, prop_selected_labels, selected_idexs, epsilons, net_out_dims=final_out_dims, tolerance_param=1e-4)
        gen_instances_file_top_k(net_dir, [net], prop_dir, selected_idexs, epsilons, instances_file)

        prop_dir_normal = os.path.join(prop_dir, 'standard')
        if not os.path.isdir(prop_dir_normal):
            os.makedirs(prop_dir_normal)
        gen_props(prop_dir_normal, selected_images, prop_selected_labels, selected_idexs, epsilons, net_out_dims=final_out_dims, is_standard_prop=True)
        gen_instances_file_top_k(net_dir, [net], prop_dir_normal, selected_idexs, epsilons, instances_file, is_standard_prop=True)




def get_aaai_images(num_images = 21):
    dataset_file = '/home/afzal/tools/VeriNN/deep_refine/benchmarks/dataset/mnist/mnist_test.csv'
    images, labels, idxs = [], [], []
    i = 0
    with open(dataset_file, 'r') as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            labels.append(int(row[0]))
            im = np.array(row[1:], dtype=np.float32) / 255
            images.append(im)
            idxs.append(i)
            i += 1

    return images[:num_images], labels[:num_images], idxs[:num_images]



def setup_on_deeppoly_images():
    orig_net_dir = '/home/afzal/tools/networks/conf_final/eran_mod'
    nets = ['mnist_relu_3_50.onnx', 'mnist_relu_3_100.onnx', 'mnist_relu_5_100.onnx', 'mnist_relu_6_100.onnx']
    nets += ['mnist_relu_6_200.onnx', 'mnist_relu_9_100.onnx', 'mnist_relu_9_200.onnx']
    nets = ['ffnnRELU__Point_6_500.onnx', 'ffnnRELU__PGDK_w_0.1_6_500.onnx', 'ffnnRELU__PGDK_w_0.3_6_500.onnx', 'mnist_relu_4_1024.onnx']
    is_softmax = False
    if is_softmax:
        confs = [0, 60, 80, 90, 95]
    else:
        confs = [40, 60, 80]
    epsilons = [0.04, 0.06]
    setup_dir = '/home/afzal/tools/networks/mod_props'
    clean_directory(setup_dir)
    net_dir = os.path.join(setup_dir, 'nets')
    prop_dir = os.path.join(setup_dir, 'props')
    instances_file = os.path.join(setup_dir, 'instances.csv')
    if os.path.isfile(instances_file):
        os.remove(instances_file)
    create_empty_dirs(net_dir, prop_dir)
    selected_images, selected_labels, selected_idxs = get_aaai_images(num_images=100)
    
    gen_props(prop_dir, selected_images, selected_labels, selected_idxs, epsilons, tolerance_param=-1e-5) 
    prop_dir_normal = os.path.join(prop_dir, 'standard')
    if not os.path.isdir(prop_dir_normal):
        os.makedirs(prop_dir_normal)
    gen_props(prop_dir_normal, selected_images, selected_labels, selected_idxs, epsilons, is_standard_prop=True)

    for net in nets:
        for conf in confs:
            append_layers([net], orig_net_dir, net_dir, selected_images, selected_labels, selected_idxs, is_softmax=is_softmax, confs=[conf], is_high_conf=False)
            if conf != 0:
                gen_instances_file(net_dir, [net], prop_dir, selected_idxs, [conf], epsilons, instances_file)    
            else:
                gen_instances_file(net_dir, [net], prop_dir_normal, selected_idxs, [conf], epsilons, instances_file)   

def select_images_with_labels(dataset, dataset_idxs_file, max_num_indexs=50):
    with open(dataset_idxs_file) as f:
        line = f.readline()
        indexes = np.fromstring(line, dtype=int, sep=',')
        indexes = indexes[:max_num_indexs]
        if dataset == mnist_dataset:
            images = IMAGES_MNIST[indexes]
            labels = LABELS_MNIST[indexes] 
        elif dataset == cifar10_dataset:
            images = IMAGES_CIFAR10[indexes]
            labels = LABELS_CIFAR10[indexes]
    
    return images, labels, indexes


def setup_on_orig_dataset_images(dataset=mnist_dataset):
    net_root_dir = '/home/afzal/tools/networks/conf_final'
    is_softmax = True
    max_num_images = 50
    timeout = 3000
    if is_softmax:
        confs = [0, 60, 80, 90, 95]
    else:
        confs = [40, 60, 80]
    

    if dataset == mnist_dataset:
        dataset_idxs_file = os.path.join(net_root_dir, 'mnist', 'selected_idxs_mnist.txt')
        orig_net_dir = os.path.join(net_root_dir, 'mnist', 'vnncomp')
        nets = ['mnist-net_256x2.onnx', 'mnist-net_256x4.onnx', 'mnist-net_256x6.onnx']
        mean = np.array([0.0], dtype=np.float32)
        std = np.array([1.0], dtype=np.float32)
        epsilons = [0.06]
    else:
        dataset_idxs_file = os.path.join(net_root_dir, 'cifar10', 'selected_idxs_cifar10.txt')
        orig_net_dir = os.path.join(net_root_dir, 'cifar10', 'vnncomp')
        nets = []
        # nets += ['cifar10_2_255_simplified.onnx']
        # nets += ['cifar10_8_255_simplified.onnx']
        # nets += ['convBigRELU__PGD.onnx']
        nets += ['resnet_2b.onnx']
        # nets += ['resnet_4b.onnx']
        mean = cifar10_mean
        std = cifar10_std
        epsilons = [0.01]


    setup_dir = os.path.join(net_root_dir, 'benchmarks')
    clean_directory(setup_dir)
    net_dir = os.path.join(setup_dir, 'nets')
    prop_dir = os.path.join(setup_dir, 'props')
    instances_file = os.path.join(setup_dir, 'instances.csv')
    if os.path.isfile(instances_file):
        os.remove(instances_file)
    create_empty_dirs(net_dir, prop_dir)
    selected_images, selected_labels, selected_idxs = select_images_with_labels(dataset, dataset_idxs_file, max_num_indexs=max_num_images)
    selected_labels = selected_labels.reshape(-1)
    gen_props(prop_dir, selected_images, selected_labels, selected_idxs, epsilons, tolerance_param=-1e-5, dataset=dataset, mean=mean, std=std) 
    prop_dir_normal = os.path.join(prop_dir, 'standard')
    if not os.path.isdir(prop_dir_normal):
        os.makedirs(prop_dir_normal)
    gen_props(prop_dir_normal, selected_images, selected_labels, selected_idxs, epsilons, is_standard_prop=True, mean=mean, std=std, dataset=dataset)

    for net in nets:
        for conf in confs:
            append_layers([net], orig_net_dir, net_dir, selected_images, selected_labels, selected_idxs, is_softmax=is_softmax, confs=[conf], is_high_conf=False)
            if conf != 0:
                gen_instances_file(net_dir, [net], prop_dir, selected_idxs, [conf], epsilons, instances_file, timeout=timeout)    
            else:
                gen_instances_file(net_dir, [net], prop_dir_normal, selected_idxs, [conf], epsilons, instances_file, timeout=timeout)   


if __name__ == '__main__':
    dataset_name = cifar10_dataset
    # setup_modified_props_gans()
    # set_up_top_k()
    # setup_on_deeppoly_images()
    # set_up_top_k_robust_paper()
    # setup_modified_props(dataset=dataset_name)
    setup_on_orig_dataset_images(dataset=dataset_name)

        



    


