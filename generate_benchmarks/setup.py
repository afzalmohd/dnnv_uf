import os
import sys
import yaml
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

mnist_dataset = 'MNIST'
cifar10_dataset = 'CIFAR10'
cifar100_dataset='CIFAR100'
imagenet_dataset = 'imagenet'
tsr_dataset = 'tsr'


def set_images_labels(dataset, is_test_data):
    global IMAGES, LABELS
    if dataset == 'MNIST':
        if is_test_data:
            IMAGES, LABELS = get_mnist_test_data()
        else:
            IMAGES, LABELS = get_mnist_train_data()
    elif dataset == 'CIFAR10':
        if is_test_data:
            IMAGES, LABELS = get_cifar10_test_data()
        else:
            IMAGES, LABELS = get_cifar10_train_data()
        
        IMAGES = np.transpose(IMAGES, (0, 3, 1, 2))
        LABELS = LABELS.reshape(-1)
    
    print(IMAGES.shape)
    print(LABELS.shape)

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
                selected_images = IMAGES[low_confs_idx]
                selected_labels = LABELS[low_confs_idx]
                append_layers([net], orig_net_dir, net_dir, selected_images, selected_labels, low_confs_idx, is_softmax=True, confs=[conf], is_high_conf=False)
                gen_props(prop_dir, selected_images, selected_labels, low_confs_idx, epsilons) 
                gen_instances_file(net_dir, [net], prop_dir, low_confs_idx, [conf], epsilons, instances_file)

                high_conf_idx = high_conf_idx[:max_high_conf_images]
                print(f"net: {net},conf:{conf},high conf images: {len(high_conf_idx)}")
                print(high_conf_idx)
                selected_images = IMAGES[high_conf_idx]
                selected_labels = LABELS[high_conf_idx]
                append_layers([net], orig_net_dir, net_dir, selected_images, selected_labels, high_conf_idx, is_softmax=True, confs=[conf], is_high_conf=True)
                gen_props(prop_dir, selected_images, selected_labels, high_conf_idx, epsilons)
                gen_instances_file(net_dir, [net], prop_dir, high_conf_idx, [conf], epsilons, instances_file)
            else:
                _, _, low_confs_idx, _, high_conf_idx, _  = run_network_mnist_test(os.path.join(orig_net_dir, net), conf_th=100)
                low_confs_idx = low_confs_idx[:max_low_conf_images]
                print(f"net: {net},conf:{conf},low conf images: {len(low_confs_idx)}")
                print(low_confs_idx)
                selected_images = IMAGES[low_confs_idx]
                selected_labels = LABELS[low_confs_idx]
                append_layers([net], orig_net_dir, net_dir, selected_images, selected_labels, low_confs_idx, is_softmax=True, confs=[conf], is_high_conf=False)
                prop_dir_normal = os.path.join(prop_dir, 'standard')
                if not os.path.isdir(prop_dir_normal):
                    os.makedirs(prop_dir_normal)
                gen_props(prop_dir_normal, selected_images, selected_labels, low_confs_idx, epsilons, is_standard_prop=True) 
                gen_instances_file(net_dir, [net], prop_dir_normal, low_confs_idx, [conf], epsilons, instances_file)

def setup_modified_props_special(nets, dataset, mean, std, confs, timeout, max_num_images, is_softmax, net_root_dir, orig_net_dir, epsilons, is_cnn):
    im_dirs = '/home/afzal/temp/fn_images'
    confs = [60]
    setup_dir = os.path.join(net_root_dir, 'benchmarks')
    clean_directory(setup_dir)
    net_dir = os.path.join(setup_dir, 'nets')
    prop_dir = os.path.join(setup_dir, 'props')
    instances_file = os.path.join(setup_dir, 'instances.csv')
    if os.path.isfile(instances_file):
        os.remove(instances_file)
    create_empty_dirs(net_dir, prop_dir)
    images = []
    idxs = []
    labels = []
    for im in os.listdir(im_dirs):
        file = os.path.join(im_dirs, im)
        im = os.path.basename(im[:-4])
        im_idx = int(im.split('_')[1])
        loaded_im = np.load(file)
        loaded_im = loaded_im.reshape(1,-1,1)
        images.append(loaded_im)
        idxs.append(im_idx)
        labels.append(LABELS[im_idx])

    images = np.array(images)
    idxs = np.array(idxs).reshape(-1)
    labels = np.array(labels).reshape(-1)

    for net in nets:
        for conf in confs:
            selected_idxs = idxs
            selected_images = images
            selected_labels = labels
            print(f"net: {net},conf:{conf},low conf images: {len(selected_idxs)}")
            
            append_layers([net], orig_net_dir, net_dir, selected_images, selected_labels, selected_idxs, is_softmax=is_softmax, confs=[conf], is_high_conf=False)
            gen_props(prop_dir, selected_images, selected_labels, selected_idxs, epsilons,conf=conf, mean=mean, std=std, dataset=dataset)
            gen_instances_file(net_dir, [net], prop_dir, selected_idxs, [conf], epsilons, instances_file, timeout=timeout)
                
            print(f"Number of images for standard prop: {len(selected_idxs)}")
            append_layers([net], orig_net_dir, net_dir, selected_images, selected_labels, selected_idxs, is_softmax=is_softmax, confs=[0], is_high_conf=False)
            gen_props(prop_dir, selected_images, selected_labels, selected_idxs, epsilons, conf=0, is_standard_prop=True, mean=mean, std=std, dataset=dataset) 
            gen_instances_file(net_dir, [net], prop_dir, selected_idxs, [0], epsilons, instances_file, timeout=timeout)
           

def setup_modified_props(nets, dataset, mean, std, confs, timeout, max_num_images, is_softmax, net_root_dir, orig_net_dir, epsilons, is_cnn):
    confs_t = [c for c in confs if c != 0]
    confs = confs_t
    max_low_conf_images = int(0.9*max_num_images)
    max_high_conf_images = max_num_images - max_low_conf_images

    if dataset == mnist_dataset:
        filter_images = run_network_mnist_test
    elif dataset == cifar10_dataset:
        filter_images = run_network_cifar10


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
            _, _, low_confs_idx, _, high_conf_idx, _  = filter_images(os.path.join(orig_net_dir, net), conf_th=conf, is_cnn=is_cnn, mean=mean, std=std, is_test_dataset=is_test_data)
            low_confs_idx = low_confs_idx[:max_low_conf_images]
            selected_idxs = low_confs_idx
            low_plus_high_conf_images_idxs += selected_idxs
            selected_images = IMAGES[selected_idxs]
            selected_labels = LABELS[selected_idxs]
            print(f"net: {net},conf:{conf},low conf images: {len(selected_idxs)}")
            
            append_layers([net], orig_net_dir, net_dir, selected_images, selected_labels, selected_idxs, is_softmax=is_softmax, confs=[conf], is_high_conf=False)
            gen_props(prop_dir, selected_images, selected_labels, selected_idxs, epsilons,conf=conf, mean=mean, std=std, dataset=dataset)
            gen_instances_file(net_dir, [net], prop_dir, selected_idxs, [conf], epsilons, instances_file, timeout=timeout)
                
            high_conf_idx = high_conf_idx[:max_high_conf_images]
            print(f"net: {net},conf:{conf},high conf images: {len(high_conf_idx)}")
            low_plus_high_conf_images_idxs += high_conf_idx
            # print(high_conf_idx)
            selected_idxs = high_conf_idx
            selected_images = IMAGES[high_conf_idx]
            selected_labels = LABELS[high_conf_idx]
            append_layers([net], orig_net_dir, net_dir, selected_images, selected_labels, selected_idxs, is_softmax=is_softmax, confs=[conf], is_high_conf=True)
            gen_props(prop_dir, selected_images, selected_labels, selected_idxs, epsilons,conf=conf, mean=mean, std=std, dataset=dataset, is_standard_prop=True)
            gen_instances_file(net_dir, [net], prop_dir, selected_idxs, [conf], epsilons, instances_file, timeout=timeout)

        low_plus_high_conf_images_idxs = list(set(low_plus_high_conf_images_idxs))
        print(f"Number of images for standard prop: {len(low_plus_high_conf_images_idxs)}")
        selected_images = IMAGES[low_plus_high_conf_images_idxs]
        selected_labels = LABELS[low_plus_high_conf_images_idxs]
        append_layers([net], orig_net_dir, net_dir, selected_images, selected_labels, low_plus_high_conf_images_idxs, is_softmax=is_softmax, confs=[0], is_high_conf=False)
        # prop_dir_normal = os.path.join(prop_dir, 'standard')
        # if not os.path.isdir(prop_dir_normal):
        #     os.makedirs(prop_dir_normal)
        gen_props(prop_dir, selected_images, selected_labels, low_plus_high_conf_images_idxs, epsilons, conf=0, is_standard_prop=True, mean=mean, std=std, dataset=dataset) 
        gen_instances_file(net_dir, [net], prop_dir, low_plus_high_conf_images_idxs, [0], epsilons, instances_file, timeout=timeout)




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
        selected_images = IMAGES[selected_idexs]
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
        selected_images = IMAGES[selected_idexs]
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




def get_deeppoly_images(num_images = 21):
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





def select_images_with_labels(dataset, dataset_idxs_file, max_num_indexs=50):
    with open(dataset_idxs_file) as f:
        line = f.readline()
        indexes = np.fromstring(line, dtype=int, sep=',')
        indexes = indexes[:max_num_indexs]
        images = IMAGES[indexes]
        labels = LABELS[indexes] 
    
    return images, labels, indexes


def setup_on_orig_dataset_images(nets, dataset, mean, std, confs, timeout, max_num_images, is_softmax, net_root_dir, orig_net_dir, dataset_idxs_file, epsilons):

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
    for conf in confs:
        gen_props(prop_dir, selected_images, selected_labels, selected_idxs, epsilons,conf=conf, tolerance_param=-1e-5, dataset=dataset, mean=mean, std=std) 
    # prop_dir_normal = os.path.join(prop_dir, 'standard')
    # if not os.path.isdir(prop_dir_normal):
    #     os.makedirs(prop_dir_normal)
    # gen_props(prop_dir_normal, selected_images, selected_labels, selected_idxs, epsilons, is_standard_prop=True, mean=mean, std=std, dataset=dataset)
    append_layers(nets, orig_net_dir, net_dir, selected_images, selected_labels, selected_idxs,confs=confs, is_softmax=is_softmax)
    gen_instances_file(net_dir, nets, prop_dir, selected_idxs, confs, epsilons, instances_file, timeout=timeout)
    # for net in nets:
    #     for conf in confs:
    #         append_layers([net], orig_net_dir, net_dir, selected_images, selected_labels, selected_idxs, is_softmax=is_softmax, confs=[conf], is_high_conf=False)
    #         if conf != 0:
    #             gen_instances_file(net_dir, [net], prop_dir, selected_idxs, [conf], epsilons, instances_file, timeout=timeout)    
    #         else:
    #             gen_instances_file(net_dir, [net], prop_dir_normal, selected_idxs, [conf], epsilons, instances_file, timeout=timeout)   



if __name__ == '__main__':
    potential_datasets = [mnist_dataset, cifar10_dataset]
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    else:
        assert False, "Please provide the config file"

    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)

    is_test_data = config['is_test_data']
    dataset = config['dataset']
    assert dataset in potential_datasets, "Invalid dataset"
    set_images_labels(dataset, is_test_data)
    mean = np.array(config['mean'], dtype=np.float32)
    std = np.array(config['std'], dtype=np.float32)
    net_root_dir = config['net_root_dir']
    nets = config['nets']
    confs = config['confs']
    timeout = config['timeout']
    max_num_images = config['max_num_images']
    is_softmax = config['is_softmax']
    epsilons = config['epsilons']
    if dataset == mnist_dataset:
        dataset_idxs_file = os.path.join(net_root_dir, 'mnist', config['dataset_idxs_file'])
        orig_net_dir = os.path.join(net_root_dir, 'mnist', 'vnncomp')
    elif dataset == cifar10_dataset:
        dataset_idxs_file = os.path.join(net_root_dir, 'cifar10', config['dataset_idxs_file'])
        orig_net_dir = os.path.join(net_root_dir, 'cifar10', 'vnncomp')
    
    print(nets, confs, timeout, max_num_images, is_softmax, epsilons)
    print(dataset_idxs_file, orig_net_dir)

    property_type = config['property']
    is_cnn = config['is_cnn']

    if property_type  == 'low_conf_cex':
        setup_on_orig_dataset_images(nets=nets, 
                                     dataset=dataset, 
                                     mean=mean,
                                     std=std,
                                     confs=confs,
                                     timeout=timeout,
                                     max_num_images=max_num_images,
                                     is_softmax=is_softmax,
                                     net_root_dir=net_root_dir,
                                     orig_net_dir=orig_net_dir,
                                     dataset_idxs_file=dataset_idxs_file,
                                     epsilons=epsilons
                                    )
    else:
        setup_modified_props_special(nets=nets, 
                             dataset=dataset, 
                             mean=mean,
                             std=std,
                             confs=confs,
                             timeout=timeout,
                             max_num_images=max_num_images,
                             is_softmax=is_softmax,
                             net_root_dir=net_root_dir,
                             orig_net_dir=orig_net_dir, 
                             epsilons=epsilons, 
                             is_cnn=is_cnn)

    # dataset_name = cifar10_dataset
    # set_parameters()
    # dataset_name = mnist_dataset
    # setup_modified_props_gans()
    # set_up_top_k()
    # setup_on_deeppoly_images()
    # set_up_top_k_robust_paper()
    # setup_modified_props(dataset=dataset_name)
    # setup_on_orig_dataset_images(dataset=dataset_name)

        



    


