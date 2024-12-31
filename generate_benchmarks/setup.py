import os
import sys
import yaml
import shutil
from PIL import Image
import csv
import numpy as np
import math
from modify_onnx import append_layers, get_delta
from modify_onnx import append_layers_vnncomp_prop
from generate_properties import gen_props
from generate_instance_file import gen_instances_file
from generate_instance_file import gen_instances_file_top_k
from generate_properties import save_vnnlib_from_vnncomp
from simulate_network import get_mnist_test_data
from simulate_network import get_mnist_train_data
from simulate_network import run_network_mnist_test
from simulate_network import select_images_top_k
from simulate_network import get_selected_images_gans, get_selected_images_gans_with_delta_th
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
    if dataset == mnist_dataset:
        if is_test_data:
            IMAGES, LABELS = get_mnist_test_data()
        else:
            IMAGES, LABELS = get_mnist_train_data()
    elif dataset == cifar10_dataset or dataset == cifar100_dataset:
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

def get_lb_conf(conf, out_dims = 10):
    delta = -math.log((100/conf) - 1)
    lb_conf = 100 / (1 + (out_dims-1)*math.exp(-delta))
    return lb_conf

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

def get_images_csv_gans(dataset_idxs_file, image_shape, start_idx, end_idx):
    with open(dataset_idxs_file, 'r') as f:
        selected_images, selected_labels, selected_indexes = [], [], []
        csv_readers = csv.reader(f, delimiter=',')
        idx = 0
        # fixed_idxs = [2,29,49,77,79,83,111,142,157,164,176]
        for row in csv_readers:
            # if idx in fixed_idxs:
            if idx >= start_idx and idx < end_idx:
                label = int(row[0])
                image = np.array(row[1:])
                image = image.reshape(image_shape)
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

def is_exist_tuple(benchmarks_list, net, ep, im_idx):
    for l in benchmarks_list:
        if l[0] == net and l[1] == ep and l[2] == im_idx:
            return True
        
    return False

def get_delta(conf):
    delta_th = -math.log((100/conf) - 1)
    delta_th = round(delta_th, 3)
    return delta_th

def setup_modified_props_gans(nets, dataset, mean, std, confs, timeout, start_idx, end_idx, is_softmax, orig_net_dir, epsilons, is_cnn, preprocessing_dir, image_shape, images_csv_file, log_dir, tolerance_param):
    confs_t = [c for c in confs if c != 0]
    confs = confs_t
    max_num_images = end_idx - start_idx
    # max_low_conf_images = int(0.9*max_num_images)
    # max_high_conf_images = max_num_images - max_low_conf_images


    setup_dir = os.path.join(preprocessing_dir, 'benchmarks')
    clean_directory(setup_dir)
    net_dir = os.path.join(setup_dir, 'nets')
    prop_dir = os.path.join(setup_dir, 'props')
    instances_file = os.path.join(setup_dir, 'instances.csv')
    if os.path.isfile(instances_file):
        os.remove(instances_file)
    create_empty_dirs(net_dir, prop_dir)

    g_images, g_labels, g_indexes = get_images_csv_gans(images_csv_file, image_shape, start_idx=start_idx, end_idx=end_idx)
    g_images = np.array(g_images)
    g_labels = np.array(g_labels)

    for net in nets:
        low_plus_high_conf_images_idxs = []
        for conf in confs:
            # lb_conf = get_lb_conf(conf)
<<<<<<< HEAD
            delta_th = get_delta(conf)
=======
            delta_th = get_delta(conf=conf)
>>>>>>> 291e930ee7def9855db5df6f692548bacc185419
            # high_confs_idx, low_confs_idx= get_selected_images_gans(os.path.join(orig_net_dir, net), g_images, g_indexes, lb_conf, image_shape=image_shape)
            high_confs_idx, low_confs_idx= get_selected_images_gans_with_delta_th(os.path.join(orig_net_dir, net), g_images, g_indexes, delta_th, image_shape=image_shape)
            selected_idxs = low_confs_idx
            low_plus_high_conf_images_idxs += selected_idxs
            selected_images = IMAGES[selected_idxs]
            selected_labels = LABELS[selected_idxs]
            print(f"net: {net},conf:{conf},delta:{delta_th},low conf images: {len(selected_idxs)}")
            prop_dir1 = os.path.join(prop_dir, net[:-5])
            if not os.path.isdir(prop_dir1):
                os.makedirs(prop_dir1)
            append_layers([net], orig_net_dir, net_dir, selected_images, selected_labels, selected_idxs, is_softmax=is_softmax, confs=[conf], is_high_conf=False)
            gen_props(prop_dir1, selected_images, selected_labels, selected_idxs, epsilons,conf=conf, mean=mean, std=std, dataset=dataset, tolerance_param=tolerance_param)
            gen_instances_file(net_dir, [net], prop_dir1, selected_idxs, [conf], epsilons, instances_file, timeout=timeout)
                
            # high_confs_idx = high_confs_idx[:max_high_conf_images]
            print(f"net: {net},conf:{conf},delta:{delta_th},high conf images: {len(high_confs_idx)}")
            low_plus_high_conf_images_idxs += high_confs_idx
            # print(high_conf_idx)
            selected_idxs = high_confs_idx
            selected_images = IMAGES[high_confs_idx]
            selected_labels = LABELS[high_confs_idx]
            append_layers([net], orig_net_dir, net_dir, selected_images, selected_labels, selected_idxs, is_softmax=is_softmax, confs=[conf], is_high_conf=True)
            gen_props(prop_dir1, selected_images, selected_labels, selected_idxs, epsilons,conf=conf, mean=mean, std=std, dataset=dataset, is_standard_prop=True, tolerance_param=tolerance_param)
            gen_instances_file(net_dir, [net], prop_dir1, selected_idxs, [conf], epsilons, instances_file, timeout=timeout)

        low_plus_high_conf_images_idxs = list(set(low_plus_high_conf_images_idxs))
        print(f"Number of images for standard prop: {len(low_plus_high_conf_images_idxs)}")
        selected_images = IMAGES[low_plus_high_conf_images_idxs]
        selected_labels = LABELS[low_plus_high_conf_images_idxs]
        append_layers([net], orig_net_dir, net_dir, selected_images, selected_labels, low_plus_high_conf_images_idxs, is_softmax=is_softmax, confs=[0], is_high_conf=False)
        prop_dir_normal = os.path.join(prop_dir, 'standard')
        if not os.path.isdir(prop_dir_normal):
            os.makedirs(prop_dir_normal)
        gen_props(prop_dir_normal, selected_images, selected_labels, low_plus_high_conf_images_idxs, epsilons, conf=0, is_standard_prop=True, mean=mean, std=std, dataset=dataset, tolerance_param=tolerance_param) 
        gen_instances_file(net_dir, [net], prop_dir_normal, low_plus_high_conf_images_idxs, [0], epsilons, instances_file, timeout=timeout)

def setup_modified_props_one_hop(nets, dataset, mean, std, confs, timeout, max_num_images, is_softmax, orig_net_dir, epsilons, is_cnn, preprocessing_dir, image_shape, images_csv_file, log_dir):

    setup_dir = os.path.join(preprocessing_dir, 'benchmarks')
    clean_directory(setup_dir)
    net_dir = os.path.join(setup_dir, 'nets')
    prop_dir = os.path.join(setup_dir, 'props')
    instances_file = os.path.join(setup_dir, 'instances.csv')
    if os.path.isfile(instances_file):
        os.remove(instances_file)
    create_empty_dirs(net_dir, prop_dir)

    print(log_dir)
    fn_dir = os.path.join(log_dir, 'cex', 'fn', 'npy')
    if not os.path.isdir(fn_dir):
        print(f"Wrong false negative directory: {fn_dir}")
    list_already_build = []
    for filename in os.listdir(fn_dir):
        filename1 = filename[:-4]
        filename1_l = filename1.split('+')
        net = filename1_l[0]+".onnx"
        im_idx = int(filename1_l[1])
        try:
            conf = int(filename1_l[2])
        except:
            conf = float(filename1_l[2])
        
        ep= float(filename1_l[3])
        epsilons = [ep]
        label = int(filename1_l[4])
        delta_th = get_delta(conf)
        np_filename = os.path.join(fn_dir, filename)
        image = np.load(np_filename)
        image_shape1 = (1,) + image_shape
        image = image.reshape(image_shape1)
        image = image.astype(np.float32)
        selected_images = image
        selected_labels = np.array([label])
        selected_idxs = [im_idx]
        high_confs_idx, low_confs_idx= get_selected_images_gans_with_delta_th(os.path.join(orig_net_dir, net), selected_images, selected_idxs, delta_th, image_shape=image_shape)
        is_high_conf = False
        if len(high_confs_idx) > 0:
            is_high_conf = True
        prop_dir1 = os.path.join(prop_dir, net[:-5])
        if not os.path.isdir(prop_dir1):
            os.makedirs(prop_dir1)
        if not is_high_conf:
            append_layers([net], orig_net_dir, net_dir, selected_images, selected_labels, selected_idxs, is_softmax=is_softmax, confs=[conf], is_high_conf=False)
            gen_props(prop_dir1, selected_images, selected_labels, selected_idxs, epsilons,conf=conf, mean=mean, std=std, dataset=dataset)
            gen_instances_file(net_dir, [net], prop_dir1, selected_idxs, [conf], epsilons, instances_file, timeout=timeout)
        else:
            append_layers([net], orig_net_dir, net_dir, selected_images, selected_labels, selected_idxs, is_softmax=is_softmax, confs=[conf], is_high_conf=True)
            gen_props(prop_dir1, selected_images, selected_labels, selected_idxs, epsilons,conf=conf, mean=mean, std=std, dataset=dataset, is_standard_prop=True)
            gen_instances_file(net_dir, [net], prop_dir1, selected_idxs, [conf], epsilons, instances_file, timeout=timeout)

        # if not is_exist_tuple(list_already_build, net, ep, im_idx):
        #     append_layers([net], orig_net_dir, net_dir, selected_images, selected_labels, selected_idxs, is_softmax=is_softmax, confs=[0], is_high_conf=False)
        #     prop_dir_normal = os.path.join(prop_dir, 'standard')
        #     if not os.path.isdir(prop_dir_normal):
        #         os.makedirs(prop_dir_normal)
        #     gen_props(prop_dir_normal, selected_images, selected_labels, selected_idxs, epsilons, conf=0, is_standard_prop=True, mean=mean, std=std, dataset=dataset) 
        #     gen_instances_file(net_dir, [net], prop_dir_normal, selected_idxs, [0], epsilons, instances_file, timeout=timeout)
        #     l = [net, ep, im_idx]
        #     list_already_build.append(l)
        # else:
        #     print(f"Already build:  {net}, {ep}, {im_idx}")




    
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

def setup_modified_props_special(nets, dataset, mean, std, confs, timeout, max_num_images, is_softmax, net_root_dir, orig_net_dir, epsilons, is_cnn, preprocessing_dir):
    im_dirs = '/home/afzal/temp/fn_images'
    confs = [60]
    setup_dir = os.path.join(preprocessing_dir, 'benchmarks')
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
                
            # print(f"Number of images for standard prop: {len(selected_idxs)}")
            # append_layers([net], orig_net_dir, net_dir, selected_images, selected_labels, selected_idxs, is_softmax=is_softmax, confs=[0], is_high_conf=False)
            # gen_props(prop_dir, selected_images, selected_labels, selected_idxs, epsilons, conf=0, is_standard_prop=True, mean=mean, std=std, dataset=dataset) 
            # gen_instances_file(net_dir, [net], prop_dir, selected_idxs, [0], epsilons, instances_file, timeout=timeout)



def setup_modified_props(nets, dataset, mean, std, confs, timeout, start_idx, end_idx, is_softmax, orig_net_dir, epsilons, is_cnn, preprocessing_dir, image_shape, dataset_idxs_file):
    confs_t = [c for c in confs if c != 0]
    confs = confs_t
    max_num_images = end_idx - start_idx
    max_low_conf_images = int(0.9*max_num_images)
    max_high_conf_images = max_num_images - max_low_conf_images

    if dataset == mnist_dataset:
        filter_images = run_network_mnist_test
    elif dataset == cifar10_dataset:
        filter_images = run_network_cifar10


    setup_dir = os.path.join(preprocessing_dir, 'benchmarks')
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
            print(os.path.join(orig_net_dir, net))
            _, _, low_confs_idx, _, high_conf_idx, _  = filter_images(os.path.join(orig_net_dir, net), conf_th=conf, is_cnn=is_cnn, mean=mean, std=std, is_test_dataset=is_test_data)
            low_confs_idx = low_confs_idx[:max_low_conf_images]
            selected_idxs = low_confs_idx
            low_plus_high_conf_images_idxs += selected_idxs
            selected_images = IMAGES[selected_idxs]
            selected_labels = LABELS[selected_idxs]
            print(f"net: {net},conf:{conf},low conf images: {len(selected_idxs)}")
            prop_dir1 = os.path.join(prop_dir, net[:-5])
            if not os.path.isdir(prop_dir1):
                os.makedirs(prop_dir1)
            append_layers([net], orig_net_dir, net_dir, selected_images, selected_labels, selected_idxs, is_softmax=is_softmax, confs=[conf], is_high_conf=False)
            print(prop_dir1)
            gen_props(prop_dir1, selected_images, selected_labels, selected_idxs, epsilons,conf=conf, mean=mean, std=std, dataset=dataset)
            gen_instances_file(net_dir, [net], prop_dir1, selected_idxs, [conf], epsilons, instances_file, timeout=timeout)
                
            high_conf_idx = high_conf_idx[:max_high_conf_images]
            print(f"net: {net},conf:{conf},high conf images: {len(high_conf_idx)}")
            low_plus_high_conf_images_idxs += high_conf_idx
            # print(high_conf_idx)
            selected_idxs = high_conf_idx
            selected_images = IMAGES[high_conf_idx]
            selected_labels = LABELS[high_conf_idx]
            append_layers([net], orig_net_dir, net_dir, selected_images, selected_labels, selected_idxs, is_softmax=is_softmax, confs=[conf], is_high_conf=True)
            print(prop_dir1)
            gen_props(prop_dir1, selected_images, selected_labels, selected_idxs, epsilons,conf=conf, mean=mean, std=std, dataset=dataset, is_standard_prop=True)
            gen_instances_file(net_dir, [net], prop_dir1, selected_idxs, [conf], epsilons, instances_file, timeout=timeout)

        low_plus_high_conf_images_idxs = list(set(low_plus_high_conf_images_idxs))
        print(f"Number of images for standard prop: {len(low_plus_high_conf_images_idxs)}")
        selected_images = IMAGES[low_plus_high_conf_images_idxs]
        selected_labels = LABELS[low_plus_high_conf_images_idxs]
        append_layers([net], orig_net_dir, net_dir, selected_images, selected_labels, low_plus_high_conf_images_idxs, is_softmax=is_softmax, confs=[0], is_high_conf=False)
        prop_dir_normal = os.path.join(prop_dir, 'standard')
        if not os.path.isdir(prop_dir_normal):
            os.makedirs(prop_dir_normal)
        gen_props(prop_dir_normal, selected_images, selected_labels, low_plus_high_conf_images_idxs, epsilons, conf=0, is_standard_prop=True, mean=mean, std=std, dataset=dataset) 
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





def select_images_with_labels(dataset_idxs_file, max_num_indexs=50):
    with open(dataset_idxs_file) as f:
        line = f.readline()
        indexes = np.fromstring(line, dtype=int, sep=',')
        indexes = indexes[:max_num_indexs]
        images = IMAGES[indexes]
        labels = LABELS[indexes] 
    
    return images, labels, indexes

def select_images_with_labels_first(dataset_idxs_file, start_idx = 0, end_idx=50):
    indexes = [i for i in range(start_idx, end_idx)]
    images = IMAGES[indexes]
    labels = LABELS[indexes] 
    return images, labels, indexes

def set_images_labels_gan_with_oracle(image_csv, image_shape):
    global IMAGES, LABELS
    with open(image_csv, 'r') as f:
        selected_images, selected_labels = [], []
        csv_readers = csv.reader(f, delimiter=',')
        for row in csv_readers:
            label = int(row[0])
            image = np.array(row[1:])
            image = image.reshape(image_shape)
            image = image.astype(np.float32)
            selected_images.append(image)
            selected_labels.append(label)

    IMAGES = np.array(selected_images)
    LABELS = np.array(selected_labels)
    print(IMAGES.shape)
    print(LABELS.shape)

def setup_on_orig_dataset_images(nets, dataset, mean, std, confs, timeout, start_idx, end_idx, is_softmax, orig_net_dir, dataset_idxs_file, epsilons, preprocessing_dir, image_shape):

    setup_dir = os.path.join(preprocessing_dir, 'benchmarks')
    clean_directory(setup_dir)
    net_dir = os.path.join(setup_dir, 'nets')
    prop_dir = os.path.join(setup_dir, 'props')
    instances_file = os.path.join(setup_dir, 'instances.csv')
    if os.path.isfile(instances_file):
        os.remove(instances_file)
    create_empty_dirs(net_dir, prop_dir)

    selected_images, selected_labels, selected_idxs = select_images_with_labels_first(dataset_idxs_file, start_idx=start_idx, end_idx=end_idx)
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

def get_label_vnncomp_prp(prp_file):
    with open(prp_file, 'r') as file:
        first_line = file.readline().strip()
        # Extract the number after 'label: '
        if "; CIFAR100 property with label:" in first_line:
            label = first_line.split("label:")[-1].strip().rstrip('.')
            return int(label) 


def setup_on_vnncomp_prop(dataset, confs, timeout, epsilons, preprocessing_dir, vnncomp_benchmarks_dir, tolerance_param):
    print(dataset, confs, timeout, epsilons, preprocessing_dir, vnncomp_benchmarks_dir, tolerance_param)
    setup_dir = os.path.join(preprocessing_dir, 'benchmarks')
    clean_directory(setup_dir)
    net_dir = os.path.join(setup_dir, 'nets')
    prop_dir = os.path.join(setup_dir, 'props')
    instances_file = os.path.join(setup_dir, 'instances.csv')
    if os.path.isfile(instances_file):
        os.remove(instances_file)
    create_empty_dirs(net_dir, prop_dir)

    vnncomp_instance_file = os.path.join(vnncomp_benchmarks_dir, 'instances.csv')

    with open(vnncomp_instance_file, 'r') as vnncomp_instance_f:
        idx = 0
        ep = epsilons[0]
        output_dims = 200
        instance_lines = []
        for line in vnncomp_instance_f:
            if 'CIFAR100' in line:
                output_dims=100
            else:
                output_dims=200
            line = line.strip()
            line_l = line.split(',')
            vnncomp_net_path = os.path.join(vnncomp_benchmarks_dir,  line_l[0])
            vnncomp_prp_path = os.path.join(vnncomp_benchmarks_dir, line_l[1])
            _, netname = os.path.split(vnncomp_net_path)
            _, prpname = os.path.split(vnncomp_prp_path)
            label = get_label_vnncomp_prp(vnncomp_prp_path)
            for conf in confs:
                target_net_path = os.path.join(net_dir, f"{netname[:-5]}_{idx}_{ep}_{conf}.onnx")
                target_prp_path = os.path.join(prop_dir, f"{prpname[:-7]}_{idx}_{ep}_{conf}.vnnlib")
                append_layers_vnncomp_prop(input_net_path=vnncomp_net_path, target_net_path=target_net_path, conf=conf, orig_label=label, existing_output_dim=output_dims)
                save_vnnlib_from_vnncomp(vnncomp_prp_path, target_prp_path, conf=conf, total_output_class=output_dims-1, tolerance_param=tolerance_param)
                ins_line = f"{target_net_path},{target_prp_path},{timeout}\n"
                instance_lines.append(ins_line)
            idx += 1

        with open(instances_file, 'w') as f:
            f.writelines(instance_lines)

def conf_delta():
    confs = [70,73]
    for conf in confs:
        delta = -math.log((100/conf) - 1)
        delta = round(delta, 2)
        conf1 = 100/(1+math.exp(-delta))
        print(conf, delta, conf1)


if __name__ == '__main__':
    # conf_delta()
    # exit(0)
    potential_datasets = [mnist_dataset, cifar10_dataset, cifar100_dataset]
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    else:
        assert False, "Please provide the config file"

    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    
    is_test_data = config.get('is_test_data', None)
    dataset = config.get('dataset', None)
    is_gans_input = config.get('is_gans_input', None)
    images_csv_file = config.get('images_csv_file', None)
    image_shape = tuple(config.get('image_shape', [1,784,1]))
    mean = np.array(config.get('mean', None), dtype=np.float32)
    std = np.array(config.get('std', None), dtype=np.float32)
    net_dir = config.get('net_dir', None)
    nets = config.get('nets', None)
    confs = config.get('confs', None)
    timeout = config.get('timeout', None)
    image_indexes_range = config.get('image_indexes_range', [0,100])
    start_idx = image_indexes_range[0]
    end_idx = image_indexes_range[1]
    is_softmax = config.get('is_softmax',None)
    epsilons = config.get('epsilons', None)
    dataset_idxs_file = config.get('dataset_idxs_file', None)
    tolerance_param = config.get('tolerance_param', None)
    preprocessing_dir = config.get('preprocessing_dir', None)
    vnncomp_benchmarks_dir = config.get('vnncomp_benchmarks_dir', None)
    property_type = config.get('property', None)
    is_cnn = config.get('is_cnn', None)
    log_dir = config.get('log_dir', None)
    
    assert dataset in potential_datasets, "Invalid dataset"
    if is_gans_input:
        set_images_labels_gan_with_oracle(images_csv_file, image_shape)
    else:
        set_images_labels(dataset, is_test_data)

    if property_type  == 'low_conf_cex':
        setup_on_vnncomp_prop(nets=nets, 
                                     dataset=dataset, 
                                     mean=mean,
                                     std=std,
                                     confs=confs,
                                     timeout=timeout,
                                     start_idx=start_idx,
                                     end_idx=end_idx,
                                     is_softmax=is_softmax,
                                     orig_net_dir=net_dir,
                                     dataset_idxs_file=dataset_idxs_file,
                                     epsilons=epsilons, 
                                     preprocessing_dir=preprocessing_dir, 
                                     vnncomp_benchmarks_dir=vnncomp_benchmarks_dir,
                                     image_shape=image_shape, 
                                     tolerance_param=tolerance_param
                                    )
    elif property_type == 'vnncomp':
         setup_on_vnncomp_prop(dataset=dataset, 
                                     confs=confs,
                                     timeout=timeout,
                                     epsilons=epsilons, 
                                     preprocessing_dir=preprocessing_dir, 
                                     vnncomp_benchmarks_dir=vnncomp_benchmarks_dir,
                                     tolerance_param=tolerance_param
                                    )
    else:
        if not is_gans_input:
            print(f"Please enable the Gans input in config file: {config_file}")
            exit(0)
        # setup_modified_props_gans(nets=nets, 
        #                      dataset=dataset, 
        #                      mean=mean,
        #                      std=std,
        #                      confs=confs,
        #                      timeout=timeout,
        #                      start_idx=start_idx,
        #                      end_idx=end_idx,
        #                      is_softmax=is_softmax,
        #                      orig_net_dir=net_dir, 
        #                      epsilons=epsilons, 
        #                      is_cnn=is_cnn,
        #                      preprocessing_dir=preprocessing_dir, 
        #                      image_shape=image_shape,
        #                      images_csv_file = images_csv_file,
        #                      log_dir = log_dir, 
        #                      tolerance_param=tolerance_param
        #                      )
        
        setup_modified_props_one_hop(nets=nets, 
                             dataset=dataset, 
                             mean=mean,
                             std=std,
                             confs=confs,
                             timeout=timeout,
                             start_idx=start_idx,
                             end_idx=end_idx,
                             is_softmax=is_softmax,
                             orig_net_dir=net_dir, 
                             epsilons=epsilons, 
                             is_cnn=is_cnn,
                             preprocessing_dir=preprocessing_dir, 
                             image_shape=image_shape,
                             images_csv_file = images_csv_file,
                             log_dir = log_dir, 
                             tolerance_param=tolerance_param
                             )

    # dataset_name = cifar10_dataset
    # set_parameters()
    # dataset_name = mnist_dataset
    # setup_modified_props_gans()
    # set_up_top_k()
    # setup_on_deeppoly_images()
    # set_up_top_k_robust_paper()
    # setup_modified_props(dataset=dataset_name)
    # setup_on_orig_dataset_images(dataset=dataset_name)

        



    


