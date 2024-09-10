import os
import shutil
from modify_onnx import append_layers
from generate_properties import gen_props
from generate_instance_file import gen_instances_file
from simulate_network import get_mnist_test_data
from simulate_network import get_mnist_train_data
from simulate_network import run_network_mnist_test



is_test_data = False

IMAGES, LABELS = get_mnist_train_data()
if is_test_data:
    IMAGES, LABELS = get_mnist_test_data()

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


def setup_modified_props():
    orig_net_dir = '/home/u1411251/Documents/tools/networks/conf_final/eran_mod'
    nets = ['mnist_relu_3_50.onnx', 'mnist_relu_3_100.onnx', 'mnist_relu_5_100.onnx', 'mnist_relu_6_100.onnx']
    nets += ['mnist_relu_6_200.onnx', 'mnist_relu_9_100.onnx', 'mnist_relu_9_200.onnx']
    nets = ['mnist_relu_3_50.onnx']
    confs = [0,60]
    epsilons = [0.04]
    max_low_conf_images = 5
    max_high_conf_images = 2
    setup_dir = '/home/u1411251/Documents/tools/networks/temp'
    clean_directory(setup_dir)
    net_dir = os.path.join(setup_dir, 'nets')
    prop_dir = os.path.join(setup_dir, 'props')
    instances_file = os.path.join(setup_dir, 'instances.csv')
    if os.path.isfile(instances_file):
        os.remove(instances_file)
    create_empty_dirs(net_dir, prop_dir)

    for net in nets:
        # prop_dir = os.path.join(prop_dir, net[:-5])
        # if not os.path.isdir(prop_dir):
        #     os.makedirs(prop_dir)
        for conf in confs:
            if conf != 0:
                _, _, low_confs_idx, _, high_conf_idx, _  = run_network_mnist_test(os.path.join(orig_net_dir, net), conf_th=conf)
            else:
                _, _, low_confs_idx, _, high_conf_idx, _  = run_network_mnist_test(os.path.join(orig_net_dir, net), conf_th=100)

            low_confs_idx = low_confs_idx[:max_low_conf_images]
            print(f"net: {net},conf:{conf},low conf images: {len(low_confs_idx)}")
            print(low_confs_idx)
            selected_images = IMAGES[low_confs_idx]
            selected_labels = LABELS[low_confs_idx]
            append_layers([net], orig_net_dir, net_dir, selected_images, selected_labels, low_confs_idx, is_softmax=True, confs=[conf], is_high_conf=False)
            gen_props(prop_dir, selected_images, selected_labels, low_confs_idx, epsilons) 
            gen_instances_file(net_dir, [net], prop_dir, low_confs_idx, [conf], epsilons, instances_file)

            if conf != 0:
                high_conf_idx = high_conf_idx[:max_high_conf_images]
                print(f"net: {net},conf:{conf},high conf images: {len(high_conf_idx)}")
                print(high_conf_idx)
                selected_images = IMAGES[high_conf_idx]
                selected_labels = LABELS[high_conf_idx]
                append_layers([net], orig_net_dir, net_dir, selected_images, selected_labels, high_conf_idx, is_softmax=True, confs=[conf], is_high_conf=True)
                gen_props(prop_dir, selected_images, selected_labels, high_conf_idx, epsilons)
                gen_instances_file(net_dir, [net], prop_dir, high_conf_idx, [conf], epsilons, instances_file)




if __name__ == '__main__':
    setup_modified_props()
        



    


