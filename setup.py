import os
import shutil
from modify_onnx import append_layers
from generate_properties import gen_props
from generate_instance_file import gen_instances_file
from generate_instance_file import gen_instances_file_top_k
from simulate_network import get_mnist_test_data
from simulate_network import get_mnist_train_data
from simulate_network import run_network_mnist_test
from simulate_network import select_images_top_k
from modify_onnx_top_k import append_layers_top_k



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


def setup_modified_props():
    orig_net_dir = '/home/afzal/tools/networks/conf_final/eran_mod'
    nets = ['mnist_relu_3_50.onnx', 'mnist_relu_3_100.onnx', 'mnist_relu_5_100.onnx', 'mnist_relu_6_100.onnx']
    nets += ['mnist_relu_6_200.onnx', 'mnist_relu_9_100.onnx', 'mnist_relu_9_200.onnx']
    nets = ['mnist_relu_5_100.onnx']
    confs = [60, 70, 80, 90, 95]
    epsilons = [0.06]
    max_low_conf_images = 1000
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
        low_plus_high_conf_images_idxs = []
        for conf in confs:
            _, _, low_confs_idx, _, high_conf_idx, _  = run_network_mnist_test(os.path.join(orig_net_dir, net), conf_th=conf)
            low_confs_idx = low_confs_idx[:max_low_conf_images]
            low_plus_high_conf_images_idxs += low_confs_idx
            print(f"net: {net},conf:{conf},low conf images: {len(low_confs_idx)}")
            # print(low_confs_idx)
            selected_images = IMAGES[low_confs_idx]
            selected_labels = LABELS[low_confs_idx]
            append_layers([net], orig_net_dir, net_dir, selected_images, selected_labels, low_confs_idx, is_softmax=True, confs=[conf], is_high_conf=False)
            gen_props(prop_dir, selected_images, selected_labels, low_confs_idx, epsilons) 
            gen_instances_file(net_dir, [net], prop_dir, low_confs_idx, [conf], epsilons, instances_file)

            high_conf_idx = high_conf_idx[:max_high_conf_images]
            print(f"net: {net},conf:{conf},high conf images: {len(high_conf_idx)}")
            low_plus_high_conf_images_idxs += high_conf_idx
            # print(high_conf_idx)
            selected_images = IMAGES[high_conf_idx]
            selected_labels = LABELS[high_conf_idx]
            append_layers([net], orig_net_dir, net_dir, selected_images, selected_labels, high_conf_idx, is_softmax=True, confs=[conf], is_high_conf=True)
            gen_props(prop_dir, selected_images, selected_labels, high_conf_idx, epsilons)
            gen_instances_file(net_dir, [net], prop_dir, high_conf_idx, [conf], epsilons, instances_file)            

        low_plus_high_conf_images_idxs = list(set(low_plus_high_conf_images_idxs))
        print(f"Number of images for standard prop: {len(low_plus_high_conf_images_idxs)}")
        selected_images = IMAGES[low_plus_high_conf_images_idxs]
        selected_labels = LABELS[low_plus_high_conf_images_idxs]
        append_layers([net], orig_net_dir, net_dir, selected_images, selected_labels, low_plus_high_conf_images_idxs, is_softmax=True, confs=[0], is_high_conf=False)
        prop_dir_normal = os.path.join(prop_dir, 'standard')
        if not os.path.isdir(prop_dir_normal):
            os.makedirs(prop_dir_normal)
        gen_props(prop_dir_normal, selected_images, selected_labels, low_plus_high_conf_images_idxs, epsilons, is_standard_prop=True) 
        gen_instances_file(net_dir, [net], prop_dir_normal, low_plus_high_conf_images_idxs, [0], epsilons, instances_file)

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
        append_layers_top_k([net], orig_net_dir, net_dir, selected_images, selected_top_k, selected_idexs)
        append_layers_top_k([net], orig_net_dir, net_dir, selected_images, selected_top_k, selected_idexs, is_standard_prop=True)

        prop_selected_labels = [l[0] for l in selected_top_k]
        final_out_dims = 10 - num_top_k
        gen_props(prop_dir, selected_images, prop_selected_labels, selected_idexs, epsilons, net_out_dims=final_out_dims)
        gen_instances_file_top_k(net_dir, [net], prop_dir, selected_idexs, epsilons, instances_file)

        prop_dir_normal = os.path.join(prop_dir, 'standard')
        if not os.path.isdir(prop_dir_normal):
            os.makedirs(prop_dir_normal)
        gen_props(prop_dir_normal, selected_images, prop_selected_labels, selected_idexs, epsilons, net_out_dims=final_out_dims, is_standard_prop=True)
        gen_instances_file_top_k(net_dir, [net], prop_dir_normal, selected_idexs, epsilons, instances_file, is_standard_prop=True)
                



if __name__ == '__main__':
    setup_modified_props()
    # set_up_top_k()

        



    


