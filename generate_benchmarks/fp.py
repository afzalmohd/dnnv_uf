import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
print(sys.path)
import yaml
import numpy as np
import shutil
import onnxruntime as ort
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from generate_benchmarks.setup import clean_directory, get_output_dims, set_images_labels, select_images_with_labels, get_image_with_label
from generate_benchmarks.generate_properties import create_input_bounds_tf, save_vnnlib_tf_standard, save_vnnlib_oracle_guided
from generate_benchmarks.modify_onnx_fp import update_fc_relu_oracle
from generate_benchmarks.setup import get_final_dirs
softmax = torch.nn.Softmax(dim=1)
# transform = transforms.Compose([
#     transforms.ToTensor(), 
#     # transforms.Normalize((0.1307,), (0.3081,)),
#     transforms.Lambda(lambda x: x.view(-1))
# ])

# is_training_dataset = True
# batch_size = 1
# dataset = MNIST(root='../data', train=is_training_dataset, download=True, transform=transform)
# dataset_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

def gen_props_standard(spec_dir, selected_images, selected_labels, selected_idxs, eps, dataset='MNIST'):
    if dataset == 'CIFAR10':
        gen_props_standard_cifar10(spec_dir, selected_images, selected_labels, selected_idxs, eps)
    else:
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

def gen_props_standard_cifar10(spec_dir, selected_images, selected_labels, selected_idxs, eps):
    mean = np.array([0.4914,0.4822,0.4465])
    std = np.array([0.2470,0.2435,0.2616])
    dataset = 'CIFAR10'
    counter = 0
    for i in range(len(selected_images)):
        image = selected_images[i]
        label = selected_labels[i]
        idx = selected_idxs[i]
        for ep in eps:
            lb,ub = create_input_bounds_tf(image, ep, mean=mean, std=std, dataset=dataset)
            spec_path = f"prop_{idx}_{ep}.vnnlib"
            spec_path = os.path.join(spec_dir, spec_path)
            save_vnnlib_tf_standard(lb, ub, label, spec_path, dataset)
            counter += 1

    print(f"Total number of props: {counter}")


def gen_single_props_standard_cifar10(spec_path, im, label, ep):
    mean = np.array([0.4914,0.4822,0.4465])
    std = np.array([0.2470,0.2435,0.2616])
    dataset = 'CIFAR10'
    lb,ub = create_input_bounds_tf(im, ep, mean=mean, std=std, dataset=dataset)
    save_vnnlib_tf_standard(lb, ub, label, spec_path, dataset)

def gen_single_props_standard(spec_path, im, label, ep):
    im = im.reshape(784)
    lb,ub = create_input_bounds_tf(im, ep)
    save_vnnlib_tf_standard(lb, ub, label, spec_path, dataset)

def gen_single_props_oracle_guided(spec_path, im, label, ep, dataset='MNIST'):
    if dataset == 'MNIST':
        im = im.reshape(784)
        lb,ub = create_input_bounds_tf(im, ep)
    else:
        mean = np.array([0.4914,0.4822,0.4465])
        std = np.array([0.2470,0.2435,0.2616])
        lb,ub = create_input_bounds_tf(im, ep, mean=mean, std=std, dataset=dataset)
    
    save_vnnlib_oracle_guided(lb, ub, label, spec_path)

def is_classified_correctly(net_path, im_idx, dataset='MNIST'):
    if dataset == 'CIFAR10':
        return is_classified_correctly_cifar10(net_path, im_idx)
    else:
        im, label = get_image_with_label(im_idx)
        im = im.reshape(1,784,1).astype(np.float32)
        session = ort.InferenceSession(net_path)
        input_name = session.get_inputs()[0].name
        output = session.run(None, {input_name: im})
        predicted_class = np.argmax(output[0][0])

        return predicted_class == label

def is_classified_correctly_cifar10(net_path, im_idx):
    mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
    std = np.array([0.2023, 0.1994, 0.2010], dtype=np.float32)
    mean = mean.reshape(1,-1,1,1)
    std = std.reshape(1,-1,1,1)
    im, label = get_image_with_label(im_idx)
    inp = np.expand_dims(im, axis=0)
    inp = (inp-mean)/std
    session = ort.InferenceSession(net_path)
    input_name = session.get_inputs()[0].name
    out = session.run(None, {input_name: inp})[0]  # shape [1,10]
    logits = torch.from_numpy(out)
    probs = softmax(logits)
    _, pred = torch.max(probs, dim=1)  # Get predicted class index
    return (pred.item() == label)

def select_idxs_net_oracle(indexes_vs_oracles, net_path, dataset='MNIST'):
    selected_idxs = []
    session = ort.InferenceSession(net_path)
    input_name = session.get_inputs()[0].name
    for idx in indexes_vs_oracles.keys():
        oracle_labels = indexes_vs_oracles[idx]
        im, lb = get_image_with_label(idx)
        if dataset == 'CIFAR10':
            im = np.expand_dims(im, axis=0)
        input_name = session.get_inputs()[0].name
        output = session.run(None, {input_name: im})
        predicted_class = np.argmax(output[0][0])
        print(f"{predicted_class},{idx}: {indexes_vs_oracles[idx]}")
        if predicted_class in oracle_labels:
            selected_idxs.append(idx)

    return selected_idxs




def setup_on_standard(netnames, im_idxs_file, dataset, timeout, epsilons, target_benchmarks_dir, vnncomp_benchmarks_dir, is_clean_old):
    print(dataset, timeout, epsilons, target_benchmarks_dir, vnncomp_benchmarks_dir)
    instances_file = os.path.join(target_benchmarks_dir, 'instances.csv')
    if is_clean_old:
        clean_directory(target_benchmarks_dir)
        if os.path.isfile(instances_file):
            os.remove(instances_file)
    
    os.makedirs(target_benchmarks_dir, exist_ok=True)

    net_path = os.path.join(target_benchmarks_dir, 'onnx')
    prp_path = os.path.join(target_benchmarks_dir, 'vnnlib')
    os.makedirs(net_path, exist_ok=True)
    os.makedirs(prp_path, exist_ok=True)

    for net in netnames:
        if dataset == 'CIFAR10':
            orig_net_path = os.path.join(vnncomp_benchmarks_dir, 'nets', net)
        else:
            orig_net_path = os.path.join(vnncomp_benchmarks_dir, 'onnx', net)
        target_net_path = os.path.join(net_path, net)
        shutil.copy(orig_net_path, target_net_path)

    images, labels, indexes = select_images_with_labels(dataset_idxs_file=im_idxs_file, max_num_indexs=10000)
    labels = labels.reshape(-1)
    
    gen_props_standard(prp_path, images, labels, indexes, epsilons, dataset=dataset)

    instance_lines = []
    for net in netnames:
        net_path1 = os.path.join(net_path, net)
        for idx in indexes:
            if is_classified_correctly(net_path1, idx, dataset=dataset):
                for ep in epsilons:
                    prp_name = f"prop_{idx}_{ep}.vnnlib"
                    prp_path1 = os.path.join(prp_path, prp_name)
                    ins_line = f"{net_path1},{prp_path1},{timeout}\n"
                    instance_lines.append(ins_line)
            else:
                # print(f"Something is wrong....")
                pass

    with open(instances_file, 'a') as f:
        f.writelines(instance_lines)

def setup_on_oracle_guided_prop(netnames, idx_with_oracles_labels, dataset, timeout, epsilons, target_benchmarks_dir, vnncomp_benchmarks_dir, is_clean_old):
    print(dataset, timeout, epsilons, target_benchmarks_dir, vnncomp_benchmarks_dir)
    instances_file = os.path.join(target_benchmarks_dir, 'instances.csv')
    if is_clean_old:
        clean_directory(target_benchmarks_dir)
        if os.path.isfile(instances_file):
            os.remove(instances_file)

    os.makedirs(target_benchmarks_dir, exist_ok=True)

    net_path = os.path.join(target_benchmarks_dir, 'onnx')
    prp_path = os.path.join(target_benchmarks_dir, 'vnnlib')
    os.makedirs(net_path, exist_ok=True)
    os.makedirs(prp_path, exist_ok=True)

    indexes_vs_labels = {}
    with open(idx_with_oracles_labels, 'r') as f:
        Lines = f.readlines()
        for line in Lines:
            line = line.strip()
            line_l = line.split(',')
            idx = int(line_l[0])
            labels = [int(val) for val in line_l[1:]]
            indexes_vs_labels[idx] = labels[:2]

    instance_lines = []
    for net in netnames:
        if dataset == 'MNIST':
            orig_net_path = os.path.join(vnncomp_benchmarks_dir, 'onnx', net)
        else:
            orig_net_path = os.path.join(vnncomp_benchmarks_dir, 'nets', net)
        indexes = select_idxs_net_oracle(indexes_vs_oracles=indexes_vs_labels, net_path=orig_net_path, dataset=dataset)
        indexes = [idx for idx in indexes if len(indexes_vs_labels[idx]) != 1]
        print(f"{netnames} : num of selected images: {len(indexes)}")
        for idx in indexes:
            oracle_labels = indexes_vs_labels[idx]
            if len(oracle_labels) != 1:
                im,lb = get_image_with_label(idx)
                netname = f"{net[:-5]}_{idx}.onnx"
                target_net_path = os.path.join(net_path, netname)
                update_fc_relu_oracle(orig_net_path, target_net_path, oracle_labels=oracle_labels, existing_model_out_dims=10)
                for ep in epsilons:
                    prp_name = f"prop_{idx}_{ep}.vnnlib"
                    prp_path1 = os.path.join(prp_path, prp_name)
                    gen_single_props_oracle_guided(spec_path=prp_path1, im=im, label=lb, ep=ep, dataset=dataset)
                    ins_line = f"{target_net_path},{prp_path1},{timeout}\n"
                    instance_lines.append(ins_line)

            # if len(oracle_labels) == 1:
            #     shutil.copy(orig_net_path, target_net_path)
            # else:
            #     update_fc_relu_oracle(orig_net_path, target_net_path, oracle_labels=oracle_labels, existing_model_out_dims=10)
            # for ep in epsilons:
            #     prp_name = f"prop_{idx}_{ep}.vnnlib"
            #     prp_path1 = os.path.join(prp_path, prp_name)
            #     if len(oracle_labels) == 1:
            #         gen_single_props_standard(spec_path=prp_path1, im=im, label=lb, ep=ep)
            #     else:
            #         gen_single_props_oracle_guided(spec_path=prp_path1, im=im, label=lb, ep=ep)
                
            #     ins_line = f"{target_net_path},{prp_path1},{timeout}\n"
            #     instance_lines.append(ins_line)

    with open(instances_file, 'a') as f:
        f.writelines(instance_lines)


if __name__ == '__main__':
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
    target_benchmarks_dir = config.get('target_benchmarks_dir', None)
    oracle_labels_file = config.get('oracle_labels', None)
    property_type = config.get('property', None)
    is_cnn = config.get('is_cnn', None)
    log_dir = config.get('log_dir', None)
    is_less_than_output_prp = config.get('is_less_than_output_prp', False)
    is_target_prop = config.get('is_target_prop', False)
    sub_property = config.get('sub_property', None) 
    orig_image_conf_th = config.get('orig_image_th', 0) 
    conf_file = config.get('conf_file', None)
    is_clean_old = config.get('is_clean_old_benchmarks', True)

    vnncomp_benchmarks_dir, target_benchmarks_dir, log_dir = get_final_dirs(sub_property, dataset, vnncomp_benchmarks_dir, target_benchmarks_dir, log_dir)
    
    try:
        shutil.rmtree(target_benchmarks_dir)
        print(f"Directory '{target_benchmarks_dir}' and its contents were removed successfully.")
    except FileNotFoundError:
        print(f"Directory '{target_benchmarks_dir}' does not exist.")
    except Exception as e:
        print(f"Error: {e}")



    set_images_labels(dataset=dataset, is_test_data=is_test_data)

    if property_type == 'fp':
        setup_on_oracle_guided_prop(nets, oracle_labels_file, dataset, timeout, epsilons, target_benchmarks_dir, vnncomp_benchmarks_dir, is_clean_old)
    else:
        setup_on_standard(nets, dataset_idxs_file, dataset, timeout, epsilons, target_benchmarks_dir, vnncomp_benchmarks_dir, is_clean_old)
    
