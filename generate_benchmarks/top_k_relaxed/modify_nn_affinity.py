import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import numpy as np
import onnx
import math
import shutil
import re
from onnx import helper, shape_inference, TensorProto, numpy_helper
from generate_benchmarks.modify_onnx import get_output_affine_layers_weights
from generate_benchmarks.generate_properties import save_vnnlib_from_vnncomp
from generate_benchmarks.top_k_relaxed.modify_nn_top_k_relaxed import update_fc_relu_top_k_relaxed, get_top_k_preds

mnist_dataset = 'MNIST'
cifar10_dataset = 'CIFAR10'
cifar100_dataset='CIFAR100'
imagenet_dataset = 'IMAGENET'
tsr_dataset = 'TSR'

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

    else:
        os.makedirs(directory_path, exist_ok=True)

def get_output_dims(dataset):
    dims = 10
    if dataset == cifar100_dataset:
        dims = 100
    elif dataset == tsr_dataset:
        dims = 43
    elif dataset == imagenet_dataset:
        dims = 1000

    return dims

def get_label_vnncomp_prp(prp_file, is_less_than_output_prp=False, is_target_prop = False):
    # with open(prp_file, 'r') as file:
    #     first_line = file.readline().strip()
    #     # Extract the number after 'label: '
    #     if "property with label:" in first_line:
    #         label = first_line.split("label:")[-1].strip().rstrip('.')
    #         return int(label) 
    label = None
    with open(prp_file, 'r') as file:
        lines = file.readlines()  # Read all lines into a list
        last_lines = lines[-10:]
        last_content = ''.join(last_lines)
        matches = re.findall(r'Y_(\d+)', last_content)
        if is_less_than_output_prp:
            if is_target_prop:
                label = int(matches[-1])
            else:
                label = int(matches[0])
        else:
            if is_target_prop:
                label = int(matches[0])
            else:
                label = int(matches[-1])
            
        return label

def are_conds_false(net_prp, file_path, th):
    # print(net_prp)
    conf = 57
    try:
        with open(file_path, 'r') as f:
            Lines = f.readlines()
            for line in Lines:
                line = line.strip()
                line_l = line.split(',')
                if net_prp[0] == line_l[0] and net_prp[1] == line_l[1]:
                    conf1 = float(line_l[2])
                    conf1 = conf1*100
                    conf = conf1
                    # print(conf, th)
    except:
        pass
    
    # print(conf, th)
    cond1 = (conf + th) >= 100
    cond2 = (conf - th) <= 0
    
    # print(f"Check....{net_prp}")
    return cond1, cond2

def is_affinity_cond_satifiy(top_preds, grouped_classes, k):
    top_preds= top_preds[:k]
    for g_classes in grouped_classes:
        if top_preds[0] in g_classes and top_preds[1] in g_classes:
            return True

    print(top_preds, grouped_classes)  
    return False

def setup_on_vnncomp_prop_affinity(dataset, timeout, epsilons, target_benchmarks_dir, vnncomp_benchmarks_dir, tolerance_param, conf_file, grouped_classes):
    input_tolerance = 1e-4
    k=2
    print(dataset, timeout, epsilons, target_benchmarks_dir, vnncomp_benchmarks_dir, tolerance_param)
    clean_directory(target_benchmarks_dir)
    os.makedirs(target_benchmarks_dir, exist_ok=True)
    instances_file = os.path.join(target_benchmarks_dir, 'instances.csv')
    if os.path.isfile(instances_file):
        os.remove(instances_file)

    vnncomp_instance_file = os.path.join(vnncomp_benchmarks_dir, 'instances.csv')

    with open(vnncomp_instance_file, 'r') as vnncomp_instance_f:
        output_dims = get_output_dims(dataset)
        instance_lines = []
        idx=0
        new_out_dims = 1
        count = 0
        for line in vnncomp_instance_f:
            line = line.strip()
            line_l = line.split(',')
            timeout = float(line_l[2])
            vnncomp_net_path = os.path.join(vnncomp_benchmarks_dir,  line_l[0])
            vnncomp_prp_path = os.path.join(vnncomp_benchmarks_dir, line_l[1])
            sub_net_dir, netname = os.path.split(line_l[0])
            sub_prp_dir, prpname = os.path.split(line_l[1])
            target_net_dir = os.path.join(target_benchmarks_dir, sub_net_dir)
            target_prp_dir = os.path.join(target_benchmarks_dir, sub_prp_dir)
            os.makedirs(target_net_dir, exist_ok=True)
            os.makedirs(target_prp_dir, exist_ok=True)
            # label = get_label_vnncomp_prp(vnncomp_prp_path, is_less_than_output_prp, is_target_prop=is_target_prop)
            top_pred_label = get_top_k_preds(line_l, conf_file)
            is_affinity_cond_sat = is_affinity_cond_satifiy(top_pred_label, grouped_classes, k=k)
            if not is_affinity_cond_sat:
                target_net_path = os.path.join(target_net_dir, netname)
                target_prp_path = os.path.join(target_prp_dir, prpname)
                shutil.copy2(vnncomp_net_path, target_net_path)
                shutil.copy2(vnncomp_prp_path, target_prp_path)
                print(f"Affinity cond failed: {netname} , {prpname}")
                ins_line = f"{os.path.join(sub_net_dir, netname)},{os.path.join(sub_prp_dir, prpname)},{timeout}\n"
                instance_lines.append(ins_line)
                count += 1
            else:
                new_net_name = f"{netname[:-5]}_{idx}_1.onnx"
                new_prp_name =  f"{prpname[:-7]}_{idx}_1.vnnlib"
                target_net_path = os.path.join(target_net_dir, new_net_name)
                target_prp_path = os.path.join(target_prp_dir, new_prp_name)
                update_fc_relu_top_k_relaxed(model_path=vnncomp_net_path, output_model_path=target_net_path, top_k_labels=top_pred_label, existing_model_out_dims=output_dims, k=k, input_error=0.001)
                save_vnnlib_from_vnncomp(vnncomp_prp_path, target_prp_path, conf=1, total_output_class=new_out_dims, tolerance_param=tolerance_param, orignal_out_classes=output_dims)
                print(f"Affinity cond hold: {new_net_name} , {new_prp_name}")
                ins_line = f"{os.path.join(sub_net_dir, new_net_name)},{os.path.join(sub_prp_dir, new_prp_name)},{timeout}\n"
                instance_lines.append(ins_line)
            idx += 1
        with open(instances_file, 'w') as f:
            f.writelines(instance_lines)

        print(f"Mismatched: {count}")





if __name__ == '__main__':
    input_model_path = '/home/u1411251/tools/vnncomp_benchmarks/mnist_fc/onnx/mnist-net_256x4.onnx'
    output_model_path = "temp.onnx"
    # update_fc_relu_smooth_both_conds(input_model_path, output_model_path, label = 0, delta_1 = 1.98, delta_2 = 2.02, existing_model_out_dims = 10, input_tolerance = 1e-5)
    # update_fc_relu_smooth_cond1(input_model_path, output_model_path, label = 0, delta=1.98, existing_model_out_dims = 10)
    # get_delta_strong(15, 10)
    # get_fc1_layer_weights_smooth(label=3, output_dims=10)