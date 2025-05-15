#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 09:09:35 2021

@author: u1411251
"""

from multiprocessing import Pool
import random
import os
import sys
import subprocess
import shlex
import yaml
import shutil
import torch

mnist_dataset = 'MNIST'
cifar10_dataset = 'CIFAR10'
cifar100_dataset='CIFAR100'
imagenet_dataset = 'IMAGENET'
tsr_dataset = 'TSR'

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'


NUM_CPU = 7
TIMEOUT = 2000 #Not relevant
DATASET = "MNIST"
NUM_IMAGES = 100


root_dir = os.getcwd()
TOOL = os.path.join(root_dir, 'mar_encod.py')
TOOL = 'mar_encod.py'
result_dir = os.path.join(root_dir, 'outfiles')
dataset_file = '/home/afzal/tools/VeriNN/deep_refine/benchmarks/dataset/mnist/mnist_test.csv'
# result_file = os.path.join(result_dir, "result.txt")

# if not os.path.isdir(result_dir):
#     os.mkdir(result_dir)


def get_final_dirs(sub_property, dataset, vnncomp_dir, benchmarks_dir, log_dir): 
    if dataset == imagenet_dataset:
        new_vnncomp_dir = os.path.join(vnncomp_dir, 'imagenet', 'vggnet16')
        new_benchmarks_dir = os.path.join(benchmarks_dir, sub_property, 'imagenet', 'vggnet16')
        new_log_dir = os.path.join(log_dir, sub_property, 'imagenet', 'vggnet16')
    elif dataset == mnist_dataset:
        new_vnncomp_dir = os.path.join(vnncomp_dir, 'mnist_fc')
        new_benchmarks_dir = os.path.join(benchmarks_dir, sub_property, 'mnist_fc')
        new_log_dir = os.path.join(log_dir, sub_property, 'mnist_fc')
    elif dataset == cifar100_dataset:
        new_vnncomp_dir = os.path.join(vnncomp_dir, 'cifar100' , 'cifar100_tinyimagenet_resnet')
        new_benchmarks_dir = os.path.join(benchmarks_dir, sub_property, 'cifar100', 'cifar100_tinyimagenet_resnet')
        new_log_dir = os.path.join(log_dir, sub_property, 'cifar100', 'cifar100_tinyimagenet_resnet')
    elif dataset == cifar10_dataset:
        new_vnncomp_dir = vnncomp_dir
        new_benchmarks_dir = benchmarks_dir
        new_log_dir = log_dir

    return new_vnncomp_dir, new_benchmarks_dir, new_log_dir


def write_script_file(file_name, cmds):
    with open(file_name, 'w') as file:
        for cm in cmds:
            file.write(cm+"\n")
        file.close()



    
def get_all_tasks():
    tasks = []
    # net_dir = '/home/afzal/Documents/tools/networks/tf/mnist'
    NETWORK_FILE = []
    NETWORK_FILE += ["temp.nnet"]
    # NETWORK_FILE += ["mnist_relu_3_100.tf"]
    # NETWORK_FILE += ["mnist_relu_5_100.tf"]
    # NETWORK_FILE += ["mnist_relu_6_100.tf", "mnist_relu_6_200.tf"]
    # # NETWORK_FILE += ["mnist_relu_4_1024.tf"]
    # NETWORK_FILE += ["mnist_relu_9_100.tf"]
    # NETWORK_FILE += ["mnist_relu_9_200.tf"]
    # NETWORK_FILE += ["ffnnRELU__Point_6_500.tf", "ffnnRELU__PGDK_w_0.1_6_500.tf", "ffnnRELU__PGDK_w_0.3_6_500.tf"]
    
    images_list = [i for i in range(NUM_IMAGES)]
    epsilons = [0.06]
    confidences =[0, 60]

    for image_index in images_list:
        for ep in epsilons:
            for nt in NETWORK_FILE:
                for conf in confidences:
                    if conf == 0:
                        is_conf_robust = 0
                    else:
                        is_conf_robust = 1

                    tasks.append([nt, image_index, ep, conf, is_conf_robust])
    # tasks=get_diff_tasks()
    # print(tasks)
    return tasks




# def print_cmnds_all(num_cpu, log_dir):
#     tasks = get_all_tasks()
#     # tasks = get_fixed_tasks()
#     # tasks = get_task_from_file_random()
#     net_dir = '/home/afzal/tools/networks/tf/mnist'
#     # random.shuffle(tasks)

#     num_tasks = len(tasks)
#     print(f"Total number of task: {num_tasks}")

#     if num_cpu >= num_tasks:
#         load_per_cpu = [1]*num_tasks
#     else:
#         load_per_cpu = [0]*num_cpu
#         for i in range(0,num_tasks):
#             j = i % num_cpu
#             load_per_cpu[j] += 1

#     print("Load per cpu: {}".format(load_per_cpu))

#     prev_load = 0
#     for idx, load in enumerate(load_per_cpu):
#         ld = tasks[prev_load:prev_load+load]
#         prev_load += load
#         cmds = []
#         for l in ld:
#             net_name = l[0]
#             image_index = l[1]
#             ep = l[2]
#             conf = l[3]
#             is_conf_rb = l[4]
#             log_file = net_name+"+"+str(image_index)+"+"+str(ep)+"+"+str(conf)
#             log_file = os.path.join(log_dir, log_file)
#             result_file = os.path.join(result_dir, f"file_{idx}.txt")
#             command = f"taskset --cpu-list {num_cores*idx}-{(num_cores*idx)+(num_cores -1)} timeout -k 2s {TIMEOUT} python {TOOL} {net_name} {image_index} {ep} {conf} {is_conf_rb} {result_file} >> {log_file}"
#             cmds.append(command)
#         file_name = os.path.join(log_dir, f"script_{idx}.sh")
#         write_script_file(file_name, cmds)


def get_tasks(instance_file):
    tasks = []
    with open(instance_file, 'r') as f:
        Lines = f.readlines()
        for line in Lines:
            line = line.strip()
            line = line.split(',')
            tasks.append([line[0], line[1], line[2]])

    return tasks

def get_tasks_mnistfc_modified():
    instance_file = '/home/afzal/tools/networks/conf_final/instances.csv'
    tasks = []
    with open(instance_file, 'r') as f:
        Lines = f.readlines()
        for line in Lines:
            line = line.strip()
            line = line.split(',')
            tasks.append([line[0], line[1], line[2]])

    return tasks

def get_confg_path_cifar100(net_path, existing_config_path):
    config_path1 , _  = os.path.split(existing_config_path)
    config_file = ''
    if 'CIFAR100_resnet_small' in net_path:
        config_file = 'cifar100_small_2022.yaml'
    elif 'CIFAR100_resnet_medium' in net_path:
        config_file = 'cifar100_med_2022.yaml'
    elif 'CIFAR100_resnet_large' in net_path:
        config_file = 'cifar100_large_2022.yaml'
    elif 'CIFAR100_resnet_super' in net_path:
        config_file = 'cifar100_super_2022.yaml'
    elif 'TinyImageNet_resnet_medium' in net_path:
        config_file = 'tinyimagenet_2022.yaml'

    return os.path.join(config_path1, config_file)


def print_cmnds_abcrowns_terminal(num_cpu, log_dir, tool_main, config_path, num_cores, instance_file, dataset):
    instance_file = os.path.join(target_benchmarks_dir, 'instances.csv')
    tasks = get_tasks(instance_file=instance_file)
    # random.shuffle(tasks)
    # tasks = get_tasks_mnistfc_modified()
    # print(tasks)
    num_tasks = len(tasks)
    print(f"Total number of task: {num_tasks}")

    if num_cpu >= num_tasks:
        load_per_cpu = [1]*num_tasks
    else:
        load_per_cpu = [0]*num_cpu
        for i in range(0,num_tasks):
            j = i % num_cpu
            load_per_cpu[j] += 1

    print("Load per cpu: {}".format(load_per_cpu))

    prev_load = 0
    for idx, load in enumerate(load_per_cpu):
        ld = tasks[prev_load:prev_load+load]
        prev_load += load
        cmds = []
        for l in ld:
            net_path = l[0]
            prop_path = l[1]
            timeout = int(l[2])
            if dataset == 'CIFAR100':
                config_path = get_confg_path_cifar100(net_path)
            log_file = os.path.basename(net_path)[:-5]+"+"+os.path.basename(prop_path)[:-7]
            log_file = os.path.join(log_dir, log_file)
            result_file = "res_"+os.path.basename(net_path)[:-5]+"+"+os.path.basename(prop_path)[:-7]
            result_file = os.path.join(log_dir, result_file)
            # command = f"taskset --cpu-list {num_cores*idx}-{(num_cores*idx)+(num_cores -1)} timeout -k 2s {timeout+200} python {tool_main} --config {config_path} --device cpu --show_adv_example --onnx_path {net_path} --vnnlib_path {prop_path} --results_file {result_file} --timeout {timeout} >> {log_file}"
            command = f"timeout -k 2s {timeout+200} python {tool_main} --config {config_path} --device cpu --show_adv_example --onnx_path {net_path} --vnnlib_path {prop_path} --results_file {result_file} --timeout {timeout} >> {log_file}"
            cmds.append(command)
        file_name = os.path.join(log_dir, f"script_{idx}.sh")
        write_script_file(file_name, cmds)


def print_cmnds_abcrowns_old(log_dir, tool_main, config_path, dataset, target_benchmarks_dir, device, num_cpu=1):
    instance_file = os.path.join(target_benchmarks_dir, 'instances.csv')
    tasks = get_tasks(instance_file=instance_file)
    random.shuffle(tasks)
    # tasks = get_tasks_mnistfc_modified()
    # print(tasks)
    num_tasks = len(tasks)
    print(f"Total number of task: {num_tasks}")

    if num_cpu >= num_tasks:
        load_per_cpu = [1]*num_tasks
    else:
        load_per_cpu = [0]*num_cpu
        for i in range(0,num_tasks):
            j = i % num_cpu
            load_per_cpu[j] += 1

    print("Load per cpu: {}".format(load_per_cpu))

    prev_load = 0
    for idx, load in enumerate(load_per_cpu):
        ld = tasks[prev_load:prev_load+load]
        prev_load += load   
        cmds = []
        for l in ld:
            net_path = os.path.join(target_benchmarks_dir, l[0])
            prop_path = os.path.join(target_benchmarks_dir, l[1])
            timeout = float(l[2])
            if dataset == 'CIFAR100':
                config_path = get_confg_path_cifar100(net_path, config_path)
            log_file = os.path.basename(net_path)[:-5]+"+"+os.path.basename(prop_path)[:-7]
            im_log_file = f"im_{log_file}"
            log_file = os.path.join(log_dir, log_file)
            os.environ['im_log_file'] = os.path.join(log_dir, im_log_file)
            # result_file = "res_"+os.path.basename(net_path)[:-5]+"+"+os.path.basename(prop_path)[:-7]
            # result_file = os.path.join(log_dir, result_file)
            command = [
                "timeout", "-k", "2s", str(timeout + 200), "python", tool_main,
                "--config", config_path,
                "--device", device,
                "--show_adv_example",
                "--onnx_path", net_path,
                "--vnnlib_path", prop_path,
                # "--results_file", result_file,
                "--timeout", str(timeout)
            ]
            print(command)
            # with open(log_file, "a") as log:
            #     try:
            #         subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            #     except subprocess.CalledProcessError as e:
            #         print(f"Command failed for {log_file}: {e}")
        # file_name = os.path.join(log_dir, f"script_{idx}.sh")
        # write_script_file(file_name, cmds)

# def print_cmnds_marabou(log_dir, tool_main, target_benchmarks_dir, prp_type='standard', start_idx=-1, end_idx=-1):
#     instance_file = os.path.join(target_benchmarks_dir, 'instances.csv')
#     tasks = get_tasks(instance_file=instance_file)
#     if start_idx != -1 and end_idx != -1:
#         tasks = tasks[start_idx:end_idx]
#     # print(tasks)
#     num_tasks = len(tasks)
#     print(f"Total number of task: {num_tasks}")

#     for ts in tasks:
#         net_path = os.path.join(target_benchmarks_dir, ts[0])
#         prop_path = os.path.join(target_benchmarks_dir, ts[1])
#         timeout = float(ts[2])
#         log_file = os.path.basename(net_path)[:-5]+"+"+os.path.basename(prop_path)[:-7]
#         log_file = os.path.join(log_dir, log_file)
#         command = [
#                 "timeout", "-k", "2s", str(timeout + 20), "python", tool_main, net_path, prop_path, prp_type
#             ]
        
#         print(command)
#         with open(log_file, "a") as log:
#             try:
#                 subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
#             except subprocess.CalledProcessError as e:
#                 print(f"Command failed for {log_file}: {e}")

def print_cmnds_abcrown(log_dir, tool_main, target_benchmarks_dir, config_path, device, start_idx=-1, end_idx=-1):
    instance_file = os.path.join(target_benchmarks_dir, 'instances.csv')
    tasks = get_tasks(instance_file=instance_file)
    if start_idx != -1 and end_idx != -1:
        tasks = tasks[start_idx:end_idx]
    # print(tasks)
    num_tasks = len(tasks)
    print(f"Total number of task: {num_tasks}")

    for ts in tasks:
        net_path = os.path.join(target_benchmarks_dir, ts[0])
        prop_path = os.path.join(target_benchmarks_dir, ts[1])
        timeout = float(ts[2])
        log_file = os.path.basename(net_path)[:-5]+"+"+os.path.basename(prop_path)[:-7]
        im_log_file = f"im_{log_file}"
        log_file = os.path.join(log_dir, log_file)
        os.environ['im_log_file'] = os.path.join(log_dir, im_log_file)
        command = [
                "timeout", "-k", "2s", str(timeout + 50), "python", tool_main,
                "--config", config_path,
                "--device", device,
                "--show_adv_example",
                "--onnx_path", net_path,
                "--vnnlib_path", prop_path,
                # "--results_file", result_file,
                "--timeout", str(timeout)
            ]
        
        print(command)
        with open(log_file, "a") as log:
            try:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Command failed for {log_file}: {e}")

def print_server_info():
    # Check PyTorch version
    print("PyTorch version:", torch.__version__)

    # Check if CUDA is available
    print("CUDA available:", torch.cuda.is_available())

    # Check the CUDA version
    print("CUDA version:", torch.version.cuda)

    # Check the GPU name
    if torch.cuda.is_available():
        print("GPU name:", torch.cuda.get_device_name(0))


if __name__ == '__main__':
    if len(sys.argv) == 4:
        config_file = sys.argv[1]
        start_idx = int(sys.argv[2])
        end_idx = int(sys.argv[3])
    else:
        print("Error: ")
        sys.exit(1)

    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    
    dataset = config['dataset']
    tool_main = config['abcrown_tool']
    property_type = config['property']
    sub_prp_type = config['sub_property']
    # num_cores_per_benchmarks = int(total_cores/num_cpu)
    config_path = config['abcrown_config']
    target_benchmarks_dir = config['target_benchmarks_dir']
    device = config.get('device', 'cpu')
    log_dir = config['log_dir']
    is_clean_old = config.get('is_clean_old_benchmarks', True)

    _, target_benchmarks_dir, log_dir = get_final_dirs(sub_prp_type, dataset, "", target_benchmarks_dir, log_dir)

    # if property_type == 'fp':
    #     target_benchmarks_dir = os.path.join(target_benchmarks_dir, 'fp')
    # else:
    #     target_benchmarks_dir = os.path.join(target_benchmarks_dir, 'standard')

    if is_clean_old:
        try:
            shutil.rmtree(log_dir)
            print(f"Directory '{log_dir}' and its contents were removed successfully.")
        except FileNotFoundError:
            print(f"Directory '{log_dir}' does not exist.")
        except Exception as e:
            print(f"Error: {e}")

    os.makedirs(log_dir, exist_ok=True)

    print_cmnds_abcrown(log_dir, tool_main=tool_main, target_benchmarks_dir=target_benchmarks_dir, config_path=config_path, device=device, start_idx=start_idx, end_idx=end_idx)




