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




def write_script_file(file_name, cmds):
    with open(file_name, 'w') as file:
        for cm in cmds:
            file.write(cm+"\n")
        file.close()


def get_tasks(instance_file):
    tasks = []
    with open(instance_file, 'r') as f:
        Lines = f.readlines()
        for line in Lines:
            line = line.strip()
            line = line.split(',')
            tasks.append([line[0], line[1], line[2]])

    return tasks


def print_cmnds_marabou_terminal(num_cpu, log_dir, tool_main, num_cores, instance_file):
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
            log_file = os.path.basename(net_path)[:-5]+"+"+os.path.basename(prop_path)[:-7]
            log_file = os.path.join(log_dir, log_file)
            command = f"taskset --cpu-list {num_cores*idx}-{(num_cores*idx)+(num_cores -1)} timeout -k 2s {timeout+200} python {tool_main} --config {config_path} --device cpu --show_adv_example --onnx_path {net_path} --vnnlib_path {prop_path} --results_file {result_file} --timeout {timeout} >> {log_file}"
            # command = f"timeout -k 2s {timeout+200} python {tool_main} --config {config_path} --device cpu --show_adv_example --onnx_path {net_path} --vnnlib_path {prop_path} --results_file {result_file} --timeout {timeout} >> {log_file}"
            cmds.append(command)
        file_name = os.path.join(log_dir, f"script_{idx}.sh")
        write_script_file(file_name, cmds)

def print_cmnds_marabou(log_dir, tool_main, target_benchmarks_dir, prp_type='standard', start_idx=-1, end_idx=-1):
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
        log_file = os.path.join(log_dir, log_file)
        command = [
                "timeout", "-k", "2s", str(timeout + 20), "python", tool_main, net_path, prop_path, prp_type
            ]
        
        print(command)
        with open(log_file, "a") as log:
            try:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Command failed for {log_file}: {e}")


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
    
    tool_main = config['marabou_tool']
    target_benchmarks_dir = config['target_benchmarks_dir']
    log_dir = config['log_dir']
    property_type = 'standard'
    is_clean_old = False

    if property_type == 'fp':
        target_benchmarks_dir = os.path.join(target_benchmarks_dir, 'fp')
        log_dir = os.path.join(log_dir, 'fp')
    else:
        target_benchmarks_dir = os.path.join(target_benchmarks_dir, 'standard')
        log_dir = os.path.join(log_dir, 'standard')

    if is_clean_old:
        try:
            shutil.rmtree(log_dir)
            print(f"Directory '{log_dir}' and its contents were removed successfully.")
        except FileNotFoundError:
            print(f"Directory '{log_dir}' does not exist.")
        except Exception as e:
            print(f"Error: {e}")

    os.makedirs(log_dir, exist_ok=True)

    print_cmnds_marabou(log_dir, tool_main=tool_main, target_benchmarks_dir=target_benchmarks_dir, prp_type=property_type, start_idx=start_idx, end_idx=end_idx)





