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
import yaml
import shutil


def write_script_file(file_name, cmds):
    with open(file_name, 'w') as file:
        for cm in cmds:
            file.write(cm+"\n")
        file.close()




def get_tasks(instance_file, confs):
    tasks = []
    with open(instance_file, 'r') as f:
        Lines = f.readlines()
        for line in Lines:
            line = line.strip()
            line = line.split(',')
            for conf in confs:
                tasks.append([line[0], line[1], line[2], conf])

    return tasks



def print_cmnds_marabou(num_cpus, log_dir, tool_main, num_workers, instance_file, dataset, confs):
    tasks = get_tasks(instance_file, confs)
    benchmarks_dir, _ = os.path.split(instance_file)
    # random.shuffle(tasks)
    # tasks = get_tasks_mnistfc_modified()
    # print(tasks)
    num_tasks = len(tasks)
    print(f"Total number of task: {num_tasks}")

    if num_cpus >= num_tasks:
        load_per_cpu = [1]*num_tasks
    else:
        load_per_cpu = [0]*num_cpus
        for i in range(0,num_tasks):
            j = i % num_cpus
            load_per_cpu[j] += 1

    print("Load per cpu: {}".format(load_per_cpu))

    prev_load = 0
    for idx, load in enumerate(load_per_cpu):
        ld = tasks[prev_load:prev_load+load]
        prev_load += load
        cmds = []
        for l in ld:
            net_path = l[0]
            net_path = os.path.join(benchmarks_dir, net_path)
            prop_path = l[1]
            prop_path = os.path.join(benchmarks_dir, prop_path)
            timeout = int(l[2])
            conf = int(l[3])
            log_file = os.path.basename(net_path)[:-5]+"+"+os.path.basename(prop_path)[:-7]+"_"+str(conf)
            log_file = os.path.join(log_dir, log_file)
            # command = f"taskset --cpu-list {num_cores*idx}-{(num_cores*idx)+(num_cores -1)} timeout -k 2s {timeout+200} python {tool_main} --config {config_path} --device cpu --show_adv_example --onnx_path {net_path} --vnnlib_path {prop_path} --results_file {result_file} --timeout {timeout} >> {log_file}"
            command = f"timeout -k 2s {timeout} python {tool_main} {net_path} {prop_path} {conf} {num_workers}  >> {log_file}"
            cmds.append(command)
        file_name = os.path.join(log_dir, f"script_{idx}.sh")
        write_script_file(file_name, cmds)



if __name__ == '__main__':
    if len(sys.argv) == 2:
        config_file = sys.argv[1]
    else:
        print("Error: ")
        sys.exit(1)

    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    
    dataset = config['dataset']
    num_parallel = config.get('num_parallel', 3)
    num_workers = config.get('num_workers', 1)
    benchmarks_dir = config['benchmarks_dir']
    log_dir = config['log_dir']
    main_file = config['main_file']
    confs = config['confs']
    instances_file = os.path.join(benchmarks_dir, 'instances.csv')
    print(instances_file)
    try:
        shutil.rmtree(log_dir)
        print(f"Directory '{log_dir}' and its contents were removed successfully.")
    except FileNotFoundError:
        print(f"Directory '{log_dir}' does not exist.")
    except Exception as e:
        print(f"Error: {e}")

    os.makedirs(log_dir, exist_ok=True)

    print_cmnds_marabou(num_parallel, log_dir, tool_main=main_file, num_workers=num_workers, instance_file=instances_file, dataset=dataset, confs=confs)








