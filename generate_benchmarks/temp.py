import os
import numpy as np
import shutil

import sys
sys.path.append(os.getcwd())
print(sys.path)
from generate_benchmarks.generate_properties import create_input_bounds_tf, save_vnnlib_tf_standard

net_dir = '/home/u1411251/tools/networks/conf_final/mnist/eran_mod'
bench_dir = '/home/u1411251/temp'
bench_net_dir = os.path.join(bench_dir, 'onnx')
bench_prp_dir = os.path.join(bench_dir, 'vnnlib')
instanc_file = os.path.join(bench_dir, 'instances.csv')
os.makedirs(bench_net_dir, exist_ok=True)
os.makedirs(bench_prp_dir, exist_ok=True)
nets = ['mnist_relu_5_100.onnx', 'mnist_relu_6_100.onnx', 'mnist_relu_6_200.onnx']
timeout = 300
dataset_files = '/home/u1411251/tools/VeriNN/deep_refine/benchmarks/dataset/mnist/mnist_test.csv'
labels = []
images = []
with open(dataset_files, 'r') as f:
    Lines = f.readlines()
    for line in Lines:
        line = line.strip()
        line = eval(line)
        labels.append(line[0])
        im = np.array(line[1:])/255
        images.append(im)

eps = [0.03, 0.08, 0.1]


for net in nets:
    old_net_path = os.path.join(net_dir, net)
    new_net_path = os.path.join(bench_net_dir, net)
    shutil.copy(old_net_path, new_net_path)

for ep in eps:
    for i in range(100):
        prp_name = f"prop_{i}_{ep}.vnnlib"
        full_prp_path = os.path.join(bench_prp_dir, prp_name)
        lb,ub = create_input_bounds_tf(images[i], ep)
        save_vnnlib_tf_standard(lb, ub, labels[i], full_prp_path)

instance_lines = []
for net in nets:
    new_net_path = os.path.join(bench_net_dir, net)
    for ep in eps:
        for i in range(100):
            prp_name = f"prop_{i}_{ep}.vnnlib"
            full_prp_path = os.path.join(bench_prp_dir, prp_name)
            line = f"{new_net_path},{full_prp_path},{timeout}\n"
            instance_lines.append(line)


with open(instanc_file, "w") as file:
    file.writelines(instance_lines)



