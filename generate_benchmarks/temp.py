import os
import numpy as np
import shutil

import sys
sys.path.append(os.getcwd())
print(sys.path)
from generate_benchmarks.generate_properties import create_input_bounds_tf, save_vnnlib_tf_standard
from generate_benchmarks.simulate_network import run_model

net_dir = '/home/u1411251/tools/networks/conf_final/mnist/eran_mod'
bench_dir = '/home/u1411251/temp'

try:
    shutil.rmtree(bench_dir)
    print(f"Directory '{bench_dir}' and its contents were removed successfully.")
except FileNotFoundError:
    print(f"Directory '{bench_dir}' does not exist.")
except Exception as e:
    print(f"Error: {e}")

bench_net_dir = os.path.join(bench_dir, 'onnx')
bench_prp_dir = os.path.join(bench_dir, 'vnnlib')
instanc_file = os.path.join(bench_dir, 'instances.csv')
os.makedirs(bench_net_dir, exist_ok=True)
os.makedirs(bench_prp_dir, exist_ok=True)
nets = ['mnist_relu_3_50.onnx', 'mnist_relu_3_100.onnx', 'mnist_relu_4_1024.onnx', 'mnist_relu_5_100.onnx', 'mnist_relu_6_100.onnx', 'mnist_relu_6_200.onnx']
nets += ['mnist_relu_9_100.onnx', 'mnist_relu_9_200.onnx', 'ffnnRELU__Point_6_500.onnx', 'ffnnRELU__PGDK_w_0.1_6_500.onnx', 'ffnnRELU__PGDK_w_0.3_6_500.onnx']
timeout = 300
dataset_files = '/home/u1411251/tools/VeriNN/deep_refine/benchmarks/dataset/mnist/mnist_test.csv'
labels = []
images = []
with open(dataset_files, 'r') as f:
    Lines = f.readlines()
    for line in Lines:
        line = line.strip()
        line = eval(line)
        labels.append(int(line[0]))
        im = np.array(line[1:])/255
        images.append(im)

eps = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05]


instance_lines = []
for net in nets:
    old_net_path = os.path.join(net_dir, net)
    new_net_path = os.path.join(bench_net_dir, net)
    shutil.copy(old_net_path, new_net_path)
    netname = os.path.splitext(os.path.basename(new_net_path))[0]
    bench_prp_dir1 = os.path.join(bench_prp_dir, netname)
    os.makedirs(bench_prp_dir1, exist_ok=True)
    for i in range(100):
        pred_label, _ = run_model(old_net_path, images[i])
        pred_label = int(pred_label)
        if labels[i] == pred_label:
            for ep in eps:
                prp_name = f"prop_{i}_{ep}.vnnlib"
                full_prp_path = os.path.join(bench_prp_dir1, prp_name)
                lb,ub = create_input_bounds_tf(images[i], ep)
                save_vnnlib_tf_standard(lb, ub, labels[i], full_prp_path)

                instance_net_path = os.path.join("onnx", net)
                instance_prp_path = os.path.join("vnnlib", netname, prp_name)
                line = f"{instance_net_path},{instance_prp_path},{timeout}\n"
                instance_lines.append(line)

# instance_lines = []
# for net in nets:
#     new_net_path = os.path.join(bench_net_dir, net)
#     for ep in eps:
#         for i in range(100):
#             prp_name = f"prop_{i}_{ep}.vnnlib"
#             full_prp_path = os.path.join(bench_prp_dir, prp_name)
#             line = f"{new_net_path},{full_prp_path},{timeout}\n"
#             instance_lines.append(line)


with open(instanc_file, "w") as file:
    file.writelines(instance_lines)



