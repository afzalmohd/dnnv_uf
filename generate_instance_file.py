import csv
import os
from simulate_network import read_images_from_dataset
from simulate_network import get_selected_images


file_path = '/home/afzal/tools/networks/mod_props/instances.csv'
if os.path.isfile(file_path):
    os.remove(file_path)
    
timeout = 2000
def write_to_file(net_path, prop_path):
    with open(file_path, 'a') as f:
        line = f"{net_path},{prop_path},{timeout}\n"
        f.write(line)



net_dir = '/home/afzal/tools/networks/conf_final/orig_dataset/nets'
orig_net_dir = '/home/afzal/tools/networks/conf_final/eran_mod'
prop_dir = '/home/afzal/tools/networks/mod_props/props'

nets = ['mnist_relu_3_50.onnx', 'mnist_relu_3_100.onnx', 'mnist_relu_5_100.onnx', 'mnist_relu_6_100.onnx']
nets += ['mnist_relu_6_200.onnx', 'mnist_relu_9_100.onnx', 'mnist_relu_9_200.onnx']

nets = ['mnist_relu_5_100.onnx']

input_net_paths = []
for net in nets:
    input_net_paths.append(os.path.join(orig_net_dir, net))

confs = [60, 80, 90, 95]
epsilons = [0.04, 0.06]



# for input_path in input_net_paths:
#     images, labels, idxs = get_selected_images()
#     net_name_orig = os.path.basename(input_path)
#     print(f"{net_name_orig} : {len(images)}")
#     for i in range(len(images)):
#         idx = idxs[i]
#         for conf in confs:
#             for ep in epsilons:
#                 net_name = f"{net_name_orig[:-5]}_{conf}_{idx}.onnx"
#                 net_path = os.path.join(net_dir, net_name)
#                 prop_name = f"prop_{idx}_{ep}.vnnlib"
#                 prop_path = os.path.join(prop_dir, prop_name)
#                 write_to_file(net_path, prop_path)
        

for input_path in input_net_paths:
    images, labels, idxs = get_selected_images()
    for i in range(len(images)):
        idx = idxs[i]
        for ep in epsilons:
            prop_name = f"prop_{idx}_{ep}.vnnlib"
            prop_path = os.path.join(prop_dir, prop_name)
            write_to_file(input_path, prop_path)


# for conf in confs:
#     for net in nets:
#         for im in images:
#             for ep in epsilons:
#                 net_name = f"{net[:-5]}_{conf}_{im}.onnx"
#                 net_path = os.path.join(net_dir, net_name)
#                 prop_name = f"prop_{im}_{ep}.vnnlib"
#                 prop_path = os.path.join(prop_dir, prop_name)
#                 write_to_file(net_path, prop_path)

# for net in nets:
#     net_path = os.path.join(net_dir, net)
#     for im in images:
#         prop_name = f"prop_{im}_{ep}.vnnlib"
#         prop_path = os.path.join(prop_dir, prop_name)
#         write_to_file(net_path, prop_path)
            




