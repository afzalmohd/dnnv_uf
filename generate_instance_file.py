import csv
import os


file_path = '/home/afzal/Documents/tools/networks/conf_final/instances.csv'
timeout = 2000
def write_to_file(net_path, prop_path):
    with open(file_path, 'a') as f:
        line = f"{net_path},{prop_path},{timeout}\n"
        f.write(line)



net_dir = '/home/afzal/Documents/tools/networks/conf_final/eran_mod_conf'
prop_dir = '/home/afzal/Documents/tools/networks/conf_final/props'

nets = ['mnist_relu_3_50.onnx', 'mnist_relu_3_100.onnx', 'mnist_relu_5_100.onnx', 'mnist_relu_6_100.onnx']
nets += ['mnist_relu_6_200.onnx', 'mnist_relu_9_100.onnx', 'mnist_relu_9_200.onnx']



confs = [60, 80, 90, 95]

ep = '0.06'

images = [i for i in range(21) if i != 8]

tasks = []

for conf in confs:
    for net in nets:
        for im in images:
            net_name = f"{net[:-5]}_{conf}_{im}.onnx"
            net_path = os.path.join(net_dir, net_name)
            prop_name = f"prop_{im}_{ep}.vnnlib"
            prop_path = os.path.join(prop_dir, prop_name)
            write_to_file(net_path, prop_path)
            




