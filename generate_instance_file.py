import csv
import os


file_path = '/home/afzal/Documents/tools/networks/conf_final/max_e/softmax/instances.csv'
if os.path.isfile(file_path):
    os.remove(file_path)
    
timeout = 2000
def write_to_file(net_path, prop_path):
    with open(file_path, 'a') as f:
        line = f"{net_path},{prop_path},{timeout}\n"
        f.write(line)



net_dir = '/home/afzal/Documents/tools/networks/conf_final/max_e/softmax/nets'
prop_dir = '/home/afzal/Documents/tools/networks/conf_final/max_e/softmax/props'

nets = ['mnist_relu_3_50.onnx', 'mnist_relu_3_100.onnx', 'mnist_relu_5_100.onnx', 'mnist_relu_6_100.onnx']
nets += ['mnist_relu_6_200.onnx', 'mnist_relu_9_100.onnx', 'mnist_relu_9_200.onnx']

nets = []
# nets = ['mnist_relu_3_50.onnx']
# nets += ['mnist_relu_6_100.onnx']
nets += ['mnist_relu_9_200.onnx']

confs = [60, 80, 70, 90, 95]
confs = [60]
# confs = [40, 60, 80]
epsilons = []
# epsilons = [0.02, 0.025, 0.03]
epsilons += [0.035, 0.04, 0.045, 0.05]
# epsilons += ['0.05', '0.06', '0.065', '0.07']
# epsilons += ['0.075', '0.08']
# epsilons += ['0.085', '0.09']
# epsilons += ['0.1']

images = [i for i in range(2) if i != 0]


for conf in confs:
    for net in nets:
        for im in images:
            for ep in epsilons:
                net_name = f"{net[:-5]}_{conf}_{im}.onnx"
                net_path = os.path.join(net_dir, net_name)
                prop_name = f"prop_{im}_{ep}.vnnlib"
                prop_path = os.path.join(prop_dir, prop_name)
                write_to_file(net_path, prop_path)

# for net in nets:
#     net_path = os.path.join(net_dir, net)
#     for im in images:
#         prop_name = f"prop_{im}_{ep}.vnnlib"
#         prop_path = os.path.join(prop_dir, prop_name)
#         write_to_file(net_path, prop_path)
            




