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

mnist_dataset = 'MNIST'
cifar10_dataset = 'CIFAR10'
cifar100_dataset='CIFAR100'
imagenet_dataset = 'IMAGENET'
tsr_dataset = 'TSR'

def get_fc_layer_weights_strong(label, output_dims):
    weights = []
    for i in range(output_dims):
        if i != label:
            l = [0.0]*output_dims
            l[i] = 1.0
            l[label] = -1.0
            weights.append(l)
    
    return weights

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

def get_output_layers_w_b(graph, w_weight_name, b_weight_name):
    is_vgg_net = False
    for initializer in graph.initializer:
        # print(initializer.name)
        if w_weight_name in  initializer.name:
            if is_vgg_net:
                output_layer_weight = numpy_helper.to_array(initializer).astype(np.float32)
            else:
                output_layer_weight = np.frombuffer(initializer.raw_data, dtype=np.float32).reshape(initializer.dims)
            # output_layer_input_dim = initializer.dims[1]
        elif b_weight_name in  initializer.name:
            if is_vgg_net:
                output_layer_bias = numpy_helper.to_array(initializer).astype(np.float32)
            else:
                output_layer_bias = np.frombuffer(initializer.raw_data, dtype=np.float32)

    return output_layer_weight, output_layer_bias

def get_delta_strong(conf, output_dims):
    # delta = -ln((1/(dim - 1))*((100/conf) -1))
    temp = (100/conf) -1
    temp = temp/(output_dims-1)
    delta = -math.log(temp)
    # print(delta)
    return delta

    

def update_fc_relu_strong(model_path, output_model_path, label = 0, delta=1.98, existing_model_out_dims = 10):
    w_weight_name, b_weight_name  = get_output_affine_layers_weights(model_path)
    model = onnx.load(model_path)
    graph = model.graph
    output_layer_weight, output_layer_bias = get_output_layers_w_b(graph, w_weight_name, b_weight_name)
    fc_output_dim = existing_model_out_dims - 1
    final_out_dim = 1
        
    new_w = get_fc_layer_weights_strong(label, output_dims=existing_model_out_dims)
    new_fc_weight = np.reshape(new_w, (fc_output_dim, existing_model_out_dims))
    new_fc_weight = np.asarray(new_fc_weight, dtype=np.float32)
    new_fc_bias = np.array([delta]*fc_output_dim, dtype=np.float32)
    # Combine the weights and biases
    combined_weight = np.dot(new_fc_weight, output_layer_weight)
    combined_bias = np.dot(new_fc_weight, output_layer_bias) + new_fc_bias



    # Update the initializers in the graph
    for initializer in graph.initializer:
        if w_weight_name in  initializer.name:
            initializer.raw_data = combined_weight.tobytes()
            initializer.dims[:] = combined_weight.shape
        elif b_weight_name in  initializer.name:
            initializer.raw_data = combined_bias.tobytes()
            initializer.dims[:] = combined_bias.shape
    # Save the modified model

    prev_output_name = graph.output[0].name
    relu_output_name = 'appnded.relu'

    relu_node = helper.make_node('Relu', inputs=[str(prev_output_name)], outputs=[str(relu_output_name)], 
                                 name=str(relu_output_name)
                                 )

    output_fc_layer_name = 'appended_fc'

    weight_name = f"layer.appended.weight"
    bias_name = f"layer.appended.bias"

    fc_node = helper.make_node('Gemm', inputs=[str(relu_output_name), weight_name, bias_name], 
                               outputs=[str(output_fc_layer_name)], alpha=1.0, beta=1.0, transB=1,
                               name=str(output_fc_layer_name)
                               )

    weight = [1.0]*fc_output_dim
    fc_weight = helper.make_tensor(name=weight_name, data_type=TensorProto.FLOAT, dims=[final_out_dim, fc_output_dim],vals=weight)

    fc_bias = helper.make_tensor(name=bias_name, data_type=TensorProto.FLOAT, dims=[final_out_dim], vals=[0.0] * (final_out_dim))


    graph.node.append(relu_node)
    graph.node.append(fc_node)
    graph.initializer.append(fc_weight)
    graph.initializer.append(fc_bias)
    graph.output[0].name = str(output_fc_layer_name)

    for output in graph.output:
        # Assuming there is only one output tensor, if there are multiple, you may need to specify the exact one
        dim_value = output.type.tensor_type.shape.dim
        if len(dim_value) == 2:
            dim_value[1].dim_value = final_out_dim

    model = shape_inference.infer_shapes(model)
    # print(f"Network modified: {output_model_path}")
    onnx.save(model, output_model_path)

def is_orig_conf_satisfied(net_prp, th, file_path):
    try:
        with open(file_path, 'r') as f:
            Lines = f.readlines()
            for line in Lines:
                line = line.strip()
                line_l = line.split(',')
                if net_prp in line:
                    conf = float(line_l[-1])
                    conf = conf*100
                    return conf >= th
    except:
        return True
    
    # print(f"Check....{net_prp}")
    return True

def setup_on_vnncomp_prop_strong(dataset, confs, timeout, epsilons, target_benchmarks_dir, vnncomp_benchmarks_dir, tolerance_param, is_less_than_output_prp, conf_file, orig_image_conf_th, is_target_prop=False):
    print(dataset, confs, timeout, epsilons, target_benchmarks_dir, vnncomp_benchmarks_dir, tolerance_param)
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
            label = get_label_vnncomp_prp(vnncomp_prp_path, is_less_than_output_prp, is_target_prop=is_target_prop)
            # print(f"Label: {label}")
            is_orig_conf_sat =  is_orig_conf_satisfied(line, orig_image_conf_th, conf_file)
            for conf in confs:
                new_net_name = f"{netname[:-5]}_{idx}_{conf}.onnx"
                new_prp_name =  f"{prpname[:-7]}_{idx}_{conf}.vnnlib"
                target_net_path = os.path.join(target_net_dir, new_net_name)
                target_prp_path = os.path.join(target_prp_dir, new_prp_name)
                if conf == 0:
                    shutil.copy2(vnncomp_net_path, target_net_path)
                    shutil.copy2(vnncomp_prp_path, target_prp_path)
                    ins_line = f"{os.path.join(sub_net_dir, new_net_name)},{os.path.join(sub_prp_dir, new_prp_name)},{timeout}\n"
                    instance_lines.append(ins_line)
                else:
                    delta = get_delta_strong(conf, output_dims)
                    if is_orig_conf_sat:
                        print(f"Generated: {os.path.join(sub_net_dir, new_net_name)},{os.path.join(sub_prp_dir, new_prp_name)},{timeout}")
                        update_fc_relu_strong(model_path=vnncomp_net_path, output_model_path=target_net_path, label=label, delta=delta, existing_model_out_dims=output_dims)
                        save_vnnlib_from_vnncomp(vnncomp_prp_path, target_prp_path, conf=conf, total_output_class=new_out_dims, tolerance_param=tolerance_param, orignal_out_classes=output_dims)
                        ins_line = f"{os.path.join(sub_net_dir, new_net_name)},{os.path.join(sub_prp_dir, new_prp_name)},{timeout}\n"
                        instance_lines.append(ins_line)
                    else:
                        print(f"Orig image conf not satisified:  {line}, {idx}")
                        pass
            idx += 1
        with open(instances_file, 'w') as f:
            f.writelines(instance_lines)





if __name__ == '__main__':
    input_model_path = '/home/u1411251/tools/vnncomp_benchmarks/mnist_fc/onnx/mnist-net_256x4.onnx'
    output_model_path = "temp.onnx"
    # update_fc_relu_strong(input_model_path, output_model_path)
    get_delta_strong(15, 10)