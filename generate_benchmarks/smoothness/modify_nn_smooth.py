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

def get_fc1_layer_weights_smooth(label, output_dims):
    weights = []
    for i in range(output_dims):
        if i != label:
            l = [0.0]*output_dims
            l[i] = 1.0
            l[label] = -1.0
            weights.append(l)
    
    for i in range(output_dims):
        if i != label:
            l = [0.0]*output_dims
            l[i] = 1.0
            l[label] = -1.0
            weights.append(l)


    return weights

def get_fc_layer_weights_smooth_single_cond(label, output_dims):
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

def get_delta_2(conf, output_dims):
    # delta = -ln((1/(dim - 1))*((100/conf) -1))
    temp = (100/conf) -1
    temp = temp/(output_dims-1)
    delta = -math.log(temp)
    # print(delta)
    return delta

def get_delta_1(conf):
    # delta = -ln((100/conf) -1))
    temp = (100/conf) -1
    delta = -math.log(temp)
    return delta

def update_fc_relu_smooth_cond1(model_path, output_model_path, label = 0, delta=1.98, existing_model_out_dims = 10):
    w_weight_name, b_weight_name  = get_output_affine_layers_weights(model_path)
    model = onnx.load(model_path)
    graph = model.graph
    output_layer_weight, output_layer_bias = get_output_layers_w_b(graph, w_weight_name, b_weight_name)
    fc_output_dim = existing_model_out_dims - 1
    final_out_dim = 1
        
    new_w = get_fc_layer_weights_smooth_single_cond(label, output_dims=existing_model_out_dims)
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

    weight = [-1.0]*fc_output_dim
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

def update_fc_relu_smooth_cond2(model_path, output_model_path, label = 0, delta=1.98, existing_model_out_dims = 10):
    w_weight_name, b_weight_name  = get_output_affine_layers_weights(model_path)
    model = onnx.load(model_path)
    graph = model.graph
    output_layer_weight, output_layer_bias = get_output_layers_w_b(graph, w_weight_name, b_weight_name)
    fc_output_dim = existing_model_out_dims - 1
    final_out_dim = 1
        
    new_w = get_fc_layer_weights_smooth_single_cond(label, output_dims=existing_model_out_dims)
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

def update_fc_relu_smooth_both_conds(model_path, output_model_path, label = 0, delta_1 = 1.98, delta_2 = 2.02, existing_model_out_dims = 10, input_tolerance = 1e-5):
    w_weight_name, b_weight_name  = get_output_affine_layers_weights(model_path)
    model = onnx.load(model_path)
    graph = model.graph
    output_layer_weight, output_layer_bias = get_output_layers_w_b(graph, w_weight_name, b_weight_name)
    fc1_output_dim = 2*(existing_model_out_dims - 1)
    fc2_output_dim = 2
    final_out_dim = 1
        
    new_w = get_fc1_layer_weights_smooth(label, output_dims=existing_model_out_dims)
    new_fc_weight = np.reshape(new_w, (fc1_output_dim, existing_model_out_dims))
    new_fc_weight = np.asarray(new_fc_weight, dtype=np.float32)
    new_fc_bias = np.array([delta_1]*(existing_model_out_dims-1) + [delta_2]*(existing_model_out_dims-1), dtype=np.float32)
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
    relu_1_output_name = 'appnded.relu1'

    relu_node_1 = helper.make_node('Relu', inputs=[str(prev_output_name)], outputs=[str(relu_1_output_name)], 
                                 name=str(relu_1_output_name)
                                 )

    output_fc_layer_1_name = 'appended_fc1'

    weight_name_1 = f"layer.appended.weight1"
    bias_name_1 = f"layer.appended.bias1"

    fc_node_1 = helper.make_node('Gemm', inputs=[str(relu_1_output_name), weight_name_1, bias_name_1], 
                               outputs=[str(output_fc_layer_1_name)], alpha=1.0, beta=1.0, transB=1,
                               name=str(output_fc_layer_1_name)
                               )

    weight = [-1.0]*(existing_model_out_dims-1) + [0.0]*fc1_output_dim + [1.0]*(existing_model_out_dims-1)
    fc_weight1 = helper.make_tensor(name=weight_name_1, data_type=TensorProto.FLOAT, dims=[fc2_output_dim, fc1_output_dim],vals=weight)

    fc_bias1 = helper.make_tensor(name=bias_name_1, data_type=TensorProto.FLOAT, dims=[fc2_output_dim], vals=[input_tolerance, -input_tolerance])

    relu_2_output_name = 'appnded.relu2'
    relu_node_2 = helper.make_node('Relu', inputs=[output_fc_layer_1_name], outputs=[str(relu_2_output_name)], 
                                 name=str(relu_2_output_name)
                                 )
    

    output_fc_layer_2_name = 'appended_fc2'
    weight_name_2 = f"layer.appended.weight2"
    bias_name_2 = f"layer.appended.bias2"
    fc_node_2 = helper.make_node('Gemm', inputs=[str(relu_2_output_name), weight_name_2, bias_name_2], 
                               outputs=[str(output_fc_layer_2_name)], alpha=1.0, beta=1.0, transB=1,
                               name=str(output_fc_layer_2_name)
                               )
    
    fc_weight2 = helper.make_tensor(name=weight_name_2, data_type=TensorProto.FLOAT, dims=[final_out_dim, fc2_output_dim],vals=[1.0,1.0])

    fc_bias2 = helper.make_tensor(name=bias_name_2, data_type=TensorProto.FLOAT, dims=[final_out_dim], vals=[0]*final_out_dim)

    graph.node.append(relu_node_1)
    graph.node.append(fc_node_1)
    graph.node.append(relu_node_2)
    graph.node.append(fc_node_2)
    graph.initializer.append(fc_weight1)
    graph.initializer.append(fc_bias1)
    graph.initializer.append(fc_weight2)
    graph.initializer.append(fc_bias2)
    graph.output[0].name = str(output_fc_layer_2_name)

    for output in graph.output:
        # Assuming there is only one output tensor, if there are multiple, you may need to specify the exact one
        dim_value = output.type.tensor_type.shape.dim
        if len(dim_value) == 2:
            dim_value[1].dim_value = final_out_dim

    model = shape_inference.infer_shapes(model)
    # print(f"Network modified: {output_model_path}")
    onnx.save(model, output_model_path)

def are_conds_false(net_prp, file_path, th):
    # print(net_prp)
    conf = 57
    try:
        with open(file_path, 'r') as f:
            Lines = f.readlines()
            for line in Lines:
                line = line.strip()
                line_l = line.split(',')
                if net_prp in line:
                    conf1 = float(line_l[-1])
                    conf1 = conf1*100
                    conf = conf1
    except:
        pass
    
    # print(conf, th)
    cond1 = (conf + th) >= 100
    cond2 = (conf - th) <= 0
    
    # print(f"Check....{net_prp}")
    return cond1, cond2

def setup_on_vnncomp_prop_smoothness(dataset, confs, timeout, epsilons, target_benchmarks_dir, vnncomp_benchmarks_dir, tolerance_param, is_less_than_output_prp, conf_file, is_target_prop=False):
    input_tolerance = 1e-4
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
            for conf in confs:
                if conf == 0:
                    shutil.copy2(vnncomp_net_path, target_net_path)
                    shutil.copy2(vnncomp_prp_path, target_prp_path)
                    ins_line = f"{os.path.join(sub_net_dir, new_net_name)},{os.path.join(sub_prp_dir, new_prp_name)},{timeout}\n"
                    instance_lines.append(ins_line)
                else:
                    tolerance_param_temp = tolerance_param
                    new_net_name = f"{netname[:-5]}_{idx}_{conf}.onnx"
                    new_prp_name =  f"{prpname[:-7]}_{idx}_{conf}.vnnlib"
                    target_net_path = os.path.join(target_net_dir, new_net_name)
                    target_prp_path = os.path.join(target_prp_dir, new_prp_name)
                    is_cond1_false, is_cond2_false = are_conds_false(line, file_path=conf_file, th=conf)
                    if is_cond1_false and is_cond2_false:
                        print(f"Both conds are false: {new_net_name},{new_prp_name}")
                        continue
                    if is_cond1_false:
                        delta2 = get_delta_2(conf, output_dims)
                        print(f"Cond1 is false: {new_net_name},{new_prp_name}")
                        update_fc_relu_smooth_cond2(model_path=vnncomp_net_path, output_model_path=target_net_path, label=label, delta=delta2, existing_model_out_dims=output_dims)
                    elif is_cond2_false:
                        delta1 = get_delta_1(conf)
                        print(f"Cond2 is false: {new_net_name},{new_prp_name}")
                        update_fc_relu_smooth_cond1(model_path=vnncomp_net_path, output_model_path=target_net_path, label=label, delta=delta1, existing_model_out_dims=output_dims)
                        tolerance_param_temp = -tolerance_param
                    else:
                        print(f"Both conds are true: {new_net_name},{new_prp_name}")
                        delta1 = get_delta_1(conf)
                        delta2 = get_delta_2(conf, output_dims)
                        update_fc_relu_smooth_both_conds(model_path=vnncomp_net_path, output_model_path=target_net_path, label=label, delta_1=delta1, delta_2=delta2, existing_model_out_dims=output_dims, input_tolerance=input_tolerance)

                    save_vnnlib_from_vnncomp(vnncomp_prp_path, target_prp_path, conf=conf, total_output_class=new_out_dims, tolerance_param=tolerance_param_temp, orignal_out_classes=output_dims)
                    ins_line = f"{os.path.join(sub_net_dir, new_net_name)},{os.path.join(sub_prp_dir, new_prp_name)},{timeout}\n"
                    instance_lines.append(ins_line)
            idx += 1
        with open(instances_file, 'w') as f:
            f.writelines(instance_lines)





if __name__ == '__main__':
    input_model_path = '/home/u1411251/tools/vnncomp_benchmarks/mnist_fc/onnx/mnist-net_256x4.onnx'
    output_model_path = "temp.onnx"
    # update_fc_relu_smooth_both_conds(input_model_path, output_model_path, label = 0, delta_1 = 1.98, delta_2 = 2.02, existing_model_out_dims = 10, input_tolerance = 1e-5)
    update_fc_relu_smooth_cond1(input_model_path, output_model_path, label = 0, delta=1.98, existing_model_out_dims = 10)
    # get_delta_strong(15, 10)