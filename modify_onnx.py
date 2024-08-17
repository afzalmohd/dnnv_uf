import onnx
from onnx import helper, shape_inference, TensorProto
import numpy as np
import math
import os
import sys

def layers_index_reduce(model):
    graph = model.graph
    rename_map = {}
    for init in model.graph.initializer:
        name = init.name
        layer_idx = int(str(name).split('.')[0])
        w_name = str(init.name).split('.')[1]
        layer_idx -= 2
        name1 = f"{layer_idx}.{w_name}"
        rename_map[name] = name1

    # print(rename_map)

    for initializer in graph.initializer:
        if initializer.name in rename_map:
            initializer.name = rename_map[initializer.name]

    for node in graph.node:
        for i, input_name in enumerate(node.input):
            if input_name in rename_map:
                node.input[i] = rename_map[input_name]
        for i, output_name in enumerate(node.output):
            if output_name in rename_map:
                node.output[i] = rename_map[output_name]

def change_input_dims(model):
    graph = model.graph
    for input_tensor in graph.input:
        # Assuming there's only one input tensor
        shape = input_tensor.type.tensor_type.shape.dim        
        # Update the dimensions to (1, 784, 1)
        shape[1].dim_value = 784  # Update the second dimension
        shape[2].dim_value = 1    # Update the third dimension
        shape.pop(3)              # Remove the fourth dimension

def get_input_of_output_node(model):
    graph = model.graph
    output_name = graph.output[0].name  # Assuming there's one output
    output_node = None

    for node in graph.node:
        if output_name in node.output:
            output_node = node
            break

    if output_node:
        print("Output node name:", output_node.name)
        print("Input(s) to the output node:", output_node.input)
        return output_node.input
    else:
        print("Output node not found.")


def change_output_node(model):
    graph = model.graph
    new_input_name = str(0)
    if len(graph.input) > 0:
        original_input = graph.input[0]
        print(f"Original input name: {original_input.name}")

        # Change the name of the input tensor
        old_input_name = original_input.name
        original_input.name = new_input_name

        # Update all nodes that use this input
        for node in graph.node:
            for i, input_name in enumerate(node.input):
                if input_name == old_input_name:
                    node.input[i] = new_input_name

        print(f"Updated input name: {new_input_name}")

    
    output_name = graph.output[0].name  # Assuming there's one output
    output_node = None
    input_name = None
    for node in graph.node:
        if output_name in node.output:
            output_node = node
            break

    if output_node:
        input_name = int(output_node.input[0])
    else:
        print("Output node not found")


    new_output_name = str(input_name+1)

    if len(graph.output) > 0:
        original_output = graph.output[0]
        print(f"Original output name: {original_output.name}")

        # Change the name of the output tensor
        old_output_name = original_output.name
        original_output.name = new_output_name

        # Update all nodes that produce this output
        for node in graph.node:
            for i, output_name in enumerate(node.output):
                if output_name == old_output_name:
                    node.output[i] = new_output_name

        print(f"Updated output name: {new_output_name}")


    # if len(graph.output) > 0:
    #     original_output = graph.output[0]
    #     print(f"Original output name: {original_output.name}")
    #     original_output_name = original_output.name
    #     # Set the new output name
    #     original_output.name = new_output_name

    #     # Also, change the corresponding node output name to match
    #     for node in graph.node:
    #         for i, output_name in enumerate(node.output):
    #             if output_name == original_output_name:
    #                 node.output[i] = new_output_name

    #     print(f"Updated output name: {new_output_name}")


def modify_onnx_model(input_model_path, output_model_path):
    # Load the existing model
    model = onnx.load(input_model_path)
    graph = model.graph

    # Identify the input tensor and the Flatten node
    input_tensor = graph.input[0]
    flatten_node = None
    nodes_to_keep = []
    add_nodes = False

    layers_index_reduce(model)
    change_input_dims(model)

    for node in graph.node:
        # print(node.name)
        if node.op_type == "Flatten":
            flatten_node = node
            add_nodes = True
        if add_nodes:
            nodes_to_keep.append(node)

    if flatten_node is None:
        raise ValueError("Flatten node not found in the model")

    # Adjust the Flatten node to take the input tensor directly
    flatten_node.input[0] = input_tensor.name

    # Create a new graph with the required nodes
    new_graph = helper.make_graph(
        nodes_to_keep,
        graph.name,
        [input_tensor],
        graph.output,
        initializer=graph.initializer
    )


    # Create a new model
    new_model = helper.make_model(new_graph, producer_name='onnx-example')

    change_output_node(new_model)

    # Infer shapes (optional but recommended)
    new_model = shape_inference.infer_shapes(new_model)


    # Save the new model
    onnx.save(new_model, output_model_path)

def get_fc_layer_weights(label, output_dims=10):
    weights = []
    for i in range(output_dims):
        if i != label:
            for j in range(output_dims):
                l = [0.0]*output_dims
                l[i] = -1.0
                if j != i:
                    l[j] = 1.0
                    weights += l

    return weights

def get_fc_layer_weights_inverse(label, output_dims=10):
    weights = []
    for i in range(output_dims):
        if i != label:
            for j in range(output_dims):
                l = [0.0]*output_dims
                l[i] = 1.0
                if j != i:
                    l[j] = -1.0
                    weights += l

    return weights
                

def get_output_layer_weight():
    weights = []
    for i in range(9):
        l = [0.0]*81
        for j in range(9):
            l[9*i + j] = -1.0
        weights += l

    return weights


def update_fc_relu_to_model_with_relu_output(model_path, output_model_path, label = 0, delta=1.98, fc_output_dim=81):
      # Load the existing ONNX model
    model = onnx.load(model_path)
    graph = model.graph
    
    # Retrieve the weight and bias initializers for the existing output FC layer
    output_layer_weight = None
    output_layer_bias = None
    output_layer_input_dim = None
    output_layer_output_dim = None

    out_layer_idx = 0
    for initializer in graph.initializer:
        output_init = initializer.name
        layer_idx = int(output_init.split('.')[0])
        if layer_idx > out_layer_idx:
            out_layer_idx = layer_idx

    print(out_layer_idx)
    for initializer in graph.initializer:
        # print(initializer.name)
        if f"{out_layer_idx}.weight" in  initializer.name:
            output_layer_weight = np.frombuffer(initializer.raw_data, dtype=np.float32).reshape(initializer.dims)
            output_layer_input_dim = initializer.dims[1]
            output_layer_output_dim = initializer.dims[0]
        elif f"{out_layer_idx}.bias" in  initializer.name:
            output_layer_bias = np.frombuffer(initializer.raw_data, dtype=np.float32)
    
    # Initialize the new fully connected layer's weights and biases
    new_w = get_fc_layer_weights(label)
    new_fc_weight1 = np.reshape(new_w, (fc_output_dim, output_layer_output_dim))
    new_fc_weight1 = np.asarray(new_fc_weight1, dtype=np.float32)
    new_fc_bias1 = np.array([delta]*fc_output_dim, dtype=np.float32)

    
    
    prev_output_name = int(graph.output[0].name)

    fc1_output_name = prev_output_name+1

    weight_name1 = f"{out_layer_idx+2}.weight"
    bias_name1 = f"{out_layer_idx+2}.bias"

    fc_node1 = helper.make_node(
        'Gemm',
        inputs=[str(prev_output_name), weight_name1, bias_name1],
        outputs=[str(fc1_output_name)],
        alpha=1.0,
        beta=1.0,
        transB=1,
        name=str(fc1_output_name)
    )


    fc_weight1 = helper.make_tensor(
        name=weight_name1,
        data_type=TensorProto.FLOAT,
        dims=[81, 10],
        vals=new_fc_weight1
    )

    fc_bias1 = helper.make_tensor(
        name=bias_name1,
        data_type=TensorProto.FLOAT,
        dims=[81],
        vals=new_fc_bias1
    )



    relu_output_name = fc1_output_name+1

    relu_node = helper.make_node(
        'Relu',
        inputs=[str(fc1_output_name)],
        outputs=[str(relu_output_name)],
        name=str(relu_output_name)
    )

    output_fc_layer_name = relu_output_name+1

    weight_name2 = f"layers.{out_layer_idx+4}.weight"
    bias_name2 = f"layers.{out_layer_idx+4}.bias"

    fc_node2 = helper.make_node(
        'Gemm',
        inputs=[str(relu_output_name), weight_name2, bias_name2],
        outputs=[str(output_fc_layer_name)],
        alpha=1.0,
        beta=1.0,
        transB=1,
        name=str(output_fc_layer_name)
    )

    new_fc_weight2 = get_output_layer_weight()
    fc_weight2 = helper.make_tensor(
        name=weight_name2,
        data_type=TensorProto.FLOAT,
        dims=[9, 81],
        vals=new_fc_weight2
    )

    fc_bias2 = helper.make_tensor(
        name=bias_name2,
        data_type=TensorProto.FLOAT,
        dims=[9],
        vals=[0.0] * 9
    )
    # print(graph.node)
    graph.node.append(fc_node1)
    graph.node.append(relu_node)
    graph.node.append(fc_node2)
    graph.initializer.append(fc_weight1)
    graph.initializer.append(fc_bias1)
    graph.initializer.append(fc_weight2)
    graph.initializer.append(fc_bias2)
    graph.output[0].name = str(output_fc_layer_name)

    for output in graph.output:
        # Assuming there is only one output tensor, if there are multiple, you may need to specify the exact one
        dim_value = output.type.tensor_type.shape.dim
        if len(dim_value) == 2:
            dim_value[1].dim_value = 9

    # Infer shapes (optional but recommended)
    model = shape_inference.infer_shapes(model)

    onnx.save(model, output_model_path)


def update_fc_relu_to_model(model_path, output_model_path, label = 0, delta=1.98, fc_output_dim=81):
      # Load the existing ONNX model
    model = onnx.load(model_path)
    graph = model.graph
    
    # Retrieve the weight and bias initializers for the existing output FC layer
    output_layer_weight = None
    output_layer_bias = None
    output_layer_input_dim = None
    output_layer_output_dim = None

    out_layer_idx = 0
    for initializer in graph.initializer:
        output_init = initializer.name
        layer_idx = int(output_init.split('.')[1])
        if layer_idx > out_layer_idx:
            out_layer_idx = layer_idx

    print(out_layer_idx)
    for initializer in graph.initializer:
        # print(initializer.name)
        if f"{out_layer_idx}.weight" in  initializer.name:
            output_layer_weight = np.frombuffer(initializer.raw_data, dtype=np.float32).reshape(initializer.dims)
            output_layer_input_dim = initializer.dims[1]
            output_layer_output_dim = initializer.dims[0]
        elif f"{out_layer_idx}.bias" in  initializer.name:
            output_layer_bias = np.frombuffer(initializer.raw_data, dtype=np.float32)
    
    # Initialize the new fully connected layer's weights and biases
    new_w = get_fc_layer_weights(label)
    new_fc_weight = np.reshape(new_w, (fc_output_dim, output_layer_output_dim))
    new_fc_bias = np.array([delta]*fc_output_dim)

    # Combine the weights and biases
    combined_weight = np.dot(new_fc_weight, output_layer_weight)
    combined_bias = np.dot(new_fc_weight, output_layer_bias) + new_fc_bias

    combined_weight = np.asarray(combined_weight, dtype=np.float32)
    combined_bias = np.asarray(combined_bias, dtype=np.float32)


    # Update the initializers in the graph
    for initializer in graph.initializer:
        if f"{out_layer_idx}.weight" in  initializer.name:
            initializer.raw_data = combined_weight.tobytes()
            initializer.dims[:] = combined_weight.shape
        elif f"{out_layer_idx}.bias" in  initializer.name:
            initializer.raw_data = combined_bias.tobytes()
            initializer.dims[:] = combined_bias.shape
    # Save the modified model

    prev_output_name = int(graph.output[0].name)
    relu_output_name = prev_output_name+1

    relu_node = helper.make_node(
        'Relu',
        inputs=[str(prev_output_name)],
        outputs=[str(relu_output_name)],
        name=str(relu_output_name)
    )

    output_fc_layer_name = relu_output_name+1

    weight_name = f"layers.{out_layer_idx+2}.weight"
    bias_name = f"layers.{out_layer_idx+2}.bias"

    fc_node = helper.make_node(
        'Gemm',
        inputs=[str(relu_output_name), weight_name, bias_name],
        outputs=[str(output_fc_layer_name)],
        alpha=1.0,
        beta=1.0,
        transB=1,
        name=str(output_fc_layer_name)
    )

    weight = get_output_layer_weight()
    fc_weight = helper.make_tensor(
        name=weight_name,
        data_type=TensorProto.FLOAT,
        dims=[9, 81],
        vals=weight
    )

    fc_bias = helper.make_tensor(
        name=bias_name,
        data_type=TensorProto.FLOAT,
        dims=[9],
        vals=[0.0] * 9
    )


    graph.node.append(relu_node)
    graph.node.append(fc_node)
    graph.initializer.append(fc_weight)
    graph.initializer.append(fc_bias)
    graph.output[0].name = str(output_fc_layer_name)

    for output in graph.output:
        # Assuming there is only one output tensor, if there are multiple, you may need to specify the exact one
        dim_value = output.type.tensor_type.shape.dim
        if len(dim_value) == 2:
            dim_value[1].dim_value = 9

    onnx.save(model, output_model_path)


def update_fc_to_model(model_path, output_model_path, label = 0, delta=1.98, fc_output_dim=81):
      # Load the existing ONNX model
    model = onnx.load(model_path)
    graph = model.graph
    
    # Retrieve the weight and bias initializers for the existing output FC layer
    output_layer_weight = None
    output_layer_bias = None
    output_layer_input_dim = None
    output_layer_output_dim = None

    out_layer_idx = 0
    for initializer in graph.initializer:
        output_init = initializer.name
        layer_idx = int(output_init.split('.')[1])
        if layer_idx > out_layer_idx:
            out_layer_idx = layer_idx

    # print(out_layer_idx)
    for initializer in graph.initializer:
        # print(initializer.name)
        if f"{out_layer_idx}.weight" in  initializer.name:
            output_layer_weight = np.frombuffer(initializer.raw_data, dtype=np.float32).reshape(initializer.dims)
            output_layer_input_dim = initializer.dims[1]
            output_layer_output_dim = initializer.dims[0]
        elif f"{out_layer_idx}.bias" in  initializer.name:
            output_layer_bias = np.frombuffer(initializer.raw_data, dtype=np.float32)
    
    # Initialize the new fully connected layer's weights and biases
    new_w = get_fc_layer_weights_inverse(label)
    new_fc_weight = np.reshape(new_w, (fc_output_dim, output_layer_output_dim))
    new_fc_weight = np.asarray(new_fc_weight, dtype=np.float32)
    new_fc_bias = np.array([-delta]*fc_output_dim, dtype=np.float32)
    print(new_fc_weight)
    print(new_fc_bias)
    # Combine the weights and biases
    combined_weight = np.dot(new_fc_weight, output_layer_weight)
    combined_bias = np.dot(new_fc_weight, output_layer_bias) + new_fc_bias

    combined_weight = np.asarray(combined_weight, dtype=np.float32)
    combined_bias = np.asarray(combined_bias, dtype=np.float32)


    # Update the initializers in the graph
    for initializer in graph.initializer:
        if f"{out_layer_idx}.weight" in  initializer.name:
            initializer.raw_data = combined_weight.tobytes()
            initializer.dims[:] = combined_weight.shape
        elif f"{out_layer_idx}.bias" in  initializer.name:
            initializer.raw_data = combined_bias.tobytes()
            initializer.dims[:] = combined_bias.shape

    for output in graph.output:
        # Assuming there is only one output tensor, if there are multiple, you may need to specify the exact one
        dim_value = output.type.tensor_type.shape.dim
        if len(dim_value) == 2:
            dim_value[1].dim_value = 81

    onnx.save(model, output_model_path)

def append_fc_layer_to_model(model_path, output_model_path, label = 7, conf = 0.6, fc_output_dim=9):
    # Load the existing ONNX model
    conf = conf / 100
    model = onnx.load(model_path)
    graph = model.graph
    
    # Retrieve the weight and bias initializers for the existing output FC layer
    output_layer_output_dim = 10

    out_layer_idx = 0
    for initializer in graph.initializer:
        output_init = initializer.name
        layer_idx = int(output_init.split('.')[0])
        if layer_idx > out_layer_idx:
            out_layer_idx = layer_idx

    print(out_layer_idx)
    new_w = []
    for i in range(output_layer_output_dim):
        if i != label:
            l = [-conf]*output_layer_output_dim
            l[i] = 1.0 - conf
            new_w.append(l)
    
    new_fc_weight = np.reshape(new_w, (fc_output_dim, output_layer_output_dim))
    new_fc_weight = np.asarray(new_fc_weight, dtype=np.float32)
    new_fc_bias = np.array([0.0]*fc_output_dim, dtype=np.float32)


    prev_output_name = int(graph.output[0].name)
    fc_output_name = prev_output_name+1

    weight_name = f"{out_layer_idx+2}.weight"
    bias_name = f"{out_layer_idx+2}.bias"

    fc_node = helper.make_node(
        'Gemm',
        inputs=[str(prev_output_name), weight_name, bias_name],
        outputs=[str(fc_output_name)],
        alpha=1.0,
        beta=1.0,
        transB=1,
        name=str(fc_output_name)
    )

    fc_weight = helper.make_tensor(
        name=weight_name,
        data_type=TensorProto.FLOAT,
        dims=[9, 10],
        vals=new_fc_weight
    )

    fc_bias = helper.make_tensor(
        name=bias_name,
        data_type=TensorProto.FLOAT,
        dims=[9],
        vals=new_fc_bias
    )


    graph.node.append(fc_node)
    graph.initializer.append(fc_weight)
    graph.initializer.append(fc_bias)
    graph.output[0].name = str(fc_output_name)

    for output in graph.output:
        # Assuming there is only one output tensor, if there are multiple, you may need to specify the exact one
        dim_value = output.type.tensor_type.shape.dim
        if len(dim_value) == 2:
            dim_value[1].dim_value = 9

    onnx.save(model, output_model_path)

def get_delta(conf):
    val = (100.0/conf) - 1
    ln = math.log(val, math.e)
    return -ln

# Example usage

input_dir = '/home/afzal/tools/networks/conf_final/eran_mod'
output_dir = '/home/afzal/tools/networks/conf_final/eran_mod_simple_conf'

nets = ['mnist_relu_3_50.onnx', 'mnist_relu_3_100.onnx', 'mnist_relu_4_1024.onnx', 'mnist_relu_5_100.onnx', 'mnist_relu_6_100.onnx']
nets += ['mnist_relu_6_200.onnx', 'mnist_relu_9_100.onnx', 'mnist_relu_9_200.onnx', 'ffnnRELU__Point_6_500.onnx']
nets += ['ffnnRELU__PGDK_w_0.1_6_500.onnx', 'ffnnRELU__PGDK_w_0.3_6_500.onnx']

input_model_paths = []
for net in nets:
    input_model_paths.append(os.path.join(input_dir, net))

dataset_path = '/home/afzal/tools/VeriNN/deep_refine/benchmarks/dataset/mnist/mnist_test.csv'
confs = [40, 60, 80]
num_images= 100


# for i in range(len(input_model_paths)):
#     input_path = input_model_paths[i]
#     output_path = os.path.join(output_dir, os.path.basename(input_path))
#     modify_onnx_model(input_path, output_path)


# exit(0)

labels = []
with open(dataset_path) as f:
    Lines = f.readlines()
    for line in Lines:
        labels.append(int(line[0]))


for input_path in input_model_paths:
    for conf in confs:
        delta = get_delta(float(conf))
        for i in range(num_images):
            net_name = os.path.basename(input_path)
            net_name = net_name[:-5]+"_"+str(conf)+"_"+str(i)+".onnx"
            out_path = os.path.join(output_dir, net_name)
            # update_fc_relu_to_model_with_relu_output(input_path, out_path, labels[i], delta)
            append_fc_layer_to_model(input_path, out_path, labels[i], conf)


# output_model_path = 'temp_appended_layer.onnx'
# # update_fc_relu_to_model(input_model_path, output_model_path)
# update_fc_relu_to_model_with_relu_output(input_model_path, "temp.onnx", 7, delta)
# get_output_layer_weight()
