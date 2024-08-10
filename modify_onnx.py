import onnx
from onnx import helper, shape_inference, TensorProto
import numpy as np
import math
import os
import sys

def modify_onnx_model(input_model_path, output_model_path):
    # Load the existing model
    model = onnx.load(input_model_path)
    graph = model.graph

    # Identify the input tensor and the Flatten node
    input_tensor = graph.input[0]
    flatten_node = None
    nodes_to_keep = []
    add_nodes = False

    for node in graph.node:
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
    # Save the modified model

    # prev_output_name = int(graph.output[0].name)
    # relu_output_name = prev_output_name+1

    # relu_node = helper.make_node(
    #     'Relu',
    #     inputs=[str(prev_output_name)],
    #     outputs=[str(relu_output_name)],
    #     name=str(relu_output_name)
    # )

    # output_fc_layer_name = relu_output_name+1

    # weight_name = f"layers.{out_layer_idx+2}.weight"
    # bias_name = f"layers.{out_layer_idx+2}.bias"

    # fc_node = helper.make_node(
    #     'Gemm',
    #     inputs=[str(relu_output_name), weight_name, bias_name],
    #     outputs=[str(output_fc_layer_name)],
    #     alpha=1.0,
    #     beta=1.0,
    #     transB=1,
    #     name=str(output_fc_layer_name)
    # )

    # weight = get_output_layer_weight()
    # fc_weight = helper.make_tensor(
    #     name=weight_name,
    #     data_type=TensorProto.FLOAT,
    #     dims=[9, 81],
    #     vals=weight
    # )

    # fc_bias = helper.make_tensor(
    #     name=bias_name,
    #     data_type=TensorProto.FLOAT,
    #     dims=[9],
    #     vals=[0.0] * 9
    # )


    # graph.node.append(relu_node)
    # graph.node.append(fc_node)
    # graph.initializer.append(fc_weight)
    # graph.initializer.append(fc_bias)
    # graph.output[0].name = str(output_fc_layer_name)

    for output in graph.output:
        # Assuming there is only one output tensor, if there are multiple, you may need to specify the exact one
        dim_value = output.type.tensor_type.shape.dim
        if len(dim_value) == 2:
            dim_value[1].dim_value = 81

    onnx.save(model, output_model_path)



def get_delta(conf):
    val = (100.0/conf) - 1
    ln = math.log(val, math.e)
    return -ln

# Example usage

output_dir = '/home/afzal/Documents/tools/networks/conf/nets1'
input_model_path = '/home/afzal/Documents/tools/networks/vnncomp2021/benchmarks/mnistfc/mnist-net_256x2.onnx'
dataset_path = '/home/afzal/Documents/tools/VeriNN/deep_refine/benchmarks/dataset/mnist/mnist_test.csv'
conf = 90
num_images= 100

if len(sys.argv) == 3:
    input_model_path = str(sys.argv[1])
    conf = str(sys.argv[2])

net_name = os.path.basename(input_model_path)
output_models= []

labels = []
with open(dataset_path) as f:
    Lines = f.readlines()
    for line in Lines:
        labels.append(int(line[0]))

for i in range(num_images):
    net_name1 = net_name[:-5]+"_"+str(conf)+"_"+str(i)+".onnx"
    output_model_path = os.path.join(output_dir, net_name1)

    output_models.append(output_model_path)

delta = get_delta(float(conf))

for i in range(num_images):
    update_fc_relu_to_model(input_model_path, output_models[i], labels[i], delta)

# output_model_path = 'temp_appended_layer.onnx'
# # update_fc_relu_to_model(input_model_path, output_model_path)
# update_fc_to_model(input_model_path, output_model_path)
# get_output_layer_weight()
