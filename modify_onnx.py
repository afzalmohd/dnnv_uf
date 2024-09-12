import onnx
from onnx import helper, shape_inference, TensorProto
import numpy as np
import math
import os
import sys
import shutil
from simulate_network import read_images_from_dataset
from simulate_network import get_selected_images
from simulate_network import run_network_mnist_test
from simulate_network import get_mnist_test_data
from simulate_network import get_mnist_train_data



def is_output_layer_activation_fn(model_path):
    onnx_model = onnx.load(model_path)
    # Get the graph nodes and outputs from the model
    graph = onnx_model.graph
    # The model's output
    model_outputs = [output.name for output in graph.output]

    # Check nodes to find the output layer and its preceding operation
    output_activations = []
    for node in graph.node:
        # If the output of the node matches the model's output, check the operation type
        for output in node.output:
            if output in model_outputs:
                if node.op_type in ['Softmax', 'Sigmoid', 'Relu']:
                    output_activations.append((output, node.op_type))

    # Check if we found an activation function applied to the model output
    if output_activations:
        for activation in output_activations:
            pass
            # print(f"Activation function {activation[1]} applied on output {activation[0]}")
        return True
    else:
        # print("No activation function applied on the model output.")
        return False

def get_delta(conf):
    val = (100.0/conf) - 1
    ln = math.log(val, math.e)
    return -ln


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

def get_fc_layer_weights_simple(label, output_dims=10, conf=0.40):
    weights = []
    for i in range(output_dims):
        if i != label:
            l1 = [conf]*output_dims
            l1[i] = (conf - 1)
            l2 = [0.0]*output_dims
            l2[label] = 1.0
            l2[i] = -1.0
            weights += l1
            weights += l2
    
    # print(len(weights))
    # print(np.array(weights).reshape(18,10))
    return weights

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

    # print(len(weights))
    # print(np.array(weights).reshape(81,10))
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

def get_output_layer_weight_simple():
    weights = []
    for i in range(9):
        l = [0.0]*18
        l[2*i] = -1.0
        l[2*i + 1] = -1.0
        weights += l

    # print(len(weights))
    # print(np.array([weights]).reshape(9,18))
    return weights

def append_fc_relu_softmax(model_path, output_model_path, label = 0, delta=1.98, fc_output_dim=81, existing_model_out_dims = 10):
    # Load the existing ONNX model
    model = onnx.load(model_path)
    graph = model.graph
    

    out_layer_idx = 0
    for initializer in graph.initializer:
        output_init = initializer.name
        layer_idx = int(output_init.split('.')[0])
        if layer_idx > out_layer_idx:
            out_layer_idx = layer_idx

    # print(out_layer_idx)
    
    # Initialize the new fully connected layer's weights and biases
    new_w = get_fc_layer_weights(label)
    new_fc_weight1 = np.reshape(new_w, (fc_output_dim, existing_model_out_dims))
    new_fc_weight1 = np.asarray(new_fc_weight1, dtype=np.float32)
    new_fc_bias1 = np.array([delta]*fc_output_dim, dtype=np.float32)
    
    prev_output_name = int(graph.output[0].name)

    fc1_output_name = prev_output_name+1

    weight_name1 = f"{out_layer_idx+2}.weight"
    bias_name1 = f"{out_layer_idx+2}.bias"

    fc_node1 = helper.make_node('Gemm', inputs=[str(prev_output_name), weight_name1, bias_name1], 
                                outputs=[str(fc1_output_name)], alpha=1.0,  beta=1.0, transB=1, 
                                name=str(fc1_output_name))


    fc_weight1 = helper.make_tensor(name=weight_name1, data_type=TensorProto.FLOAT, 
                                    dims=[fc_output_dim, existing_model_out_dims], vals=new_fc_weight1)

    fc_bias1 = helper.make_tensor(name=bias_name1, data_type=TensorProto.FLOAT, dims=[fc_output_dim],vals=new_fc_bias1)



    relu_output_name = fc1_output_name+1

    relu_node = helper.make_node('Relu', inputs=[str(fc1_output_name)], outputs=[str(relu_output_name)], 
                                 name=str(relu_output_name)
                                 )

    output_fc_layer_name = relu_output_name+1

    weight_name2 = f"layers.{out_layer_idx+4}.weight"
    bias_name2 = f"layers.{out_layer_idx+4}.bias"

    fc_node2 = helper.make_node('Gemm', inputs=[str(relu_output_name), weight_name2, bias_name2], 
                                outputs=[str(output_fc_layer_name)], alpha=1.0, beta=1.0, transB=1, 
                                name=str(output_fc_layer_name)
                                )

    new_fc_weight2 = get_output_layer_weight()
    fc_weight2 = helper.make_tensor(name=weight_name2, data_type=TensorProto.FLOAT, dims=[9, fc_output_dim], 
                                    vals=new_fc_weight2
                                    )

    fc_bias2 = helper.make_tensor(name=bias_name2, data_type=TensorProto.FLOAT, dims=[9], vals=[0.0] * 9)
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


def update_fc_relu_softmax(model_path, output_model_path, label = 0, delta=1.98, fc_output_dim=81, existing_model_out_dims = 10):
      # Load the existing ONNX model
    model = onnx.load(model_path)
    graph = model.graph
    
    # Retrieve the weight and bias initializers for the existing output FC layer
    output_layer_weight = None
    output_layer_bias = None
    output_layer_output_dim = None

    out_layer_idx = 0
    for initializer in graph.initializer:
        output_init = initializer.name
        # print(output_init)
        try:
            layer_idx = int(output_init.split('.')[0])
            if layer_idx > out_layer_idx:
                out_layer_idx = layer_idx
        except:
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
    new_fc_weight = np.asarray(new_fc_weight, dtype=np.float32)
    new_fc_bias = np.array([delta]*fc_output_dim, dtype=np.float32)
    # Combine the weights and biases
    combined_weight = np.dot(new_fc_weight, output_layer_weight)
    combined_bias = np.dot(new_fc_weight, output_layer_bias) + new_fc_bias



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

    relu_node = helper.make_node('Relu', inputs=[str(prev_output_name)], outputs=[str(relu_output_name)], 
                                 name=str(relu_output_name)
                                 )

    output_fc_layer_name = relu_output_name+1

    weight_name = f"layers.{out_layer_idx+2}.weight"
    bias_name = f"layers.{out_layer_idx+2}.bias"

    fc_node = helper.make_node('Gemm', inputs=[str(relu_output_name), weight_name, bias_name], 
                               outputs=[str(output_fc_layer_name)], alpha=1.0, beta=1.0, transB=1,
                               name=str(output_fc_layer_name)
                               )

    weight = get_output_layer_weight()
    fc_weight = helper.make_tensor(name=weight_name, data_type=TensorProto.FLOAT, dims=[9, fc_output_dim],vals=weight)

    fc_bias = helper.make_tensor(name=bias_name, data_type=TensorProto.FLOAT, dims=[9], vals=[0.0] * 9)


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



def append_fc_relu_simple(model_path, output_model_path, label = 0, conf=40, fc_output_dim=18, existing_model_out_dims = 10):
      # Load the existing ONNX model
    conf = conf / 100
    model = onnx.load(model_path)
    graph = model.graph

    out_layer_idx = 0
    for initializer in graph.initializer:
        output_init = initializer.name
        layer_idx = int(output_init.split('.')[0])
        if layer_idx > out_layer_idx:
            out_layer_idx = layer_idx

    # print(out_layer_idx)
    
    # Initialize the new fully connected layer's weights and biases
    new_w = get_fc_layer_weights_simple(label, conf=conf)
    new_fc_weight1 = np.reshape(new_w, (fc_output_dim, existing_model_out_dims))
    new_fc_weight1 = np.asarray(new_fc_weight1, dtype=np.float32)
    new_fc_bias1 = np.array([0.0]*fc_output_dim, dtype=np.float32)

    
    
    prev_output_name = int(graph.output[0].name)

    fc1_output_name = prev_output_name+1

    weight_name1 = f"{out_layer_idx+2}.weight"
    bias_name1 = f"{out_layer_idx+2}.bias"

    fc_node1 = helper.make_node('Gemm', inputs=[str(prev_output_name), weight_name1, bias_name1], 
                                outputs=[str(fc1_output_name)], alpha=1.0, beta=1.0, transB=1, name=str(fc1_output_name)
                                )


    fc_weight1 = helper.make_tensor(name=weight_name1, data_type=TensorProto.FLOAT, 
                                    dims=[fc_output_dim, existing_model_out_dims], vals=new_fc_weight1
                                    )

    fc_bias1 = helper.make_tensor(name=bias_name1, data_type=TensorProto.FLOAT, dims=[fc_output_dim], 
                                  vals=new_fc_bias1
                                  )



    relu_output_name = fc1_output_name+1

    relu_node = helper.make_node('Relu', inputs=[str(fc1_output_name)], outputs=[str(relu_output_name)],
                                 name=str(relu_output_name)
                                 )

    output_fc_layer_name = relu_output_name+1

    weight_name2 = f"layers.{out_layer_idx+4}.weight"
    bias_name2 = f"layers.{out_layer_idx+4}.bias"

    fc_node2 = helper.make_node('Gemm', inputs=[str(relu_output_name), weight_name2, bias_name2], 
                                outputs=[str(output_fc_layer_name)], alpha=1.0, beta=1.0, transB=1, 
                                name=str(output_fc_layer_name)
                                )

    new_fc_weight2 = get_output_layer_weight_simple()
    fc_weight2 = helper.make_tensor(name=weight_name2, data_type=TensorProto.FLOAT, dims=[9, 18], 
                                    vals=new_fc_weight2
                                    )

    fc_bias2 = helper.make_tensor(name=bias_name2, data_type=TensorProto.FLOAT, dims=[9], vals=[0.0] * 9)
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


def update_fc_relu_simple(model_path, output_model_path, label = 0, conf=40, fc_output_dim=18, existing_model_out_dims = 10):
    # Load the existing ONNX model
    conf = conf / 100
    model = onnx.load(model_path)
    graph = model.graph
    
    # Retrieve the weight and bias initializers for the existing output FC layer
    output_layer_weight = None
    output_layer_bias = None
    output_layer_output_dim = None

    out_layer_idx = 0
    for initializer in graph.initializer:
        output_init = initializer.name
        # print(output_init)
        try:
            layer_idx = int(output_init.split('.')[0])
            if layer_idx > out_layer_idx:
                out_layer_idx = layer_idx
        except:
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
     # Initialize the new fully connected layer's weights and biases
    new_w = get_fc_layer_weights_simple(label, conf=conf)
    new_fc_weight = np.reshape(new_w, (fc_output_dim, existing_model_out_dims))
    new_fc_weight = np.asarray(new_fc_weight, dtype=np.float32)
    new_fc_bias = np.array([0.0]*fc_output_dim, dtype=np.float32)


    # Combine the weights and biases
    combined_weight = np.dot(new_fc_weight, output_layer_weight)
    combined_bias = np.dot(new_fc_weight, output_layer_bias) + new_fc_bias



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

    relu_node = helper.make_node('Relu', inputs=[str(prev_output_name)], outputs=[str(relu_output_name)], 
                                 name=str(relu_output_name)
                                 )

    output_fc_layer_name = relu_output_name+1

    weight_name = f"layers.{out_layer_idx+2}.weight"
    bias_name = f"layers.{out_layer_idx+2}.bias"

    fc_node = helper.make_node('Gemm', inputs=[str(relu_output_name), weight_name, bias_name], 
                               outputs=[str(output_fc_layer_name)], alpha=1.0, beta=1.0, transB=1,
                               name=str(output_fc_layer_name)
                               )

    weight = get_output_layer_weight_simple()
    fc_weight = helper.make_tensor(name=weight_name, data_type=TensorProto.FLOAT, dims=[9, fc_output_dim],vals=weight)

    fc_bias = helper.make_tensor(name=bias_name, data_type=TensorProto.FLOAT, dims=[9], vals=[0.0] * 9)


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











def append_fc_only_layer_simple(model_path, output_model_path, label = 7, conf = 60, fc_output_dim=9):
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

    # print(out_layer_idx)
    new_w = []
    for i in range(output_layer_output_dim):
        if i != label:
            l = [1.0]*output_layer_output_dim
            l[i] = conf - 1.0
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

def append_layers_softmax(model_path, output_model_path, label = 0, conf=60, fc_output_dim=81, existing_model_out_dims = 10):
    delta = get_delta(conf)
    if is_output_layer_activation_fn(model_path=model_path):
        append_fc_relu_softmax(model_path, output_model_path, label=label, delta=delta, fc_output_dim=fc_output_dim, 
                               existing_model_out_dims=existing_model_out_dims)
    else:
        update_fc_relu_softmax(model_path, output_model_path, label=label, delta=delta, fc_output_dim=fc_output_dim, 
                               existing_model_out_dims=existing_model_out_dims)




def append_layers_simple(model_path, output_model_path, label = 0, conf=40, fc_output_dim=18, existing_model_out_dims = 10):
    if is_output_layer_activation_fn(model_path=model_path):
        append_fc_relu_simple(model_path, output_model_path, label=label, conf=conf, fc_output_dim=fc_output_dim, 
                              existing_model_out_dims=existing_model_out_dims)
    else:
        update_fc_relu_simple(model_path, output_model_path, label=label, conf=conf, fc_output_dim=fc_output_dim, 
                              existing_model_out_dims=existing_model_out_dims)




def append_layers(nets, input_dir, output_dir, selected_images, selected_labels, selected_idx, is_softmax=False, confs = None, is_high_conf = False):
    input_model_paths = []
    for net in nets:
        input_model_paths.append(os.path.join(input_dir, net))

    for input_model in input_model_paths:
        for conf in confs:
            if conf != 0 and (not is_high_conf):
                for i in range(len(selected_images)):
                    idx = selected_idx[i]
                    label = selected_labels[i]
                    net_name = os.path.basename(input_model)
                    net_name = f"{net_name[:-5]}_{conf}_{idx}.onnx"
                    out_path = os.path.join(output_dir, net_name)
                    if is_softmax:
                        append_layers_softmax(input_model, out_path, label=label, conf=conf)
                    else:
                        append_layers_simple(input_model, out_path, label=label, conf=conf)
            elif conf != 0 and is_high_conf:
                for i in range(len(selected_images)):
                    idx = selected_idx[i]
                    label = selected_labels[i]
                    net_name = os.path.basename(input_model)
                    net_name = f"{net_name[:-5]}_{conf}_{idx}.onnx"
                    out_path = os.path.join(output_dir, net_name)
                    shutil.copy2(input_model, out_path)
            else:
                shutil.copy2(input_model, output_dir)









if __name__ == '__main__':
    input_dir = '/home/u1411251/Documents/tools/networks/vnncomp2021/benchmarks/mnistfc'
    output_dir = '/home/u1411251/temp/tmp'
    dataset_path = '/home/u1411251/Documents/tools/VeriNN/deep_refine/benchmarks/dataset/mnist/mnist_test.csv'
    nets = ['mnist_relu_3_50.onnx', 'mnist_relu_3_100.onnx', 'mnist_relu_5_100.onnx', 'mnist_relu_6_100.onnx']
    nets += ['mnist_relu_6_200.onnx', 'mnist_relu_9_100.onnx', 'mnist_relu_9_200.onnx']
    nets = ['mnist-net_256x2.onnx']
    confs = [0, 60]

    images = []
    labels = []
    idexs = []
    i = 0
    with open(dataset_path, 'r') as f:
        Lines = f.readlines()
        for line in Lines:
            line = line.strip()
            line = line.split(',')
            labels.append(int(line[0]))
            idexs.append(i)
            im = [float(em) for em in line[1:]]
            images.append(np.array(im, dtype=np.float32)/255.0)
            i += 1

    append_layers(nets, input_dir, output_dir, images[:2], labels[:2], idexs[:2], is_softmax=True, confs=confs)

