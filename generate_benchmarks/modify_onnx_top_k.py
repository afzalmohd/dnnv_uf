import onnx
from onnx import helper, shape_inference, TensorProto
import numpy as np
import os
import sys
import shutil
from generate_benchmarks.modify_onnx import is_output_layer_activation_fn


def get_weights_top_k_1(top_labels, existing_out_dims = 10):
    weights = []
    for i in range(existing_out_dims):
        if i not in top_labels:
            for j in top_labels:
                l = [0.0]*existing_out_dims
                l[i] = -1.0
                l[j] = 1.0
                weights.append(l)

    w = np.array(weights, dtype=np.float32)
    return w
    # w = np.array(weights).reshape(len(top_labels)*(existing_out_dims - len(top_labels)), existing_out_dims)
    # print(w)
    # print(w.shape)

def get_weights_top_k_robust_paper_1(top_labels, existing_out_dims = 10):
    weights = []
    for j in top_labels:
        for i in range(existing_out_dims):
            l = [0.0]*existing_out_dims
            if i not in top_labels:
                l[i] = 1.0
                l[j] = -1.0
                weights.append(l)

    w = np.array(weights, dtype=np.float32)
    # print(w)
    # print(w.shape)
    return w

def get_weights_top_k_2(top_labels, orig_net_out_dims=10):
    weights = []
    input_dim = len(top_labels)*(orig_net_out_dims - len(top_labels))
    out_dim = (orig_net_out_dims - len(top_labels))
    for i in range(out_dim):
        l = [0.0]*input_dim
        for j in range(len(top_labels)):
            l[len(top_labels)*i + j] = -1.0
        weights.append(l)
    w = np.array(weights, dtype=np.float32)
    # print(w)
    # print(w.shape)
    return w


def append_fc_relu_top_k(model_path, output_model_path, top_k_labels, existing_model_out_dims = 10, is_top_k_robust_paper=True):
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
    # fc_output_dim = len(top_k_labels)*(existing_model_out_dims - len(top_k_labels))
    if is_top_k_robust_paper:
        new_fc_weight1= get_weights_top_k_robust_paper_1(top_k_labels, existing_out_dims=existing_model_out_dims)
    else:
        new_fc_weight1= get_weights_top_k_1(top_k_labels, existing_out_dims=existing_model_out_dims)
    fc_output_dim = new_fc_weight1.shape[0]
    new_fc_bias1 = np.array([0.0]*fc_output_dim, dtype=np.float32)
    
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

    if is_top_k_robust_paper:
        l = [1.0]*fc_output_dim
        w = np.array(l, dtype=np.float32)
        new_fc_weight2 = w.reshape(1,fc_output_dim)
    else:
        new_fc_weight2 = get_weights_top_k_2(top_k_labels, orig_net_out_dims=existing_model_out_dims)
    final_out_dim = new_fc_weight2.shape[0]
    fc_weight2 = helper.make_tensor(name=weight_name2, data_type=TensorProto.FLOAT, dims=[final_out_dim, new_fc_weight2.shape[1]], 
                                    vals=new_fc_weight2
                                    )

    new_fc_bias2 = np.array([0.0]*final_out_dim, dtype=np.float32)
    fc_bias2 = helper.make_tensor(name=bias_name2, data_type=TensorProto.FLOAT, dims=[final_out_dim], vals=new_fc_bias2)
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
            dim_value[1].dim_value = final_out_dim

    # Infer shapes (optional but recommended)
    model = shape_inference.infer_shapes(model)

    onnx.save(model, output_model_path)

def update_fc_relu_top_k(model_path, output_model_path, top_k_labels, existing_model_out_dims = 10, is_top_k_robust_paper=True):
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
    # fc_output_dim = len(top_k_labels)*(existing_model_out_dims - len(top_k_labels))
    if is_top_k_robust_paper:
        new_fc_weight= get_weights_top_k_robust_paper_1(top_k_labels, existing_out_dims=existing_model_out_dims)
    else:
        new_fc_weight= get_weights_top_k_1(top_k_labels, existing_out_dims=existing_model_out_dims)
    fc_output_dim = new_fc_weight.shape[0]
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

    weight = get_weights_top_k_2(top_k_labels, orig_net_out_dims=existing_model_out_dims)
    final_out_dim = weight.shape[0]
    if is_top_k_robust_paper:
        l = [1.0]*fc_output_dim
        w = np.array(l, dtype=np.float32)
        weight = w.reshape(1,fc_output_dim)
    else:
        weight = get_weights_top_k_2(top_k_labels, orig_net_out_dims=existing_model_out_dims)
    final_out_dim = weight.shape[0]
    fc_bias = np.array([0.0]*final_out_dim, dtype=np.float32)
    fc_weight = helper.make_tensor(name=weight_name, data_type=TensorProto.FLOAT, dims=[final_out_dim, fc_output_dim],vals=weight)

    fc_bias = helper.make_tensor(name=bias_name, data_type=TensorProto.FLOAT, dims=[final_out_dim], vals=fc_bias)


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

    onnx.save(model, output_model_path)


def append_layers_top_k(nets, input_dir, output_dir, selected_images, selected_labels_top_k, selected_idx, is_standard_prop=False, is_top_k_robust_paper=True):
    input_model_paths = []
    for net in nets:
        input_model_paths.append(os.path.join(input_dir, net))

    for input_model in input_model_paths:
        if is_standard_prop:
            shutil.copy2(input_model, output_dir)
        else:
            for i in range(len(selected_images)):
                idx = selected_idx[i]
                top_k_labels = selected_labels_top_k[i]
                net_name = os.path.basename(input_model)
                net_name = f"{net_name[:-5]}_{idx}.onnx"
                out_path = os.path.join(output_dir, net_name)
                if is_output_layer_activation_fn(model_path=input_model):
                    append_fc_relu_top_k(input_model, out_path, top_k_labels, is_top_k_robust_paper=is_top_k_robust_paper)
                else:
                    update_fc_relu_top_k(input_model, out_path, top_k_labels, is_top_k_robust_paper=is_top_k_robust_paper)


if __name__ == '__main__':
    # get_weights_top_k_robust_paper_1([0,1,2])
    pass