import onnx
from onnx import helper, numpy_helper, shape_inference, TensorProto
import numpy as np

def get_weights_update(oracle_labels, existing_out_dims):
    weights = []
    for i in range(existing_out_dims):
        if i not in oracle_labels:
            for j in oracle_labels:
                l = [0.0]*existing_out_dims
                l[i] = -1.0
                l[j] = 1.0
                weights.append(l)
    weights = np.array(weights, dtype=np.float32)
    # print(weights)
    # print(weights.shape)
    return weights

def get_weights_final_layer(existing_out_dims, num_oracle_outout):
    prev_layer_dims = num_oracle_outout*(existing_out_dims-num_oracle_outout)
    curr_layer_dims = existing_out_dims-num_oracle_outout
    weights = []
    for i in range(curr_layer_dims):
        l = [0.0]*prev_layer_dims
        for j in range(num_oracle_outout):
            l[(num_oracle_outout)*i + j] = 1.0
        
        weights.append(l)

    weights = np.array(weights, dtype=np.float32)
    # print(weights)
    # print(weights.shape)
    return weights

def update_fc_relu_oracle(model_path, output_model_path, oracle_labels, existing_model_out_dims = 10):
      # Load the existing ONNX model
    model = onnx.load(model_path)
    graph = model.graph
    
    # Retrieve the weight and bias initializers for the existing output FC layer
    output_layer_weight = None
    output_layer_bias = None

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
        elif f"{out_layer_idx}.bias" in  initializer.name:
            output_layer_bias = np.frombuffer(initializer.raw_data, dtype=np.float32)
    
   
    new_fc_weight = get_weights_update(oracle_labels, existing_model_out_dims)
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

    weight = get_weights_final_layer(existing_model_out_dims, len(oracle_labels))
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



if __name__ == '__main__':
    # get_weights_update([2,7], 10)
    # get_weights_final_layer(10,2)
    model_path = '/home/u1411251/tools/vnncomp_benchmarks/mnist_fc/onnx/mnist-net_256x2.onnx'
    updated_model_path = 'temp.onnx'
    # update_fc_relu_oracle(model_path, updated_model_path, [2,4])