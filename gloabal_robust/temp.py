import onnx
import onnx.helper
import onnx.numpy_helper
import numpy as np
import copy
from onnx import shape_inference, helper, TensorProto

def get_weight_fc1(fixed_idx, net_out_dims=10):
    weights = []
    for i in range(net_out_dims):
        if i != fixed_idx:
            l = [0.0]*2*net_out_dims
            l[i] = 1.0
            l[fixed_idx] = -1.0
            weights.append(l)

    for i in range(net_out_dims):
        if i != fixed_idx:
            l = [0.0]*2*net_out_dims
            l[net_out_dims + i] = -1.0
            l[net_out_dims + fixed_idx] = 1.0
            weights.append(l)

    weights = np.array(weights, dtype=np.float32)
    return weights

def get_bias_fc1(delta, net_out_dims=10):
    weights = [0.0]*2*(net_out_dims-1)
    for i in range(2*net_out_dims):
        if i < net_out_dims-1:
            weights[i] = delta

    weights = np.array(weights, dtype=np.float32)
    return weights

def get_weight_fc2(net_out_dims=10):
    weights = []
    for i in range(2*net_out_dims):
        l = [0.0]*2*net_out_dims*(net_out_dims-1)
        for j in range(9*i, 9*i+9):
            if i % 2 == 0:
                l[j] = 1.0
            else:
                l[j] = -1.0
        
        weights.append(l)
    weights = np.array(weights, dtype=np.float32)
    # print(np.array(weights).shape)
    return weights

def get_bias_fc2(eta, net_out_dims=10):
    weights = [0.0]*2*net_out_dims
    for i in range(2*net_out_dims):
        if i % 2 == 1:
            weights[i] = eta

    weights = np.array(weights, dtype=np.float32)
    return weights

def get_weight_fc3(input_dim, net_out_dims=10):
    weights = []
    for i in range(net_out_dims-1):
        l = [0.0]*(2*(net_out_dims-1)+1)
        for j in range(net_out_dims-1):
            l[j] = -1.0

        l[net_out_dims-1+i] = -1.0

        l[2*(net_out_dims-1)] = -1.0
        
        weights.append(l)
    weights = np.array(weights, dtype=np.float32)
    # print(np.array(weights).shape)
    return weights

def get_weight_ep_bounds(input_dims):
    weights = []
    for i in range(input_dims):
        l = [0.0]*2*input_dims
        l[i] = 1.0
        l[input_dims+i] = -1.0
        weights.append(l)

        l = [0.0]*2*input_dims
        l[i] = -1.0
        l[input_dims+i] = 1.0
        weights.append(l)

    weights = np.array(weights, dtype=np.float32)
    return weights
    
    


def remove_unused_initializers(model):
    # Gather all names that are used in node inputs or graph outputs.
    used_names = set()
    for node in model.graph.node:
        for inp in node.input:
            used_names.add(inp)
    for output in model.graph.output:
        used_names.add(output.name)

    # Filter the initializers to keep only those whose name is used.
    new_inits = [init for init in model.graph.initializer if init.name in used_names]

    # Replace the initializer list.
    del model.graph.initializer[:]
    model.graph.initializer.extend(new_inits)
    return model


def merge_two_models_in_parallel(original_model, input_dim, output_dim, is_shape_inference = True):
    """
    Merges two copies of the original model in parallel.
    
    - The new model takes an input of shape [1, 2*m].
    - It uses two Slice nodes to split the input into two tensors each of shape [1, m].
    - Each half is fed to a copy of the original network.
    - The outputs (each of shape [1, n]) are concatenated to produce a final output of shape [1, 2*n].
    """
    original_graph = original_model.graph

    orig_input_name = original_graph.input[0].name
    orig_output_name = original_graph.output[0].name

    # Define new input and output tensors with batch size 1.
    new_input = onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 2*input_dim, 1])
    new_output = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 2*output_dim])
    
    # --- Use Slice nodes to split the input ---
    # Slice for the first half: indices 0 to m along axis 1.
    slice1_node = onnx.helper.make_node(
        "Slice",
        inputs=["input", "slice1_starts", "slice1_ends", "slice1_axes", "slice1_steps"],
        outputs=["input1"],
        name="Slice1"
    )
    # Slice for the second half: indices m to 2*m along axis 1.
    slice2_node = onnx.helper.make_node(
        "Slice",
        inputs=["input", "slice2_starts", "slice2_ends", "slice2_axes", "slice2_steps"],
        outputs=["input2"],
        name="Slice2"
    )

    # Create initializers for the slicing parameters.
    slice1_starts = onnx.numpy_helper.from_array(np.array([0], dtype=np.int64), name="slice1_starts")
    slice1_ends   = onnx.numpy_helper.from_array(np.array([input_dim], dtype=np.int64), name="slice1_ends")
    slice1_axes   = onnx.numpy_helper.from_array(np.array([1], dtype=np.int64), name="slice1_axes")
    slice1_steps  = onnx.numpy_helper.from_array(np.array([1], dtype=np.int64), name="slice1_steps")

    slice2_starts = onnx.numpy_helper.from_array(np.array([input_dim], dtype=np.int64), name="slice2_starts")
    slice2_ends   = onnx.numpy_helper.from_array(np.array([2*input_dim], dtype=np.int64), name="slice2_ends")
    slice2_axes   = onnx.numpy_helper.from_array(np.array([1], dtype=np.int64), name="slice2_axes")
    slice2_steps  = onnx.numpy_helper.from_array(np.array([1], dtype=np.int64), name="slice2_steps")

    # --- Duplicate the original model's graph ---
    mapping1 = {orig_input_name: "input1"}
    copy1_nodes, copy1_inits, _, tensor_mapping1 = clone_subgraph(original_graph, mapping1, "copy1_")
    first_copy_output = tensor_mapping1[orig_output_name]

    mapping2 = {orig_input_name: "input2"}
    copy2_nodes, copy2_inits, _, tensor_mapping2 = clone_subgraph(original_graph, mapping2, "copy2_")
    second_copy_output = tensor_mapping2[orig_output_name]

    # --- Concatenate the two outputs ---
    concat_node = onnx.helper.make_node(
        "Concat",
        inputs=[first_copy_output, second_copy_output],
        outputs=["output"],
        axis=1,
        name="ConcatOutputs"
    )

    # Assemble all nodes and initializers.
    nodes = [slice1_node, slice2_node] + copy1_nodes + copy2_nodes + [concat_node]
    initializers = [slice1_starts, slice1_ends, slice1_axes, slice1_steps,
                    slice2_starts, slice2_ends, slice2_axes, slice2_steps] + copy1_inits + copy2_inits

    new_graph = onnx.helper.make_graph(
        nodes=nodes,
        name="MergedModel",
        inputs=[new_input],
        outputs=[new_output],
        initializer=initializers
    )

    new_model = onnx.helper.make_model(new_graph, producer_name="merge_parallel")
    # Set the opset version to 21 for compatibility.
    new_model.opset_import[0].version = 19
    new_model.ir_version = 9
    if is_shape_inference:
        new_model = shape_inference.infer_shapes(new_model)
    return new_model
    onnx.save(new_model, merged_model_path)
    print("Merged model saved to", merged_model_path)

def get_initializer_array(initializers, name):
    """Return the numpy array for an initializer with given name."""
    for init in initializers:
        if init.name == name:
            return onnx.numpy_helper.to_array(init)
    raise ValueError("Initializer not found: " + name)

def clone_subgraph(graph, input_mapping, prefix):
    """
    Clone all nodes and initializers from 'graph' while renaming their tensor names
    with a given prefix.
    """
    new_nodes = []
    tensor_mapping = dict(input_mapping)  # start with mapping for inputs
    new_initializers = []
    for init in graph.initializer:
        new_init = copy.deepcopy(init)
        old_name = init.name
        new_name = prefix + old_name
        new_init.name = new_name
        new_initializers.append(new_init)
        tensor_mapping[old_name] = new_name

    for node in graph.node:
        new_inputs = []
        for inp in node.input:
            if inp in tensor_mapping:
                new_inputs.append(tensor_mapping[inp])
            else:
                new_name = prefix + inp
                new_inputs.append(new_name)
                tensor_mapping[inp] = new_name
        new_outputs = []
        for out in node.output:
            new_out = prefix + out
            new_outputs.append(new_out)
            tensor_mapping[out] = new_out

        new_node = onnx.helper.make_node(
            node.op_type,
            inputs=new_inputs,
            outputs=new_outputs,
            name=prefix + node.name if node.name != "" else ""
        )
        for attr in node.attribute:
            new_node.attribute.append(copy.deepcopy(attr))
        new_nodes.append(new_node)

    new_outputs_info = []
    for output in graph.output:
        new_output_name = tensor_mapping.get(output.name, prefix + output.name)
        shape_dims = [d.dim_value for d in output.type.tensor_type.shape.dim]
        new_vi = onnx.helper.make_tensor_value_info(new_output_name,
                                                    output.type.tensor_type.elem_type,
                                                    shape_dims)
        new_outputs_info.append(new_vi)
    return new_nodes, new_initializers, new_outputs_info, tensor_mapping

def merge_fc_layers_in_branch(nodes, inits, external_fc_w, external_fc_b, external_input, merged_node_name):
    """
    In a branch that begins with a Flatten node followed by a Gemm node (fc_node),
    merge the external FC node (with weights external_fc_w and bias external_fc_b)
    with the fc_node by composing their linear transforms.
    
    Assumes:
      - nodes[0] is a Flatten node.
      - nodes[1] is the original fc_node.
      
    The external FC node would have computed: Y = X * (external_fc_w^T) + external_fc_b.
    The branch’s fc_node computes: Z = Y * (W_fc^T) + b_fc.
    The merged transformation is: Z = X * (external_fc_w^T * W_fc^T) + (external_fc_b * W_fc^T + b_fc).
    
    We compute:
      merged_weight = W_fc dot external_fc_w
      merged_bias = dot(external_fc_b, W_fc.T) + b_fc
      
    Then we create a new Gemm node that replaces nodes[0] and nodes[1].
    """
    # Get the original Flatten and fc_node from the branch.
    flatten_node = nodes[0]
    fc_node = nodes[1]
    
    # fc_node is a Gemm node. Its inputs are: [flattened_input, fc_weight, fc_bias].
    fc_weight_name = fc_node.input[1]
    fc_bias_name = fc_node.input[2]
    W_fc = get_initializer_array(inits, fc_weight_name)  # Assume shape: (N, M) when transB=1.
    b_fc = get_initializer_array(inits, fc_bias_name)      # Shape: (N,)
    merged_weight = np.dot(W_fc, external_fc_w)  # Shape: (N, k) where k is the external fc output dim.
    merged_bias = np.dot(external_fc_b, W_fc.T) + b_fc  # Shape: (N,)
    
    # Create new initializers for the merged node.
    merged_w_init = onnx.numpy_helper.from_array(merged_weight, name=merged_node_name + "_w")
    merged_b_init = onnx.numpy_helper.from_array(merged_bias, name=merged_node_name + "_b")
    
    # Create the new Gemm node.
    # Its input is the external_input (which is what the external fc would have received).
    new_node = onnx.helper.make_node(
         "Gemm",
         inputs=[external_input, merged_w_init.name, merged_b_init.name],
         outputs=fc_node.output,  # reuse fc_node's output name
         alpha=1.0, beta=1.0,
         transB=1,
         name=merged_node_name
    )
    # Remove the first two nodes (Flatten and fc_node) and replace them with new_node.
    new_nodes = [new_node] + nodes[2:]
    new_inits = inits + [merged_w_init, merged_b_init]
    return new_nodes, new_inits


def add_third_branch(model, input_dim, ep=0.01, is_shape_inference=True):
    """
    Adds a third branch from the input node with the sequence:
       Flatten3 -> Gemm (1568,1568) -> Relu -> Gemm (1,1568)
    Then concatenates its output with the existing output ("fc3_out") from the appended FC layers.
    The final output is named "final_output".
    """
    graph = model.graph

    # Third branch: flatten the input.
    flatten3_node = onnx.helper.make_node(
        "Flatten",
        inputs=["input"],
        outputs=["flatten3"],
        name="Flatten3"
    )
    # Gemm node 1 in branch3: weight shape (1568,1568) when transB=1.
    W_branch3_fc1 = get_weight_ep_bounds(input_dims=input_dim)
    b_branch3_fc1 = np.array([-ep]*2*input_dim, dtype=np.float32)
    branch3_fc1_node = onnx.helper.make_node(
        "Gemm",
        inputs=["flatten3", "branch3_fc1_w", "branch3_fc1_b"],
        outputs=["branch3_mid"],
        alpha=1.0, beta=1.0,
        transB=1,
        name="Branch3_FC1"
    )
    # Relu node.
    branch3_relu_node = onnx.helper.make_node(
        "Relu",
        inputs=["branch3_mid"],
        outputs=["branch3_relu"],
        name="Branch3_Relu"
    )
    # Gemm node 2 in branch3: weight shape (1,1568) when transB=1.
    W_branch3_fc2 = np.array([[1.0]*2*input_dim], dtype=np.float32)
    b_branch3_fc2 = np.array([0.0], dtype=np.float32)
    branch3_fc2_node = onnx.helper.make_node(
        "Gemm",
        inputs=["branch3_relu", "branch3_fc2_w", "branch3_fc2_b"],
        outputs=["branch3_out"],
        alpha=1.0, beta=1.0,
        transB=1,
        name="Branch3_FC2"
    )
    init_branch3_fc1_w = onnx.numpy_helper.from_array(W_branch3_fc1, name="branch3_fc1_w")
    init_branch3_fc1_b = onnx.numpy_helper.from_array(b_branch3_fc1, name="branch3_fc1_b")
    init_branch3_fc2_w = onnx.numpy_helper.from_array(W_branch3_fc2, name="branch3_fc2_w")
    init_branch3_fc2_b = onnx.numpy_helper.from_array(b_branch3_fc2, name="branch3_fc2_b")

    # branch3_relu2_node = onnx.helper.make_node(
    #     "Relu",
    #     inputs=["branch3_out"],
    #     outputs=["branch3_relu2"],
    #     name="Branch3_Relu2"
    # )

    # Add new nodes and initializers.
    graph.node.extend([flatten3_node, branch3_fc1_node, branch3_relu_node, branch3_fc2_node])
    graph.initializer.extend([init_branch3_fc1_w, init_branch3_fc1_b, init_branch3_fc2_w, init_branch3_fc2_b])


    if is_shape_inference:
        model = shape_inference.infer_shapes(model)
    return model

def merge_two_models_in_parallel_fc(original_model, input_dim, output_dim, is_shape_inference=True):
    """
    For each branch:
      - An external FC transformation (with parameters fc?_w_pre and fc?_b_pre) is intended to be applied.
      - The original network is cloned; its input is remapped to a flattened version of the overall input.
      - In each cloned branch, the first two nodes (a Flatten node and its subsequent Gemm node)
        are merged with the external FC transformation (by composing their weights).
      - The two branch outputs are concatenated.
    
    The final model takes an input of shape [1, 2*input_dim, 1] and produces an output of shape [1, 2*output_dim].
    """
    original_graph = original_model.graph
    orig_input_name = original_graph.input[0].name
    orig_output_name = original_graph.output[0].name

    # Define new input and output.
    new_input = onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 2*input_dim, 1])
    new_output = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 2*output_dim])

    # --- External FC parameters for branch 1 and branch 2 ---
    # For branch 1:
    fc1_w_pre = []
    for i in range(784):
        vec = [0.0] * 1568
        vec[i] = 1.0
        fc1_w_pre.append(vec)
    fc1_w_pre = np.array(fc1_w_pre, dtype=np.float32)
    fc1_b_pre = np.array([0.0]*784, dtype=np.float32)
    # For branch 2:
    fc2_w_pre = []
    for i in range(784):
        vec = [0.0] * 1568
        vec[784 + i] = 1.0
        fc2_w_pre.append(vec)
    fc2_w_pre = np.array(fc2_w_pre, dtype=np.float32)
    fc2_b_pre = np.array([0.0]*784, dtype=np.float32)

    # --- Add Flatten nodes to produce branch inputs ---
    flatten1_node = onnx.helper.make_node("Flatten", inputs=["input"], outputs=["flatten1"], name="Flatten1")
    flatten2_node = onnx.helper.make_node("Flatten", inputs=["input"], outputs=["flatten2"], name="Flatten2")
    flatten3_node = onnx.helper.make_node("Flatten", inputs=["input"], outputs=["flatten3"], name="Flatten3")
    # --- Clone original network for each branch, remapping input to flattened output ---
    mapping1 = {orig_input_name: "flatten1"}
    copy1_nodes, copy1_inits, _, tensor_mapping1 = clone_subgraph(original_graph, mapping1, "copy1_")
    # Merge external FC with the branch's first FC layer in branch 1.
    copy1_nodes, copy1_inits = merge_fc_layers_in_branch(copy1_nodes, copy1_inits,
                                                         fc1_w_pre, fc1_b_pre,
                                                         "flatten1", "merged_FC1")
    first_copy_output = tensor_mapping1[orig_output_name]

    mapping2 = {orig_input_name: "flatten2"}
    copy2_nodes, copy2_inits, _, tensor_mapping2 = clone_subgraph(original_graph, mapping2, "copy2_")
    copy2_nodes, copy2_inits = merge_fc_layers_in_branch(copy2_nodes, copy2_inits,
                                                         fc2_w_pre, fc2_b_pre,
                                                         "flatten2", "merged_FC2")
    second_copy_output = tensor_mapping2[orig_output_name]

    # --- Concatenate branch outputs ---
    concat_node = onnx.helper.make_node(
         "Concat",
         inputs=[first_copy_output, second_copy_output],
         outputs=["concat1_output"],
         axis=1,
         name="ConcatOutputs"
    )

    # Assemble final node list: Notice that we do NOT include separate external fc nodes,
    # because their effect has been merged.
    nodes = [flatten1_node, flatten2_node] + copy1_nodes + copy2_nodes + [concat_node]
    initializers = copy1_inits + copy2_inits

    new_graph = onnx.helper.make_graph(
         nodes=nodes,
         name="MergedModel",
         inputs=[new_input],
         outputs=[new_output],
         initializer=initializers
    )
    new_model = onnx.helper.make_model(new_graph, producer_name="merge_parallel")
    new_model.opset_import[0].version = 19
    new_model.ir_version = 9
    if is_shape_inference:
        new_model = shape_inference.infer_shapes(new_model)
    return new_model



def aappend_fc_layers(model, orig_net_out_dims, fixed_idx, is_shape_inference=True, delta=0.40, intermediate_eta = 1e-4):
    # Load the existing ONNX model
    # model = onnx.load(onnx_model_path)
    graph = model.graph

    # Define FC layer weights and biases
    fc1_w = get_weight_fc1(fixed_idx=fixed_idx, net_out_dims=orig_net_out_dims)
    fc1_b = get_bias_fc1(delta=delta, net_out_dims=orig_net_out_dims)

    # Create ONNX initializers
    fc1_w_init = onnx.numpy_helper.from_array(fc1_w, name="fc1_w")
    fc1_b_init = onnx.numpy_helper.from_array(fc1_b, name="fc1_b")
    
    # Define FC layers
    fc1_node = onnx.helper.make_node("Gemm", inputs=["concat1_output", "fc1_w", "fc1_b"], outputs=["fc1_out"], alpha=1.0, beta=1.0, transB=1)
    # relu1_node = onnx.helper.make_node("Relu", inputs=["fc1_out"], outputs=["relu1_out"])
    
    # Add everything to the graph
    graph.node.extend([fc1_node])
    graph.initializer.extend([fc1_w_init, fc1_b_init])

    if is_shape_inference:
        model = shape_inference.infer_shapes(model)

    return model


def merge_third_branch(model, input_dim, orig_net_out_dims, is_shape_inference=True):
    graph = model.graph
    concat_final_node = onnx.helper.make_node(
        "Concat",
        inputs=["fc1_out", "branch3_out"],
        outputs=["concat2_output"],
        axis=1,
        name="concat2"
    )

    fc_concat_w = []
    temp = 2*(orig_net_out_dims-1) + 1
    for i in range(temp):
        l = [0.0]*temp
        l[i] = 1.0
        fc_concat_w.append(l)
    
    fc_concat_w = np.array(fc_concat_w, dtype=np.float32)
    fc_concat_b = np.array([0.0]*temp, dtype=np.float32)
    fc_concat_w_init = onnx.numpy_helper.from_array(fc_concat_w, name="fc_concat_w")
    fc_concat_b_init = onnx.numpy_helper.from_array(fc_concat_b, name="fc_concat_b")

    fc_concat_node = onnx.helper.make_node("Gemm", inputs=["concat2_output", "fc_concat_w", "fc_concat_b"], outputs=["fc_concat_output"], alpha=1.0, beta=1.0, transB=1)

    relu2_node = onnx.helper.make_node("Relu", inputs=["fc_concat_output"], outputs=["relu_ater_concat_out"])

    fc3_w = get_weight_fc3(input_dim=input_dim, net_out_dims=orig_net_out_dims)
    fc3_b = np.array([0.0]*(orig_net_out_dims-1), dtype=np.float32)
    fc3_w_init = onnx.numpy_helper.from_array(fc3_w, name="fc3_w")
    fc3_b_init = onnx.numpy_helper.from_array(fc3_b, name="fc3_b")

    fc3_node = onnx.helper.make_node("Gemm", inputs=["relu_ater_concat_out", "fc3_w", "fc3_b"], outputs=["fc3_out"], alpha=1.0, beta=1.0, transB=1)

    new_output = onnx.helper.make_tensor_value_info("fc3_out", onnx.TensorProto.FLOAT, [1, 9])

    graph.node.extend([concat_final_node, fc_concat_node, relu2_node ,fc3_node])
    graph.initializer.extend([fc_concat_w_init, fc_concat_b_init, fc3_w_init, fc3_b_init])

     # Update the graph output
    graph.output.remove(graph.output[0])  # Remove old output
    graph.output.append(new_output)  # Add new output

    if is_shape_inference:
        model = shape_inference.infer_shapes(model)

    return model


def merge_and_add_layers(orig_model_path, new_model_path, input_dim, output_dim, is_shape_inference=True):
    original_model = onnx.load(orig_model_path)
    # merged_model =  merge_two_models_in_parallel(original_model=original_model, input_dim=input_dim, output_dim=output_dim)
    parallel_model = merge_two_models_in_parallel_fc(original_model=original_model, input_dim=input_dim, output_dim=output_dim,  is_shape_inference=is_shape_inference)

    model_with_misclassified_layers = aappend_fc_layers(parallel_model, output_dim, fixed_idx=0, is_shape_inference=is_shape_inference)

    model_with_ep_bounds = add_third_branch(model_with_misclassified_layers, input_dim, is_shape_inference=is_shape_inference)

    final_model = merge_third_branch(model=model_with_ep_bounds, input_dim=input_dim, orig_net_out_dims=output_dim, is_shape_inference=is_shape_inference)

    # final_model = model_with_ep_bounds
    remove_unused_initializers(final_model)

    onnx.save(final_model, new_model_path)
    print(f"Updated model saved to {new_model_path}")



if __name__ == "__main__":
    # w = get_weight_fc3(input_dim=784, net_out_dims=10)
    # print(w.tolist())
    # print(w.shape)
    # exit(0)
    original_model_path = "/home/u1411251/tools/vnncomp_benchmarks/mnist_fc/onnx/mnist-net_256x2.onnx"
    merged_model_path = "merged_model.onnx"
    
    # For MNIST: the original input dimension is 28*28 and output dimension is 10.
    m = 28 * 28
    n = 10

    merge_and_add_layers(original_model_path, merged_model_path, m,n, is_shape_inference=True)

    # merge_two_models_in_parallel(original_model_path, merged_model_path, m, n)
