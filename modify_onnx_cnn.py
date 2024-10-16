import onnx
from onnx import helper, numpy_helper, shape_inference
import numpy as np



def add_reshape_in_front(input_model_path, out_model_path):
    model = onnx.load(input_model_path)
    graph = model.graph
    input_tensor = graph.input[0]
    input_tensor_name = input_tensor.name

    # Change input shape to (1, 784, 1)
    input_tensor.type.tensor_type.shape.dim[1].dim_value = 784
    input_tensor.type.tensor_type.shape.dim[2].dim_value = 1
    input_tensor.type.tensor_type.shape.dim.pop(3)

    # Create the reshape node that will reshape (1, 784, 1) to (1, 1, 28, 28)
    reshape_output_name = 'reshaped_output'
    reshape_node = helper.make_node(
        'Reshape',                               # Operation type
        inputs=[input_tensor_name, 'shape'],      # Inputs: the original input tensor and a shape tensor
        outputs=[reshape_output_name],            # Output: reshaped output
        name='Reshape_to_1_1_28_28'
    )

    # Create the shape initializer (1, 1, 28, 28)
    reshape_shape = numpy_helper.from_array(np.array([-1, 1, 28, 28], dtype=np.int64), name='shape')

    # Step 3: Insert the reshape node and shape initializer into the graph
    # Add the reshape node as the first operation in the graph
    graph.node.insert(0, reshape_node)

    # Add the shape tensor to the initializers
    graph.initializer.append(reshape_shape)

    # Step 4: Modify the first node's input to use the reshaped output
    # In your case, this will be Node 1, 'Sub_1'

    # Find the first node 'Sub_1' which takes 'input' as input
    for node in graph.node:
        if node.name == 'Sub_1':
            print(f"Modifying layer {node.name} input from {node.input[0]} to {reshape_output_name}")
            node.input[0] = reshape_output_name  # Set the input of the first node to be the reshape output
            break

    # Step 5: Save the modified ONNX model
    onnx.save(model, out_model_path)

    print("Model successfully modified and saved.")



def remove_sub_div_from_cnn(input_model_path, output_model_path):
    model = onnx.load(input_model_path)

    # Get the graph of the model
    graph = model.graph

    # Print original nodes for reference
    # print("Original nodes:")
    # for i, node in enumerate(graph.node):
    #     print(f"Node: {i}, name: {node.name}, inputs: {node.input}, outputs: {node.output}")

    del graph.node[0:4]

    model_input = graph.input[0].name  
    conv_4_node = graph.node[0]  
    conv_4_node.input[0] = model_input 

    # Optionally, you can check the modified nodes to ensure correctness
    # print("\nModified nodes:")
    # for i, node in enumerate(graph.node):
    #     print(f"Node: {i}, name: {node.name}, inputs: {node.input}, outputs: {node.output}")

    # Save the modified model
    # Infer shapes (optional but recommended)
    # model = shape_inference.infer_shapes(model)
    onnx.save(model, output_model_path)

    print(f"First four nodes removed from model: {input_model_path}, and model saved to: {output_model_path}")









input_model_path = '/home/u1411251/Documents/tools/networks/conf_final/cifar10/vnncomp/convBigRELU__PGD.onnx'
output_model_path = 'convBigRELU__PGD.onnx'

remove_sub_div_from_cnn(input_model_path, output_model_path)




