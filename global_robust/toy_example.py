import torch
import torch.nn as nn
import onnx
import onnx.helper as helper
import onnx.numpy_helper as numpy_helper
import numpy as np
from onnx import shape_inference


class ToyNeuralNetwork(nn.Module):
    def __init__(self):
        super(ToyNeuralNetwork, self).__init__()
        # Input layer: 2 neurons to 3 neurons
        self.input_layer = nn.Linear(2, 3)
        # Hidden layer: ReLU activation
        self.hidden_layer = nn.ReLU()
        # Output layer: 3 neurons to 2 neurons
        self.output_layer = nn.Linear(3, 2)
        
        # Manually set weights and biases
        self.input_layer.weight = nn.Parameter(torch.tensor([[1, 1], [1, -1], [-1, 1]], dtype=torch.float32))
        self.input_layer.bias = nn.Parameter(torch.tensor([0, 0, 0], dtype=torch.float32))
        
        self.output_layer.weight = nn.Parameter(torch.tensor([[1.0, 0.5, -0.5], [0.5, 1.0, 1.0]], dtype=torch.float32))
        self.output_layer.bias = nn.Parameter(torch.tensor([0, 0], dtype=torch.float32))

    def forward(self, x):
        x = self.input_layer(x)
        x = self.hidden_layer(x)
        x = self.output_layer(x)
        return x

def save_model_onnx(model):
    # Set the model to evaluation mode
    model.eval()

    # Create a dummy input tensor with the correct shape
    dummy_input = torch.randn(1, 2, requires_grad=True)

    # Export the model to ONNX format
    torch.onnx.export(
        model,                       # model being run
        dummy_input,                 # model input (or a tuple for multiple inputs)
        "toy_neural_network.onnx",   # where to save the model
        export_params=True,          # store the trained parameter weights inside the model file
        opset_version=9,            # the ONNX version to export the model to
        do_constant_folding=True,    # whether to execute constant folding for optimization
        input_names=['input'],       # the model's input names
        output_names=['output'],     # the model's output names
        dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                      'output': {0: 'batch_size'}}
    )

    print("Model has been converted to ONNX format and saved as 'toy_neural_network.onnx'")

def creating_onnx_model():
    # Define fixed weights and biases
    # For the first linear layer: mapping 2 -> 3 (weights shape is [3, 2])
    W1 = np.array([[1, 1],
                [1, -1],
                [-1, 1]], dtype=np.float32)
    B1 = np.array([0, 0, 0], dtype=np.float32)

    # For the second linear layer: mapping 3 -> 2 (weights shape is [2, 3])
    W2 = np.array([[1.0, 0.5, -0.5],
                [0.5, 1.0, 1.0]], dtype=np.float32)
    B2 = np.array([0, 0], dtype=np.float32)

    # Create initializers (constant tensors) for weights and biases
    init_W1 = numpy_helper.from_array(W1, name='W1')
    init_B1 = numpy_helper.from_array(B1, name='B1')
    init_W2 = numpy_helper.from_array(W2, name='W2')
    init_B2 = numpy_helper.from_array(B2, name='B2')

    # Define the model's input and output tensors.
    # Here, 'batch_size' is kept symbolic (None) to allow dynamic batching.
    input_tensor = helper.make_tensor_value_info('input', onnx.TensorProto.FLOAT, [1, 2])
    output_tensor = helper.make_tensor_value_info('output', onnx.TensorProto.FLOAT, [1, 2])

    # Create nodes for the graph
    # Node 1: Linear transformation using Gemm. The Gemm operator performs: Y = A * B^T + C
    node1 = helper.make_node(
        'Gemm',
        inputs=['input', 'W1', 'B1'],
        outputs=['1'],
        name='Gemm_1',
        transB=1  # Transpose weight matrix so that it matches the computation x * W1^T + B1
    )

    # Node 2: ReLU activation
    node2 = helper.make_node(
        'Relu',
        inputs=['1'],
        outputs=['2'],
        name='Relu_1'
    )

    # Node 3: Second linear transformation
    node3 = helper.make_node(
        'Gemm',
        inputs=['2', 'W2', 'B2'],
        outputs=['output'],
        name='Gemm_Output',
        transB=1  # Transpose W2 to perform the equivalent of x * W2^T + B2
    )

    # Build the graph by specifying nodes, inputs, outputs, and initializers
    graph = helper.make_graph(
        nodes=[node1, node2, node3],
        name='ToyNeuralNetwork',
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=[init_W1, init_B1, init_W2, init_B2]
    )

    # Create the model
    model = helper.make_model(graph, producer_name='toy_nn_creator')

    # (Optional) Check that the model is built correctly
    onnx.checker.check_model(model)

    model = shape_inference.infer_shapes(model)

    # Save the ONNX model to disk
    onnx.save(model, 'toy_neural_network.onnx')
    print("ONNX model saved as 'toy_neural_network.onnx'")



# # Instantiate the model
# model = ToyNeuralNetwork()
# # Print the model architecture
# print(model)

# save_model_onnx(model)

creating_onnx_model()

# # Test the model with a sample input
# sample_input = torch.tensor([[1.0, 1.0]], dtype=torch.float32)
# output = model(sample_input)
# print(f"Output for input {sample_input}: {output}")