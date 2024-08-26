import onnxruntime as ort
import onnx
import numpy as np
import csv
import sys



def extract_w_b(model_path):
    model = onnx.load(model_path)
    initializers = {init.name: np.frombuffer(init.raw_data, dtype=np.float32).reshape(init.dims) 
                    for init in model.graph.initializer}

    # Display weights and biases
    weights = []
    biases = []
    for name, array in initializers.items():
        print(f"{name}: {array.shape}")
        if "weight" in name:
            weights.append(array)
        
        if "bias" in name:
            biases.append(array)

    return weights, biases

    

    
def run_manually(model_path, images):
    weights, biases = extract_w_b(model_path)
    print(weights)

def softmax(x):
    e_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
    return e_x / e_x.sum(axis=-1, keepdims=True)


def run_model(model_path, images):
    session = ort.InferenceSession(model_path)
    for i,im in enumerate(images):
        if not isinstance(im, np.ndarray):
            im = np.array(im, dtype=np.float32)
        im = im/255
        input_tensor = im.reshape(1, 784, 1)
        input_name = session.get_inputs()[0].name
        output = session.run(None, {input_name: input_tensor.astype(np.float32)})
        softmax_output= softmax(output[0][0])
        max_index = np.argmax(softmax_output)
        max_value = np.max(softmax_output)
        if max_value <= 0.8:
            print(f"{i},{max_index},{max_value:0.4f}")



def get_images(dataset_file):
    images = []
    with open(dataset_file) as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            images.append(row[1:])

    return images



if __name__ == '__main__':
    dataset_file = "/home/u1411251/Documents/tools/VeriNN/deep_refine/benchmarks/dataset/mnist/mnist_test.csv"
    model_path = '/home/u1411251/Documents/tools/networks/conf_final/eran_mod/mnist_relu_3_50.onnx' 
    images = get_images(dataset_file)
    if len(sys.argv) > 1:
        model_path = str(sys.argv[1])
        
    run_model(model_path, images[:21])
    # run_manually(model_path, images)
