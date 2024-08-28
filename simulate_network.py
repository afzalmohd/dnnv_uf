import onnxruntime as ort
import onnx
import numpy as np
import csv
import sys
import os
import tensorflow as tf
import matplotlib.pyplot as plt


def get_mnist_test_data():
    mnist = tf.keras.datasets.mnist
    (_, _), (x_test, y_test) = mnist.load_data()

    # Normalize the test images to the range [0, 1]
    x_test = x_test.astype("float32") / 255.0

    # Flatten the images to match the input shape (1, 784) of the model
    x_test = x_test.reshape(x_test.shape[0], 1, 784, 1)
    return x_test, y_test

def read_images_from_dataset(model_path):
    _ , _ , images_idx = run_network_mnist_test(model_path)
    x_test, y_test = get_mnist_test_data()
    x_test, y_test = x_test[images_idx], y_test[images_idx]
    return x_test, y_test, images_idx

def run_network_mnist_test(model_path):
    x_test, y_test = get_mnist_test_data()
    session = ort.InferenceSession(model_path)
    # Get the model's input name
    input_name = session.get_inputs()[0].name

    correct_predictions = 0
    num_low_conf_im = 0
    low_conf_images_idx = []
    confs = []
    # Loop through the test dataset
    for i in range(len(x_test)):
        # Prepare the test input
        test_input = x_test[i].astype(np.float32)
        
        # Run the model on the test input
        output = session.run(None, {input_name: test_input})
        
        # Get the predicted class
        predicted_class = np.argmax(output[0][0])
        softmax_output= softmax(output[0][0])
        
        # Compare with the true label
        if predicted_class == y_test[i]:
            correct_predictions += 1
            if softmax_output[predicted_class] <= 0.4:
                num_low_conf_im += 1
                low_conf_images_idx.append(i)
                confs.append(softmax_output[predicted_class])
        
    # print(low_conf_images_idx)
    # print_images(None, None, low_conf_images_idx, confs)
    accuracy = correct_predictions / len(x_test)
    # print(f"Accuracy on MNIST test dataset: {accuracy * 100:.2f}%")
    # print(f"Number of low confidence images: {num_low_conf_im}")
    return accuracy, num_low_conf_im, low_conf_images_idx

def print_images(x_test, y_test, images_idx, confs):
    x_test, y_test = get_mnist_test_data()
    x_test_2d = x_test.reshape(x_test.shape[0], 28, 28)
    # Number of images to display
    num_images = 50  # For example, 16 images in a 4x4 grid
    fig, axes = plt.subplots(6, 5, figsize=(8, 8))

    # Loop through the grid and display images with legends
    for i, ax in enumerate(axes.flat):
        if i < num_images and i < len(images_idx):
            # Plot the image
            im_idx = images_idx[i]
            ax.imshow(x_test_2d[im_idx], cmap='gray')
            
            # Set the legend/title for each image
            ax.set_title(f"{y_test[im_idx]},{im_idx},{confs[i] * 100:.2f}")
            # ax.set_title(f"ddft")
            
            # Hide the axis
            ax.axis('off')

    plt.tight_layout()
    plt.show()


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
    net_dirs = '/home/u1411251/Documents/tools/networks/conf_final/eran_mod' 
    nets = ['mnist_relu_3_50.onnx', 'mnist_relu_3_100.onnx', 'mnist_relu_5_100.onnx', 'mnist_relu_6_100.onnx', 'mnist_relu_6_200.onnx', 'mnist_relu_9_100.onnx']
    nets += ['mnist_relu_9_200.onnx', 'mnist_relu_4_1024.onnx', 'ffnnRELU__Point_6_500.onnx', 'ffnnRELU__PGDK_w_0.1_6_500.onnx', 'ffnnRELU__PGDK_w_0.3_6_500.onnx']
    images = get_images(dataset_file)
    if len(sys.argv) > 1:
        model_path = str(sys.argv[1])


    # net = nets[2]
    # model_path = os.path.join(net_dirs, net)
    # acc, num_low_images = run_network_mnist_test(model_path)
    # print(f"{net},{acc},{num_low_images}") 
    # run_model(model_path, images[:21])
    # run_manually(model_path, images)
    # print_images(None, None)
    for net in nets:
        model_path = os.path.join(net_dirs, net)
        acc, num_low_conf_images, low_conf_images_idx = run_network_mnist_test(model_path)
        print(f"{net},{acc},{num_low_conf_images}")
