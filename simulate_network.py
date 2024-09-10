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

def get_mnist_train_data():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (_, _) = mnist.load_data()

    # Normalize the test images to the range [0, 1]
    x_train = x_train.astype("float32") / 255.0

    # Flatten the images to match the input shape (1, 784) of the model
    x_train = x_train.reshape(x_train.shape[0], 1, 784, 1)
    return x_train, y_train


def get_selected_images():
    image_idxs = []
    file_path = '/home/afzal/tools/my_scripts/low_conf_images.txt'
    with open(file_path, 'r') as f:
        Lines = f.readlines()
        for line in Lines:
            line = line.strip()
            # print(line)
            image_idxs.append(int(line))

    image_idxs = list(set(image_idxs))
    # print(image_idxs)
    print(len(image_idxs))
    x_train, y_train = get_mnist_train_data()
    x_train = x_train[image_idxs]
    y_train = y_train[image_idxs]
    return x_train, y_train, image_idxs

def read_images_from_dataset(model_path):
    _ , _ , images_idx = run_network_mnist_test(model_path)
    x_test, y_test = get_mnist_test_data()
    x_test, y_test = x_test[images_idx], y_test[images_idx]
    return x_test, y_test, images_idx

def run_network_mnist_test(model_path, is_test_dataset = False, is_cnn = False, is_softmax_output = False, conf_th = 70):
    conf_th = conf_th / 100.0
    if is_test_dataset:
        x_test, y_test = get_mnist_test_data()
    else:
        x_test, y_test = get_mnist_train_data()
    if is_cnn:
        x_test = x_test.reshape(x_test.shape[0], 1, 28,28)
    session = ort.InferenceSession(model_path)
    # Get the model's input name
    input_name = session.get_inputs()[0].name

    correct_predictions = 0
    num_low_conf_im = 0
    low_conf_images_idx = []
    low_confs = []
    high_conf_image_idx = []
    high_confs = []
    # Loop through the test dataset
    for i in range(len(x_test)):
        # Prepare the test input
        test_input = x_test[i].astype(np.float32)
        if is_cnn:
            test_input = test_input.reshape(1,1,28,28)
        
        # Run the model on the test input
        output = session.run(None, {input_name: test_input})
        
        # Get the predicted class
        predicted_class = np.argmax(output[0][0])
        if is_softmax_output:
            softmax_output = output[0][0]
        else:
            softmax_output= softmax(output[0][0])
        
        # Compare with the true label
        if predicted_class == y_test[i]:
            correct_predictions += 1
            # print(softmax_output[predicted_class])
            if softmax_output[predicted_class] <= conf_th:
                num_low_conf_im += 1
                low_conf_images_idx.append(i)
                low_confs.append(softmax_output[predicted_class])
            else:
                high_conf_image_idx.append(i)
                high_confs.append(softmax_output[predicted_class])
                
        
    # print(low_conf_images_idx)
    # print_images(None, None, low_conf_images_idx, confs)
    accuracy = correct_predictions / len(x_test)
    # print(f"Accuracy on MNIST test dataset: {accuracy * 100:.2f}%")
    # print(f"Number of low confidence images: {num_low_conf_im}")
    # print(accuracy, num_low_conf_im)
    return accuracy, num_low_conf_im, low_conf_images_idx, low_confs, high_conf_image_idx, high_confs

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

def print_images_1(images_idx, confs, is_test_dataset=False):

    # Assuming x_test and y_test are provided
    if is_test_dataset:
        x_test, y_test = get_mnist_test_data()
    else:
        x_test, y_test = get_mnist_train_data()

    # Reshape x_test back to 2D images for visualization
    x_test_2d = x_test.reshape(x_test.shape[0], 28, 28)

    # Number of images to display
    num_images = len(images_idx)  # Number of images you want to display
    images_per_grid = 100  # Maximum number of images per grid (10x10)
    max_num_rows, max_num_columns = 10,10

    # Loop through the images and display them in grids
    for grid_idx in range(0, num_images, images_per_grid):
        # Determine the range of images to display in this grid
        grid_end_idx = min(grid_idx + images_per_grid, num_images)
        grid_images = x_test_2d[images_idx[grid_idx:grid_end_idx]]
        grid_labels = y_test[images_idx[grid_idx:grid_end_idx]]
        grid_confs = confs[grid_idx:grid_end_idx]

        # Determine the grid size (rows and columns)
        num_images_in_grid = len(grid_images)
        grid_rows = (num_images_in_grid // max_num_rows) + (num_images_in_grid % max_num_rows != 0)
        grid_cols = min(max_num_columns, num_images_in_grid)

        # Create a figure with a grid of subplots
        fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(grid_cols+5, grid_rows+5))

        # Ensure axes is always 2D array even for single row or column
        if grid_rows == 1:
            axes = np.expand_dims(axes, axis=0)
        if grid_cols == 1:
            axes = np.expand_dims(axes, axis=1)

        # Loop through the grid and display images with legends
        for i, ax in enumerate(axes.flat):
            if i < num_images_in_grid:
                # Plot the image
                im_idx = grid_idx + i
                ax.imshow(grid_images[i], cmap='gray_r')
                
                # Set the legend/title for each image
                ax.set_title(f"{grid_labels[i]},{images_idx[im_idx]},{grid_confs[i] * 100:.1f}", fontsize=8)
                
                # Hide the axis
                ax.axis('off')
            else:
                ax.axis('off')  # Hide any unused subplots

        # plt.tight_layout()
        plt.subplots_adjust(hspace=0.5, wspace=0.3)
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
    is_test_dataset = True
    dataset_file = "/home/u1411251/Documents/tools/VeriNN/deep_refine/benchmarks/dataset/mnist/mnist_test.csv"
    is_cnn = False
    net_dirs = '/home/u1411251/Documents/tools/networks/conf_final/eran_mod' 
    nets = ['mnist_relu_3_50.onnx', 'mnist_relu_3_100.onnx', 'mnist_relu_5_100.onnx', 'mnist_relu_6_100.onnx', 'mnist_relu_6_200.onnx', 'mnist_relu_9_100.onnx']
    nets += ['mnist_relu_9_200.onnx', 'mnist_relu_4_1024.onnx', 'ffnnRELU__Point_6_500.onnx', 'ffnnRELU__PGDK_w_0.1_6_500.onnx', 'ffnnRELU__PGDK_w_0.3_6_500.onnx']

    #cnn
    is_cnn = True
    net_dirs = '/home/u1411251/Documents/tools/networks/erans_nets'
    nets = []
    nets += ['convSmallRELU__Point.onnx', 'convSmallRELU__PGDK.onnx', 'convSmallRELU__DiffAI.onnx']
    nets += ['convMedGRELU__Point.onnx']
    nets += ['mnist_conv_maxpool.onnx']
    nets += ['convBigRELU__DiffAI.onnx']

    is_softmax_out_layer = False
    net_dirs = '/home/u1411251/Documents/tools/my_scripts'
    nets = ['pytorch_model.onnx']

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
        acc, num_low_conf_images, low_conf_images_idx, confs = run_network_mnist_test(model_path, is_cnn=is_cnn, is_test_dataset=is_test_dataset, is_softmax_output=is_softmax_out_layer)
        print(f"{net},{acc},{num_low_conf_images}")
        print_images_1(low_conf_images_idx, confs, is_test_dataset=is_test_dataset)
