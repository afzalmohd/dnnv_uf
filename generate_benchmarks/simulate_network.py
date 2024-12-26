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

def get_cifar10_test_data():
    cifar10 = tf.keras.datasets.cifar10
    (_, _), (x_test, y_test) = cifar10.load_data()

    # Normalize the test images to the range [0, 1]
    x_test = x_test.astype("float32") / 255.0

    # Flatten the images to match the input shape (1, 784) of the model
    # x_test = x_test.reshape(x_test.shape[0], 1, 784, 1)
    return x_test, y_test

def get_cifar10_train_data():
    cifar10 = tf.keras.datasets.cifar10
    (x_train, y_train), (_, _) = cifar10.load_data()

    # Normalize the test images to the range [0, 1]
    x_train = x_train.astype("float32") / 255.0

    # Flatten the images to match the input shape (1, 784) of the model
    # x_train = x_train.reshape(x_train.shape[0], 1, 784, 1)
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

def top_k_pred(softmax_output, k):
    top_indices = np.argsort(softmax_output)[-k:][::-1]

    # Get the top three confidence scores
    top_confidences = softmax_output[top_indices]

    return top_indices, top_confidences

def read_images_from_dataset(model_path):
    _ , _ , images_idx = run_network_mnist_test(model_path)
    x_test, y_test = get_mnist_test_data()
    x_test, y_test = x_test[images_idx], y_test[images_idx]
    return x_test, y_test, images_idx


def select_images_top_k(model_path, is_test_dataset = False, is_cnn = False, conf_th = 30, is_softmax_output = False, num_top_k = 2):
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

    selected_images_idxs = []
    selected_labels_top_k = []
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
            top_indices, top_confs = top_k_pred(softmax_output, num_top_k)
            if top_confs[0] >= conf_th and top_confs[1] >= conf_th:
                selected_images_idxs.append(i)
                selected_labels_top_k.append(list(top_indices)[:num_top_k])

    return selected_images_idxs, selected_labels_top_k
            

def run_network_mnist_test(model_path, is_test_dataset = False, is_cnn = False, is_softmax_output = False, conf_th = 70, mean=0.0, std=1.0, image_shape = (1,784,1)):
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
    counter = 0
    for i in range(len(x_test)):
        # Prepare the test input
        counter += 1
        # if counter >= 100:
        #     break
        test_input = x_test[i].astype(np.float32)
        # image_shape = (-1,) + image_shape
        test_input = test_input.reshape(image_shape)
        # if is_cnn:
        #     test_input = test_input.reshape(1,784,1)
        
        # print(model_path)
        # print(test_input.shape)

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
    # print(f"Accuracy: {correct_predictions / counter}")
    # print(f"Accuracy on MNIST test dataset: {accuracy * 100:.2f}%")
    # print(f"Number of low confidence images: {num_low_conf_im}")
    # print(accuracy, num_low_conf_im)
    return accuracy, num_low_conf_im, low_conf_images_idx, low_confs, high_conf_image_idx, high_confs


def run_network_cifar10(model_path, is_test_dataset = False, is_cnn = True, is_softmax_output = False, conf_th = 70, mean=0.0, std=1.0):
    conf_th = conf_th / 100.0
    if is_test_dataset:
        x_test, y_test = get_cifar10_test_data()
    else:
        x_test, y_test = get_cifar10_train_data()
    if is_cnn:
        x_test = np.transpose(x_test, (0, 3, 1, 2))
    session = ort.InferenceSession(model_path)
    # Get the model's input name
    mean = np.array(mean).reshape((1,-1,1,1)).astype(np.float32)
    std = np.array(std).reshape((1,-1,1,1)).astype(np.float32)
    # print(mean, std)
    # print(x_test[0,1,:5,:5])
    x_test = (x_test - mean)/std
    # print(x_test[0,1,:5,:5])
    input_name = session.get_inputs()[0].name

    correct_predictions = 0
    num_low_conf_im = 0
    low_conf_images_idx = []
    low_confs = []
    high_conf_image_idx = []
    high_confs = []
    # Loop through the test dataset
    counter = 0
    for i in range(len(x_test)):
        # Prepare the test input
        counter += 1
        # if counter >= 100:
        #     break
        test_input = x_test[i].astype(np.float32)
        if is_cnn:
            test_input = test_input.reshape(-1,3,32,32)
        
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
    # print(f"Cifar-10 accuracy: {correct_predictions/counter}")
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


def get_selected_images_gans(net_path, images, idxes, conf_th, is_normalized = True, image_shape=(1,784,1)):
    conf_th = conf_th / 100
    session = ort.InferenceSession(net_path)
    input_name = session.get_inputs()[0].name
    low_conf_indexes = []
    high_conf_indexes = []
    for i in range(len(images)):
        image = images[i]
        test_input = image.reshape(image_shape)
        # print(test_input.shape)
        if not is_normalized:
            test_input /= 255
        # test_input = test_input.astype(np.float32)
        output = session.run(None, {input_name: test_input})
        softmax_output= softmax(output[0][0])
        # print(softmax_output)
        top_indecis, top_confidences = top_k_pred(softmax_output, 3)
        # print(i, top_indecis[0], top_confidences[0])
        # print(top_confidences)
        if top_confidences[0] >= conf_th:
            high_conf_indexes.append(idxes[i])
        else:
            low_conf_indexes.append(idxes[i])

    return high_conf_indexes, low_conf_indexes

def get_selected_images_gans_with_delta_th(net_path, images, idxes, delta_th, is_normalized = True, image_shape=(1,784,1)):
    session = ort.InferenceSession(net_path)
    input_name = session.get_inputs()[0].name
    low_conf_indexes = []
    high_conf_indexes = []
    for i in range(len(images)):
        image = images[i]
        test_input = image.reshape(image_shape)
        # print(test_input.shape)
        if not is_normalized:
            test_input /= 255
        # test_input = test_input.astype(np.float32)
        output = session.run(None, {input_name: test_input})
        output = output[0][0]
        
        max_value = np.max(output)
        max_count = np.sum(output == max_value)
        if max_count > 1:
            second_max = max_value  # Second max is the same as max
            gap = 0
        else:
            masked_arr = output[output != max_value]
            second_max = np.max(masked_arr) if masked_arr.size > 0 else None
            gap = (max_value - second_max) if second_max is not None else None
            
        assert gap != None, "Something wrong in images filteration with delta threshold"
       
        if gap >= delta_th:
            high_conf_indexes.append(idxes[i])
        else:
            low_conf_indexes.append(idxes[i])

    return high_conf_indexes, low_conf_indexes



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
    is_cnn = True
    net_dirs = '/home/afzal/tools/networks_old/networks/onnx/mnist' 
    nets = ['mnist_relu_3_50.onnx', 'mnist_relu_3_100.onnx', 'mnist_relu_5_100.onnx', 'mnist_relu_6_100.onnx', 'mnist_relu_6_200.onnx', 'mnist_relu_9_100.onnx']
    nets += ['mnist_relu_9_200.onnx', 'mnist_relu_4_1024.onnx', 'ffnnRELU__Point_6_500.onnx', 'ffnnRELU__PGDK_w_0.1_6_500.onnx', 'ffnnRELU__PGDK_w_0.3_6_500.onnx']

    #cnn
    # is_cnn = True
    # # net_dirs = '/home/afzal/tools/networks_old/networks/eran_mnist'
    # nets = []
    nets += ['convSmallRELU__Point.onnx', 'convSmallRELU__PGDK.onnx', 'convSmallRELU__DiffAI.onnx']
    nets += ['convMedGRELU__Point.onnx']
    nets += ['mnist_conv_maxpool.onnx']
    nets += ['convBigRELU__DiffAI.onnx']

    is_softmax_out_layer = False
    # net_dirs = '/home/u1411251/Documents/tools/my_scripts'
    # nets = ['pytorch_model.onnx']

    # images = get_images(dataset_file)
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
        acc, num_low_conf_images, low_conf_images_idx, confs, _, _ = run_network_mnist_test(model_path, is_cnn=is_cnn, is_test_dataset=is_test_dataset, is_softmax_output=is_softmax_out_layer)
        print(f"{net},{acc}")
        # print_images_1(low_conf_images_idx, confs, is_test_dataset=is_test_dataset)
