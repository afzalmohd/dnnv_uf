import onnxruntime as ort
import numpy as np
import torchvision.datasets as datasets
import random

def load_mnist_data():
    """Downloads and loads the MNIST dataset without extra normalization."""
    mnist_dataset = datasets.MNIST(root="../data", train=False, transform=None, download=True)
    return mnist_dataset

def preprocess_images(img1, img2):
    """Prepares two MNIST images by flattening and scaling pixel values to [0,1]."""
    img1 = np.array(img1, dtype=np.float32) / 255.0  # Scale pixels to [0,1]
    img2 = np.array(img2, dtype=np.float32) / 255.0

    img1 = img1.reshape(1, -1)  # Flatten from (28, 28) to (1, 28*28)
    img2 = img2.reshape(1, -1)

    input_data = np.concatenate((img1, img2), axis=1)  # Shape: (1, 2*28*28)
    input_data = input_data.reshape(1,-1,1)
    return input_data

def get_image_from_npy(file_path='atack_images.npy'):
    image_idx = 9 # total 10 images
    loaded_array = np.load(file_path)
    print(np.allclose(loaded_array[0][1], loaded_array[0][5]))
    image = loaded_array[0][image_idx]
    image = image.reshape(1,-1,1)
    return image

def run_onnx_inference(model_path):
    """Runs inference using the merged ONNX model with MNIST dataset."""
    # Load MNIST dataset
    mnist = load_mnist_data()

    # Select two random images
    idx1, idx2 = random.sample(range(len(mnist)), 2)
    img1, label1 = mnist[idx1]
    img2, label2 = mnist[idx2]

    print(f"Using digits: {label1} and {label2}")

    # Preprocess images
    input_data = preprocess_images(img1, img2)
    # input_data = get_image_from_npy()
    print(input_data.shape)
    # Load ONNX model
    session = ort.InferenceSession(model_path)

    # Get input/output names
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # Run inference
    output = session.run([output_name], {input_name: input_data})[0]

    print("Merged Model Output Shape:", output.shape)
    print("Merged Model Output:", output)


if __name__ == '__main__':
    # Path to the merged ONNX model
    merged_model_path = "merged_model.onnx"

    # Run inference
    # get_image_from_npy()
    run_onnx_inference(merged_model_path)
