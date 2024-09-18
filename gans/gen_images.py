import tensorflow as tf
import os
import numpy as np
from PIL import Image
import onnxruntime as ort
from gan_mnist import generate_fake_samples

def softmax(x):
    e_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
    return e_x / e_x.sum(axis=-1, keepdims=True)

def top_k_pred(softmax_output, k):
    top_indices = np.argsort(softmax_output)[-k:][::-1]

    # Get the top three confidence scores
    top_confidences = softmax_output[top_indices]

    return top_indices, top_confidences

def get_labels(net_path, images, is_normalized = True):
    session = ort.InferenceSession(net_path)
    input_name = session.get_inputs()[0].name
    # print(image.shape)
    labels = []
    confs = []
    for i in range(len(images)):
        image = images[i]
        test_input = image.reshape(1,784,1)
        # print(test_input.shape)
        if not is_normalized:
            test_input /= 255
        test_input = test_input.astype(np.float32)
        output = session.run(None, {input_name: test_input})
        softmax_output= softmax(output[0][0])
        # print(softmax_output)
        top_indeces, top_confidences = top_k_pred(softmax_output, 3)
        labels.append(top_indeces[0])
        confs.append(top_confidences[0])
        # print(f"Classification with class: {top_indeces}, conf: {top_confidences}")
    return labels, confs


def visualize_images(mnist_images, dir_path):
    os.makedirs(dir_path, exist_ok=True)

    labels, confs = get_labels(net_path, mnist_images)
    # exit(0)
    # Convert and save each image
    for i in range(100):
        # Extract the image at index i and reshape it to (28, 28)
        img_array = mnist_images[i].reshape(28, 28)
        # inverted_img_array = 1 - img_array
        
        # Convert the NumPy array to an image
        img = Image.fromarray((img_array * 255).astype(np.uint8))  # Multiply by 255 to get pixel values in the range 0-255
        # Save the image in PNG format (or change to JPEG if you prefer)
        label = labels[i]
        conf = confs[i]
        image_path= os.path.join(dir_path, f"image_{i}_{label}.png")
        # image_path = os.path.join(dir_path, f"image_{i}.png")
        img.save(image_path)

    print("Images saved successfully!")


model_path = os.path.join(os.getcwd(), 'results', 'generator_model_070.h5')
images_path = os.path.join(os.getcwd(), 'images')
net_path= '/home/afzal/tools/networks/conf_final/eran_mod/mnist_relu_5_100.onnx'
model = tf.keras.models.load_model(model_path)
latent_dims = 100
n_samples = 100

X, _ =  generate_fake_samples(model, latent_dims, n_samples)
visualize_images(X, images_path)
print(X.shape)

