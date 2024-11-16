import sys
import os
project_path = os.path.join(os.getcwd())
sys.path.append(project_path)
import tensorflow as tf
import numpy as np
from PIL import Image
import onnxruntime as ort
import torch
import yaml
from gan_mnist import generate_fake_samples
from cond_gan import get_one_hot_labels, combine_vectors, get_noise,Generator, show_tensor_images
from oracle.oracle import get_oracle_output


if len(sys.argv) > 1:
    config_file = sys.argv[1]

with open(config_file, 'r') as file:
    config = yaml.safe_load(file)

oracle_net_dir = config['orcale_net_dir']
orcale_nets = config['orcale_nets']  


def get_label_by_oracle(image_tensor):
    image_unflat = image_tensor.detach().cpu()
    image_numpy = image_unflat.numpy()  # Shape: (200, 1, 28, 28)
    labels = []
    for idx, image in enumerate(image_numpy):
        image = image.astype(np.float32)
        print(f"index: {idx}", end=' , ')
        l = get_oracle_output(image, oracle_net_dir, orcale_nets)[0]
        labels.append(l)
    # print(labels)
    return labels



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
    net_path= '/home/afzal/tools/networks/conf_final/eran_mod/mnist_relu_5_100.onnx'
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

def gen_images_tf_gans():
    model_path = os.path.join(os.getcwd(), 'results', 'generator_model_070.h5')
    images_path = os.path.join(os.getcwd(), 'images')
    model = tf.keras.models.load_model(model_path)
    latent_dims = 100
    n_samples = 100

    X, _ =  generate_fake_samples(model, latent_dims, n_samples)
    visualize_images(X, images_path)
    print(X.shape)

def gen_images_pytorch_gans():
    device = 'cpu'
    n_classes = 10
    num_images = 200
    z_dim = 64
    generator_input_dim = z_dim + n_classes
    labels1 = torch.randint(0, 2, (60,)) * 6 + 1
    labels2 = torch.randint(0, 2, (60,)) * 5 + 4
    labels3 = torch.randint(0, 2, (40,)) * 5 + 3
    labels4 = torch.randint(0, 2, (20,)) * 2
    labels5 = torch.randint(0, 2, (20,)) * 1 + 5
    labels = torch.cat((labels1, labels2, labels3, labels4, labels5), dim=0)
    result_path = os.path.join(os.getcwd(), 'gans', 'results')
    model_path = os.path.join(result_path, 'generator_80.pth')

    gen = Generator(input_dim=generator_input_dim).to(device)
    gen.load_state_dict(torch.load(model_path, weights_only=True))
    gen.eval()
    one_hot_labels = get_one_hot_labels(labels.to(device), n_classes)
    fake_noise = get_noise(num_images, z_dim, device=device)
    noise_and_labels = combine_vectors(fake_noise, one_hot_labels)

    fake = gen(noise_and_labels)
    labels = get_label_by_oracle(fake)
    labels = torch.tensor(labels)
    show_tensor_images(fake, num_images=num_images,nrow=14, save_path=os.path.join(result_path, 'images.png'), csv_path= os.path.join(result_path, 'images_csv.csv'), labels=labels)
    

gen_images_pytorch_gans()
    



