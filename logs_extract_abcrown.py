import os
import csv
import onnxruntime as ort
import numpy as np
import matplotlib.pyplot as plt

dataset_file = '/home/u1411251/Documents/tools/VeriNN/deep_refine/benchmarks/dataset/mnist/mnist_test.csv'
net_dir = '/home/u1411251/Documents/tools/networks/conf_final/eran_mod'
out_dir = '/home/u1411251/Documents/tools/my_scripts/logs'
IMAGES = []
LABELS = []


def extract_ce(log_file):
    is_adv = False
    is_tensor = False
    ce = []
    with open(log_file, 'r') as f:
        Lines = f.readlines()
        for line in Lines:
            line = line.strip()
            if "Adv example:" in line:
                is_adv = True
            elif is_adv and 'tensor([[[' in line:
                line = line.replace('tensor([[[', '')
                line = line.replace('],', '')
                # print(line)
                ce.append(float(line))
                is_tensor = True
            elif is_tensor and is_adv and "]]])" in line:
                line = line.replace(']]])', '')
                line = line.replace('[', '')
                ce.append(float(line))
                # print(line)
                is_adv = False
                is_tensor = False
            elif is_adv and is_tensor:
                line = line.replace('[', '')
                line = line.replace('],', '')
                ce.append(float(line))
                # print(line)
    return np.array(ce)


def softmax(x):
    e_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
    return e_x / e_x.sum(axis=-1, keepdims=True)

def run_network(net_path, image, is_normalized=False):
    session = ort.InferenceSession(net_path)
    input_name = session.get_inputs()[0].name
    # print(image.shape)
    test_input = image.reshape(1,784,1)
    # print(test_input.shape)
    if not is_normalized:
        test_input /= 255
    test_input = test_input.astype(np.float32)
    output = session.run(None, {input_name: test_input})
    predicted_class = np.argmax(output[0][0])
    softmax_output= softmax(output[0][0])
    return predicted_class, softmax_output[predicted_class]


def get_result(file_path):
    with open(file_path, 'r') as f:
        Lines =  f.readlines()
        for line in Lines:
            line = line.strip()
            if 'Result:' in line:
                line_list = line.split(':')
                return line_list[1].replace(' ', '')
            
def extract_dir_conf(dir_name):
    for filename in os.listdir(dir_name):
        file_path = os.path.join(dir_name, filename)
        if os.path.isfile(file_path) and (not 'res' in filename) and (not 'script' in filename):
            res = get_result(file_path)
            filename_split = filename.split('+')
            filename_split_1 = filename_split[0]
            conf_imidx = filename_split_1.split('_')
            im_idx = conf_imidx[-1]
            conf = conf_imidx[-2]
            netname = "_".join(conf_imidx[:-2])+".onnx"
            _, orig_conf = run_network(os.path.join(net_dir, netname), IMAGES[int(im_idx)])
            filename_split_2 = filename_split[1]
            ep = filename_split_2.split('_')[-1]
            ce_conf = 1
            if res == 'sat':
                ce = extract_ce(file_path)
                print_ce(ce, out_dir, is_conf_logs=True)
                _, ce_conf = run_network(os.path.join(net_dir, netname), ce, is_normalized=True)
            print(f"{netname},{im_idx},{ep},{conf},{orig_conf * 100:.2f},{ce_conf * 100:.2f},{res}")

def extract_dir_normal(dir_name):
    for filename in os.listdir(dir_name):
        file_path = os.path.join(dir_name, filename)
        if os.path.isfile(file_path) and (not 'res' in filename) and (not 'script' in filename):
            res = get_result(file_path)
            filename_split = filename.split('+')
            filename_split_1 = filename_split[0]
            netname = filename_split_1+".onnx"
            filename_split_2 = filename_split[1]
            ep = filename_split_2.split('_')[-1]
            im_idx = filename_split_2.split('_')[-2]
            _, orig_conf = run_network(os.path.join(net_dir, netname), IMAGES[int(im_idx)])
            ce_conf = 1
            if res == 'sat':
                ce = extract_ce(file_path)
                _, ce_conf = run_network(os.path.join(net_dir, netname), ce, is_normalized=True)
                print_ce(file_path, out_dir, is_conf_logs=False)
            print(f"{netname},{im_idx},{ep},0,{orig_conf * 100:.2f},{ce_conf * 100:.2f},{res}")

def get_images_list(dataset_file):
    labels = []
    images = []
    with open(dataset_file, 'r') as f:
        csv_readers = csv.reader(f)
        for row in csv_readers:
            labels.append(row[0])
            images.append(np.array(row[1:]).astype(np.float32))
    
    return images, labels

def print_ce(log_file, output_dir, is_conf_logs=True):
    ce = extract_ce(log_file)
    filename = os.path.basename(log_file)
    image_idx = None
    if is_conf_logs:
        filename_split = filename.split('+')
        filename_split_1 = filename_split[0]
        conf_imidx = filename_split_1.split('_')
        im_idx = conf_imidx[-1]
        netname = "_".join(conf_imidx[:-2])+".onnx"
        image_idx = int(im_idx)
    else:
        filename_split = filename.split('+')
        filename_split_1 = filename_split[0]
        netname = filename_split_1+".onnx"
        filename_split_2 = filename_split[1]
        im_idx = filename_split_2.split('_')[-2]
        image_idx = int(im_idx)
    
    orig_image = IMAGES[image_idx]
    orig_image /= 255
    ce = ce.reshape(28,28)
    orig_image = orig_image.reshape(28,28)
    pred_class, orig_conf = run_network(os.path.join(net_dir, netname), orig_image)
    ce_pred_class, ce_conf = run_network(os.path.join(net_dir, netname), ce, is_normalized=True)

    fig, axes = plt.subplots(1, 3, figsize=(4, 4))

    axes[0].imshow(orig_image, cmap='gray_r')
    axes[0].set_title(f"{pred_class},{orig_conf * 100:.2f}")
    axes[0].axis('off')

    diff_image = np.absolute(orig_image - ce)
    diff_image[0][0] = 1.0
    axes[1].imshow(diff_image, cmap='gray_r')
    # axes[1].set_title(f"{ce_pred_class},{ce_conf * 100:.2f}")
    axes[1].axis('off')

    axes[2].imshow(ce, cmap='gray_r')
    axes[2].set_title(f"{ce_pred_class},{ce_conf * 100:.2f}")
    axes[2].axis('off')

    # plt.tight_layout()
    # plt.show()
    output_file = os.path.join(output_dir, f"{filename}.png")
    plt.savefig(output_file)









if __name__ == '__main__':
    dir_name = '/home/u1411251/Documents/tools/result_dir/aaai25/abcrown/conf/logs'
    dir_name = '/home/u1411251/Documents/tools/result_dir/aaai25/abcrown/logs_normal'
    is_conf_logs = False
    IMAGES, LABELS =  get_images_list(dataset_file)
    # log_file = '/home/u1411251/Documents/tools/result_dir/aaai25/abcrown/conf/logs/mnist_relu_3_50_60_12+prop_12_0.06'
    # extract_ce(log_file)
    # print(images[0]/255)  
    if is_conf_logs:
        extract_dir_conf(dir_name)
    else:
        extract_dir_normal(dir_name)

    # log_file = '/home/u1411251/Documents/tools/result_dir/aaai25/abcrown/logs_normal/mnist_relu_9_200+prop_4_0.06'
    # print_ce(log_file, out_dir, is_conf_logs=False)