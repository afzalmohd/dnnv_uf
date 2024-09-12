import os
import csv
import sys
import onnxruntime as ort
import numpy as np
import matplotlib.pyplot as plt
from simulate_network import get_mnist_test_data
from simulate_network import get_mnist_train_data

IMAGES = []
LABELS = []

top_k = 3


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

def top_k_pred(softmax_output, k):
    top_indices = np.argsort(softmax_output)[-k:][::-1]

    # Get the top three confidence scores
    top_confidences = softmax_output[top_indices]

    return top_indices, top_confidences



def run_network(net_path, image, is_normalized=True):
    session = ort.InferenceSession(net_path)
    input_name = session.get_inputs()[0].name
    # print(image.shape)
    test_input = image.reshape(1,784,1)
    # print(test_input.shape)
    if not is_normalized:
        test_input /= 255
    test_input = test_input.astype(np.float32)
    output = session.run(None, {input_name: test_input})
    # predicted_class = np.argmax(output[0][0])
    softmax_output= softmax(output[0][0])
    top_indeces, top_confidences = top_k_pred(softmax_output, top_k)
    # print(f"Classification with class: {top_indeces}, conf: {top_confidences}")
    return top_indeces, top_confidences


def get_result(file_path):
    with open(file_path, 'r') as f:
        Lines =  f.readlines()
        for line in Lines:
            line = line.strip()
            if 'Result:' in line:
                line_list = line.split(':')
                return line_list[1].replace(' ', '')
            
def extract_dir_conf(log_dir, cex_dir, net_dir, is_print_ce = False):
    for filename in os.listdir(log_dir):
        file_path = os.path.join(log_dir, filename)
        if os.path.isfile(file_path) and (not 'res' in filename) and (not 'script' in filename):
            res = get_result(file_path)
            netname, im_idx, conf_th, ep = get_net_im_conf_ep(file_path)
            top_indecies, orig_conf = run_network(os.path.join(net_dir, netname), IMAGES[int(im_idx)], is_normalized=True)
            ce_conf = [1,0,0]
            if conf_th == 0.0:
                print(res, LABELS[int(im_idx)], top_indecies[0])
            if res == 'sat' and LABELS[int(im_idx)] == top_indecies[0]:
                ce = extract_ce(file_path)
                if is_print_ce:
                    print_ce(file_path, cex_dir, is_conf_logs=True)
                _, ce_conf = run_network(os.path.join(net_dir, netname), ce, is_normalized=True)
                print(conf_th)
                print(f"{netname},{im_idx},{ep},{conf_th},{orig_conf[0] * 100:.2f},{ce_conf[0] * 100:.2f},{res}")
            elif res == 'unsat':
                print(conf_th)
                print(f"{netname},{im_idx},{ep},{conf_th},{orig_conf[0] * 100:.2f},{ce_conf[0] * 100:.2f},{res}")
            
            # print(f"{netname},{im_idx},{ep},{conf},{orig_conf[0] * 100:.2f},{ce_conf[0] * 100:.2f},{res}")

def extract_dir_normal(log_dir, cex_dir, net_dir):
    for filename in os.listdir(log_dir):
        file_path = os.path.join(log_dir, filename)
        if os.path.isfile(file_path) and (not 'res' in filename) and (not 'script' in filename):
            res = get_result(file_path)
            filename_split = filename.split('+')
            filename_split_1 = filename_split[0]
            netname = filename_split_1+".onnx"
            filename_split_2 = filename_split[1]
            ep = filename_split_2.split('_')[-1]
            im_idx = filename_split_2.split('_')[-2]
            top_indecies, orig_conf = run_network(os.path.join(net_dir, netname), IMAGES[int(im_idx)], is_normalized=True)
            ce_conf = [1,0,0]
            if res == 'sat' and LABELS[int(im_idx)] == top_indecies[0]:
                ce = extract_ce(file_path)
                _, ce_conf = run_network(os.path.join(net_dir, netname), ce, is_normalized=True)
                print_ce(file_path, cex_dir, is_conf_logs=False)
                print(f"{netname},{im_idx},{ep},0,{orig_conf[0] * 100:.2f},{ce_conf[0] * 100:.2f},{res}")
            elif res == 'unsat':
                print(f"{netname},{im_idx},{ep},0,{orig_conf[0] * 100:.2f},{ce_conf[0] * 100:.2f},{res}")


def get_images_list(dataset_file):
    labels = []
    images = []
    with open(dataset_file, 'r') as f:
        csv_readers = csv.reader(f)
        for row in csv_readers:
            labels.append(row[0])
            images.append(np.array(row[1:]).astype(np.float32))
    
    return images, labels

def get_net_im_conf_ep(file_path):
    filename = os.path.basename(file_path)
    filename_l = filename.split('+')
    prop_l = filename_l[1].split('_')
    im = int(prop_l[1])
    ep = float(prop_l[2])
    netname_l = filename_l[0].split('_')
    if len(netname_l) >= 6:
        conf = float(netname_l[-2])
        netname = "_".join(netname_l[:-2])+".onnx"
    else:
        print(filename)
        conf = 0.0
        netname = "_".join(netname_l)+".onnx"

    return netname, im, conf, ep




def create_dir(out_dir, file_path):
    net, im, conf, ep = get_net_im_conf_ep(file_path)
    dir_path = os.path.join(out_dir, net[:-5], str(conf)+"+"+str(ep))
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    
    return dir_path



def print_ce(log_file, output_dir, is_conf_logs=True):
    ce = extract_ce(log_file)
    filename = os.path.basename(log_file)
    image_idx = None
    if is_conf_logs:
        netname, image_idx, conf, ep = get_net_im_conf_ep(log_file)
    else:
        filename_split = filename.split('+')
        filename_split_1 = filename_split[0]
        netname = filename_split_1+".onnx"
        filename_split_2 = filename_split[1]
        im_idx = filename_split_2.split('_')[-2]
        image_idx = int(im_idx)
    # print(image_idx)
    orig_image = IMAGES[image_idx]
    ce = ce.reshape(28,28)
    orig_image = orig_image.reshape(28,28)
    top_classes, top_confidences = run_network(os.path.join(net_dir, netname), orig_image, is_normalized=True)
    ce_top_classes, ce_top_confs = run_network(os.path.join(net_dir, netname), ce, is_normalized=True)
    # print(f"Original class: {LABELS[image_idx]}")
    fig, axes = plt.subplots(1, 3, figsize=(6, 6))
    titled_str = ""
    for i in range(top_k):
        titled_str += f"{top_classes[i]},{top_confidences[i] * 100:.2f}\n"
    axes[0].imshow(orig_image, cmap='gray_r')
    axes[0].set_title(titled_str)
    axes[0].axis('off')

    diff_image = np.absolute(orig_image - ce)
    diff_image[0][0] = 1.0
    axes[1].imshow(diff_image, cmap='gray_r')
    # axes[1].set_title(f"{ce_pred_class},{ce_conf * 100:.2f}")
    axes[1].axis('off')

    titled_str = ""
    for i in range(top_k):
        titled_str += f"{ce_top_classes[i]},{ce_top_confs[i] * 100:.2f}\n"
    axes[2].imshow(ce, cmap='gray_r')
    axes[2].set_title(titled_str)
    axes[2].axis('off')

    plt.tight_layout()
    # plt.show()
    # return
    new_dir = create_dir(output_dir, log_file)
    output_file = os.path.join(new_dir, f"{filename}.png")
    plt.savefig(output_file)









if __name__ == '__main__':
    IS_CONF_ANALYSIS = True
    if IS_CONF_ANALYSIS:
        log_dir = '/home/u1411251/Documents/tools/result_dir/mod_prop/logs_5_100_6_200'
    else:
        log_dir = '/home/u1411251/Documents/tools/result_dir/with_venky/logs_simple_selected_idx'

    dataset_file = '/home/u1411251/Documents/tools/VeriNN/deep_refine/benchmarks/dataset/mnist/mnist_test.csv'
    net_dir = '/home/u1411251/Documents/tools/networks/conf_final/eran_mod'
    cex_dir = os.path.join(log_dir, 'cexs')
    
    IMAGES, LABELS = get_mnist_train_data() 
    if IS_CONF_ANALYSIS:
        extract_dir_conf(log_dir, cex_dir, net_dir, is_print_ce=False)
    else:
        extract_dir_normal(log_dir, cex_dir, net_dir)

    # log_file = '/home/u1411251/Documents/tools/result_dir/aaai25/abcrown/logs_orig_dataset_normal/mnist_relu_3_50+prop_9954_0.06'
    # if len(sys.argv) > 1:
    #     log_file = str(sys.argv[1])

    
    # print_ce(log_file, out_dir, is_conf_logs=False)