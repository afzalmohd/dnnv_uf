import sys
import os
import numpy as np
import onnxruntime as ort
import yaml
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
import csv
import math
cwd = os.getcwd()
sys.path.append(f"{cwd}")
sys.path.append(f"{cwd}/..")
print(sys.path)
from generate_benchmarks.simulate_network import get_mnist_test_data, get_mnist_train_data, get_cifar10_test_data, get_cifar10_train_data, softmax, get_max_smax
from oracle import get_im_label, get_oracle_output
from extract_logs.logs_extract_abcrown import get_result
import pandas as pd
import json




mnist_dataset = 'MNIST'
cifar10_dataset = 'CIFAR10'
cifar100_dataset='CIFAR100'
imagenet_dataset = 'imagenet'
tsr_dataset = 'tsr'

RES_TABLE = {}

def set_images_labels(dataset, is_test_data):
    global IMAGES, LABELS
    if dataset == 'MNIST':
        if is_test_data:
            IMAGES, LABELS = get_mnist_test_data()
        else:
            IMAGES, LABELS = get_mnist_train_data()
    elif dataset == 'CIFAR10':
        if is_test_data:
            IMAGES, LABELS = get_cifar10_test_data()
        else:
            IMAGES, LABELS = get_cifar10_train_data()
        
        IMAGES = np.transpose(IMAGES, (0, 3, 1, 2))
        LABELS = LABELS.reshape(-1)
    
    print(IMAGES.shape)
    print(LABELS.shape)


def write_to_csv_file(row):
    # return
    with open(result_csv, 'a') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(row)
        csv_file.close()

def update_res_table(netname, ep, res):
    res_table = RES_TABLE.get(netname, {})
    res_table_ep = res_table.get(ep, {})
    count = res_table_ep.get(res, 0)
    count += 1
    res_table_ep[res] = count
    res_table[ep] = res_table_ep
    RES_TABLE[netname] = res_table

def get_net_im_ep(file_path):
    filename = os.path.basename(file_path)
    filename_l = filename.split('+')
    netname = filename_l[0]+".onnx"
    prop_l = filename_l[1].split('_')
    im = int(prop_l[1])
    ep = float(prop_l[2])
    return netname, im, ep

def get_cex_info(net_path, cex_im):
    indeces_top, conf_top, max_val, max_ind, smax_val, smax_ind, min_val, min_ind = get_im_label(cex_im, net_path, top_k=3)
    return indeces_top[0], conf_top[0]

def print_cex_with_oracle_labels(output_file, orig_im, orig_label, orig_oracle_labels, cex_im, cex_label, cex_oracle_labels):
    orig_im = orig_im.reshape(28,28)
    cex_im = cex_im.reshape(28,28)
    fig, axes = plt.subplots(1, 2, figsize=(4, 4))
    top_k = 3
    # titled_str = f"orig label: {orig_label}\norig oracle: {orig_oracle_labels}"
    titled_str = f"orig label: {orig_label}"
    axes[0].imshow(orig_im, cmap='gray_r')
    axes[0].set_title(titled_str)
    axes[0].axis('off')


    titled_str = f"cex label: {cex_label}\ncex oracle: {cex_oracle_labels}"
    axes[1].imshow(cex_im, cmap='gray_r')
    axes[1].set_title(titled_str)
    axes[1].axis('off')

    plt.tight_layout()
    # plt.show()
    # return
    plt.savefig(output_file)
    plt.close(fig)
    plt.clf()

def get_cex_im_filepath(log_file_path, res1):
    log_dir = os.path.dirname(log_file_path)
    filename = os.path.basename(log_file_path)
    cex_dir = os.path.join(log_dir, res1)
    os.makedirs(cex_dir, exist_ok=True)
    cex_file_path = os.path.join(cex_dir, f"{filename}.png")
    return cex_file_path

def get_oracles_labels_on_orig_images(index_files='/home/u1411251/tools/my_scripts/oracle/indices.txt'):
    with open(index_files) as f:
        line = f.readline()
        indexes = np.fromstring(line, dtype=int, sep=',')
        images = IMAGES[indexes]
        labels = LABELS[indexes] 
    Lines = []
    counter = 0
    for idx in indexes:
        preds, _ =  get_oracle_output(im=IMAGES[idx], net_dir = oracle_net_dir, nets= oracle_nets)
        preds = preds[:2]
        preds = [str(int(val)) for val in preds]
        preds = ",".join(preds)
        line = f"{idx},{preds}\n"
        # print(line)
        Lines.append(line)
        print(f"processed: {counter}")
        counter += 1

    with open('oracles_lables_mnist.txt', 'w') as f:
        f.writelines(Lines)


    
    return images, labels, indexes




def analyse_log_file_count(log_file_path):
    netname, im, ep = get_net_im_ep(log_file_path)
    net_path = os.path.join(orig_net_dir, netname)
    orig_indeces_top, orig_conf_top, max_val, max_ind, smax_val, smax_ind, min_val, min_ind = get_im_label(IMAGES[im], net_path, top_k=3)
    orig_oracle_preds, orig_oracle_logs =  get_oracle_output(im=IMAGES[im], net_dir = oracle_net_dir, nets= oracle_nets)
    orig_oracle_preds = [int(pred) for pred in orig_oracle_preds]
    res = get_result(log_file_path)
    cex_label, cex_conf = None, None
    res1 = None
    cex_oracle_preds, cex_oracle_logs = None, None
    if res == 'sat':
        res1 = 'tp'
        log_dir = os.path.dirname(log_file_path)
        log_file1 = os.path.basename(log_file_path)
        im_log_file = "im_"+log_file1+".npy"
        im_log_file_path = os.path.join(log_dir, im_log_file)
        cex_im = np.load(im_log_file_path)
        cex_label, cex_conf = get_cex_info(net_path, cex_im)
        cex_oracle_preds, cex_oracle_logs =  get_oracle_output(im=IMAGES[im], net_dir = oracle_net_dir, nets= oracle_nets)
        cex_oracle_preds = [int(pred) for pred in cex_oracle_preds]
        if cex_label in cex_oracle_preds:
            res1 = 'fp'
        update_res_table(netname, ep, res1)

        cex_im_path = get_cex_im_filepath(log_file_path=log_file_path, res1=res1)
        
        if is_print_images:
            print_cex_with_oracle_labels(output_file=cex_im_path, orig_im=IMAGES[im], orig_label=orig_indeces_top[0], 
                                     orig_oracle_labels=orig_oracle_preds, cex_im=cex_im, cex_label=cex_label,
                                     cex_oracle_labels=cex_oracle_preds)
    else:
        update_res_table(netname, ep, res)


    data_dict['log_file'] = os.path.basename(log_file_path)
    data_dict['netname'] = netname
    data_dict['image_index'] = im
    data_dict['epsilon'] = ep
    data_dict['dataset_label'] = LABELS[im]
    data_dict['orig_net_label'] = orig_indeces_top[0]
    data_dict['orig_net_conf'] = round(float(orig_conf_top[0]) * 100, 2)
    data_dict['result'] = res
    data_dict['cex_label'] = cex_label
    data_dict['cex_conf'] = cex_conf
    data_dict['result1'] = res1
    data_dict['orig_label_oracle'] = orig_oracle_preds
    data_dict['orig_im_oracle_log'] = orig_oracle_logs
    data_dict['cex_label_oracle'] = cex_oracle_preds
    data_dict['cex_im_oracle_log'] = cex_oracle_logs

    res_row = [os.path.basename(log_file_path), netname, im, ep, data_dict['dataset_label'], data_dict['orig_net_label'], data_dict['orig_net_conf'], res, cex_label, cex_conf, 
               res1, orig_oracle_preds, orig_oracle_logs, cex_oracle_preds, cex_oracle_logs]
    
    write_to_csv_file(res_row)



def analyse_dir(log_dir='/home/u1411251/tools/result_dir/saiv/mnist/net1'):
    res_csv_header = ['logfile_name', 'netname', 'image_index', 'epsilon', 'dataset_label', 'orig_net_label', 'orig_net_conf', 'result', 'cex_label', 'cex_conf', 
                      'result1', 'orig_im_oracle', 'orig_im_oracle_log', 'cex_im_oracle', 'cex_im_oracle_log']
    global data_dict
    write_to_csv_file(res_csv_header)
    file_list = os.listdir(log_dir)
    count = 0
    for filename in file_list:
        log_file =  os.path.join(log_dir, filename)
        if os.path.isfile(log_file) and not filename.startswith('res_') and not filename.startswith('script') and not filename.startswith('im_'):
            data_dict = {}
            # if count in [408, 1526]:
            analyse_log_file_count(log_file)
            count += 1
            print(f"Processed file: {count}")




if __name__ == '__main__':
    global orig_net_dir, oracle_net_dir, oracle_nets, log_dir, is_print_images, result_csv
    potential_datasets = [mnist_dataset, cifar10_dataset]

    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    else:
        assert False, "Please provide the config file"

    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)

    is_test_data = config['is_test_data']
    dataset = config['dataset']
    is_gans_input = config['is_gans_input']
    image_shape = config['image_shape']
    vnncomp_benchmarks_dir = config['vnncomp_benchmarks_dir']
    orig_net_dir = os.path.join(vnncomp_benchmarks_dir, 'onnx')
    oracle_net_dir = config['oracle_net_dir']
    oracle_nets = config['oracle_nets']
    log_dir = config['res_log_dir']
    is_print_images = config['is_print_images']
    result_csv = config['result_csv']
    if os.path.exists(result_csv):
        os.remove(result_csv) 
    assert dataset in potential_datasets, "Invalid dataset"
    set_images_labels(dataset, is_test_data) 

    # analyse_dir(log_dir=log_dir)

    # print(RES_TABLE)

    # with open("res.json", 'w') as f:
    #     json.dump(RES_TABLE, f, indent=4)

    get_oracles_labels_on_orig_images()
