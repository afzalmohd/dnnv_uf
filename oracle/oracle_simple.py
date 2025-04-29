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
import shutil




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


def select_idxs_net_oracle(indexes_vs_oracles, net_path):
    selected_idxs = []
    session = ort.InferenceSession(net_path)
    input_name = session.get_inputs()[0].name
    for idx in indexes_vs_oracles.keys():
        oracle_labels = indexes_vs_oracles[idx]
        im = IMAGES[idx]
        input_name = session.get_inputs()[0].name
        output = session.run(None, {input_name: im})
        predicted_class = np.argmax(output[0][0])
        print(f"{predicted_class},{idx}: {indexes_vs_oracles[idx]}")
        if predicted_class in oracle_labels:
            selected_idxs.append(idx)

    return selected_idxs

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

def get_net_im_ep(file_path, is_standard=True):
    filename = os.path.basename(file_path)
    filename_l = filename.split('+')
    prop_l = filename_l[1].split('_')
    im = int(prop_l[1])
    ep = float(prop_l[2])
    if is_standard:
        netname = filename_l[0]+".onnx"
    else:
        netname_l = filename_l[0].split('_')
        netname = "_".join(netname_l[:-1])+".onnx"

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

def get_cex_im_filepath(netname, ep, log_file_path, res1):
    log_dir = os.path.dirname(log_file_path)
    log_dir = '/home/u1411251/temp/temp'
    filename = os.path.basename(log_file_path)
    cex_dir = os.path.join(log_dir, netname[:-5], str(ep), res1)
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


def is_false_negative(netname, idx, ep):
    filtered_df = df[(df['netname'] == netname) & (df['image_index'] == idx) & (df['epsilon'] == ep)]
    if filtered_df.empty:
        print('------------------------------------------------------------------------------------------------------------------------')
        print(netname, idx, ep)
        return False
    res = filtered_df['result'].values[0]
    if res == 'sat':
        res1 = filtered_df['result1'].values[0]
        return res1 == 'tp'
    else:
        return False



def analyse_log_file_count(log_file_path, is_analyse_standard = True):
    netname, im, ep = get_net_im_ep(log_file_path, is_standard=is_analyse_standard)
    net_path = os.path.join(orig_net_dir, netname)
    orig_indeces_top, orig_conf_top, max_val, max_ind, smax_val, smax_ind, min_val, min_ind = get_im_label(IMAGES[im], net_path, top_k=3)
    orig_oracle_preds, orig_oracle_logs =  get_oracle_output(im=IMAGES[im], net_dir = oracle_net_dir, nets= oracle_nets)
    orig_oracle_preds = [int(pred) for pred in orig_oracle_preds]
    res = get_result(log_file_path)
    if res is None:
        res = 'timeout'
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

        cex_im_path = get_cex_im_filepath(netname, ep, log_file_path=log_file_path, res1=res1)
        
        if is_print_images:
            print_cex_with_oracle_labels(output_file=cex_im_path, orig_im=IMAGES[im], orig_label=orig_indeces_top[0], 
                                     orig_oracle_labels=orig_oracle_preds, cex_im=cex_im, cex_label=cex_label,
                                     cex_oracle_labels=cex_oracle_preds)
    else:
        if is_analyse_standard:
            update_res_table(netname, ep, res)
        else:
            if res == 'unsat':
                is_fn = is_false_negative(netname, im, ep)
                if is_fn:
                    res1 = 'fn'
                else:
                    res1 = 'tn'
                update_res_table(netname, ep, res1)
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



def analyse_dir(vnncomp_benchmarks_dir, netnames, epsilons, oracle_labels_file, log_dir='/home/u1411251/tools/result_dir/saiv/mnist/net1'):
    res_csv_header = ['logfile_name', 'netname', 'image_index', 'epsilon', 'dataset_label', 'orig_net_label', 'orig_net_conf', 'result', 'cex_label', 'cex_conf', 
                      'result1', 'orig_im_oracle', 'orig_im_oracle_log', 'cex_im_oracle', 'cex_im_oracle_log']
    global data_dict
    # write_to_csv_file(res_csv_header)
    # file_list = os.listdir(log_dir)
    # count = 0
    # for filename in file_list:
    #     log_file =  os.path.join(log_dir, filename)
    #     if os.path.isfile(log_file) and not filename.startswith('res_') and not filename.startswith('script') and not filename.startswith('im_'):
    #         data_dict = {}
    #         # if count in [408, 1526]:
    #         analyse_log_file_count(log_file)
    #         count += 1
    #         print(f"Processed file: {count}")

    write_to_csv_file(res_csv_header)
    indexes_vs_oracles = {}
    with open(oracle_labels_file, 'r') as f:
        Lines = f.readlines()
        for line in Lines:
            line = line.strip()
            line_l = line.split(',')
            idx = int(line_l[0])
            labels = [int(val) for val in line_l[1:]]
            indexes_vs_oracles[idx] = labels
    
    count = 1
    for net in netnames:
        net_path = os.path.join(vnncomp_benchmarks_dir, 'onnx', net)
        indices = select_idxs_net_oracle(indexes_vs_oracles, net_path)
        for idx in indices:
            for ep in epsilons:
                data_dict = {}
                log_file = f"{net[:-5]}+prop_{idx}_{ep}"
                log_file_path = os.path.join(log_dir, log_file)
                if os.path.exists(log_file_path):
                    analyse_log_file_count(log_file_path, is_analyse_standard=True)
                    print(f"Processed file: {count}")
                    count += 1




def analyse_standard_logfile_oracle(netname, idx, ep, is_already_analysed=True):
    standard_log_dir = '/home/u1411251/tools/result_dir/saiv/mnist/net_all'
    if is_already_analysed:
        filtered_df = df[(df['netname'] == netname) & (df['image_index'] == idx) & (df['epsilon'] == ep)]
        res = filtered_df['result'].values[0]
        if res == 'sat':
            res1 = filtered_df['result1'].values[0]
            update_res_table(netname, ep, res1)
            standard_log_file_path = os.path.join(standard_log_dir, f"{netname[:-5]}+prop_{idx}_{ep}")
            std_im_path = get_cex_im_filepath(netname, ep, standard_log_file_path, res1)
            curr_log_file_path= os.path.join(log_dir, f"{netname[:-5]}_{idx}+prop_{idx}_{ep}")
            curr_im_path = get_cex_im_filepath(netname, ep, curr_log_file_path, res1)
            shutil.copy2(std_im_path, curr_im_path)
        elif res == 'unsat':
            res1 = 'tn'
            filtered_df['result1'] = res1
            update_res_table(netname, ep, res1)
        else:
            update_res_table(netname, ep, res)
        
        filtered_df.to_csv(result_csv, mode='a', index=False, header=False)




def analyse_oracle_result(vnncomp_benchmarks_dir, netnames, epsilons, oracle_labels_file, log_dir='/home/u1411251/tools/result_dir/saiv/mnist/oracle_all', standard_res_csv='/home/u1411251/tools/my_scripts/res_standard.csv'):
    res_csv_header = ['logfile_name', 'netname', 'image_index', 'epsilon', 'dataset_label', 'orig_net_label', 'orig_net_conf', 'result', 'cex_label', 'cex_conf', 
                      'result1', 'orig_im_oracle', 'orig_im_oracle_log', 'cex_im_oracle', 'cex_im_oracle_log']
    global data_dict, df
    df = pd.read_csv(standard_res_csv)
    write_to_csv_file(res_csv_header)
    indexes_vs_oracles = {}
    with open(oracle_labels_file, 'r') as f:
        Lines = f.readlines()
        for line in Lines:
            line = line.strip()
            line_l = line.split(',')
            idx = int(line_l[0])
            labels = [int(val) for val in line_l[1:]]
            indexes_vs_oracles[idx] = labels
    
    count = 0
    for net in netnames:
        net_path = os.path.join(vnncomp_benchmarks_dir, 'onnx', net)
        indices = select_idxs_net_oracle(indexes_vs_oracles, net_path)
        for idx in indices:
            for ep in epsilons:
                log_file = f"{net[:-5]}_{idx}+prop_{idx}_{ep}"
                if len(indexes_vs_oracles[idx]) == 1:
                    # filtered_df = df[(df['netname'] == net) & (df['image_index'] == idx) & (df['epsilon'] == ep)]
                    # if filtered_df.empty:
                    #     analyse_log_file_count(log_file_path, is_analyse_standard=False)
                    # else:
                    #     analyse_standard_logfile_oracle(net, idx, ep)
                    pass
                else:
                    data_dict = {}
                    log_file_path = os.path.join(log_dir, log_file)
                    analyse_log_file_count(log_file_path, is_analyse_standard=False)
                
                print(f"Counter: {count}")
                count += 1



if __name__ == '__main__':
    global orig_net_dir, oracle_net_dir, oracle_nets, log_dir, is_print_images, result_csv
    potential_datasets = [mnist_dataset, cifar10_dataset]

    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    else:
        assert False, "Please provide the config file"

    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)

    netnames=['mnist-net_256x2.onnx', 'mnist-net_256x4.onnx', 'mnist-net_256x6.onnx']
    epsilons = [0.03, 0.05]

    is_test_data = config['is_test_data']
    dataset = config['dataset']
    is_gans_input = config['is_gans_input']
    image_shape = config['image_shape']
    vnncomp_benchmarks_dir = config['vnncomp_benchmarks_dir']
    orig_net_dir = os.path.join(vnncomp_benchmarks_dir, 'onnx')
    oracle_net_dir = config['oracle_net_dir']
    oracle_nets = config['oracle_nets']
    oracle_labels_file = config['oracle_labels']
    log_dir = config['res_log_dir']
    is_print_images = config['is_print_images']
    result_csv = config['result_csv']
    if os.path.exists(result_csv):
        os.remove(result_csv) 
    assert dataset in potential_datasets, "Invalid dataset"
    set_images_labels(dataset, is_test_data) 

    # analyse_dir(vnncomp_benchmarks_dir, netnames, epsilons, oracle_labels_file, log_dir=log_dir)
    analyse_oracle_result(vnncomp_benchmarks_dir, netnames, epsilons, oracle_labels_file, log_dir)

    print(RES_TABLE)

    # with open("res.json", 'w') as f:
    #     json.dump(RES_TABLE, f, indent=4)

    # get_oracles_labels_on_orig_images()
