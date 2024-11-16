import sys
import os
import numpy as np
import onnxruntime as ort
import yaml
from collections import Counter
cwd = os.getcwd()
sys.path.append(f"{cwd}")

from extract_logs.logs_extract_abcrown import extract_ce, get_result, get_net_im_conf_ep_1
from generate_benchmarks.simulate_network import get_mnist_test_data, get_mnist_train_data, get_cifar10_test_data, get_cifar10_train_data, softmax

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




def get_oracle_output(im:np.ndarray, net_dir, nets):
    im = im.reshape(-1,1,28,28)
    pred_labels = []
    confs = []
    label_dictionary = {}
    for net in nets:
        net_path = os.path.join(net_dir, net)
        session = ort.InferenceSession(net_path)
        input_name = session.get_inputs()[0].name
        output = session.run(None, {input_name: im})
        pred = np.argmax(output[0][0])
        pred_labels.append(pred)
        softmax_output= softmax(output[0][0])
        confs.append(softmax_output[pred])
    
    for val in pred_labels:
        count = label_dictionary.get(val, 0)
        count += 1
        label_dictionary[val] = count
    
    sorted_labeled_dictionary = dict(sorted(label_dictionary.items(), key=lambda item: item[1], reverse=True))
    multi_preds = [item[0] for item in sorted_labeled_dictionary.items() if item[1] >= 3]
    if len(multi_preds) == 0:
        multi_preds.append([sorted_labeled_dictionary[0]])
    print(f"Final output: {multi_preds}")
    return multi_preds

def get_cex_label(cex, net_path):
    cex = cex.reshape(1,784,1)
    session = ort.InferenceSession(net_path)
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: cex})
    pred = np.argmax(output[0][0])
    return pred

def is_fp(cex_label, cex: np.ndarray):
    oracle_output = get_oracle_output(cex, oracle_net_dir, orcale_nets)
    if cex_label in oracle_output:
        return True
    else:
        return False

def is_tp(cex_label, cex: np.ndarray):
    return (not is_fp(cex_label, cex))

def is_fn(cex_label, cex: np.ndarray):
    pass

def is_tn():
    pass



def get_zero_conf_log_file(netname, ep, im):
    log_file_0_conf = f"{netname[:-5]}_0_{im}+prop_{im}_{ep}_0"
    log_file_0_conf = os.path.join(log_dir, log_file_0_conf)
    return log_file_0_conf

def is_fp_log(im, logfile, netname):
    net_path = os.path.join(orig_net_dir, netname)
    cex = extract_ce(logfile)
    cex_label = get_cex_label(cex, net_path)
    print(f"cex label: {cex_label}, orig label: {LABELS[im]}")
    is_fp_1 = is_fp(cex_label, cex)
    return is_fp_1

def update_res_table(conf, is_zero, res):
    res_tabel_1 = RES_TABLE.get(conf, {})
    if is_zero:
        res_4_table = res_tabel_1.get(0, {})
        count = res_4_table.get(res, 0)
        res_4_table[res] = count + 1
        res_tabel_1[0] = res_4_table
    else:
        res_4_table = res_tabel_1.get(conf, {})
        count = res_4_table.get(res, 0)
        res_4_table[res] = count + 1
        res_tabel_1[conf] = res_4_table
    
    RES_TABLE[conf] = res_tabel_1


def analyse_log_file_count(log_file):
    netname, im, conf, ep = get_net_im_conf_ep_1(log_file)
    res = get_result(log_file)
    if conf != 0.0:
        if res == 'sat':
            is_fp = is_fp_log(im, log_file, netname)
            if is_fp:  
                update_res_table(conf, False, 'fp')
            else:
                update_res_table(conf, False, 'tp')
            
            log_file_0_conf = get_zero_conf_log_file(netname, ep, im)
            res1 = get_result(log_file_0_conf)
            if res1 == 'sat':
                is_fp1 = is_fp_log(im, log_file_0_conf, netname)
                if is_fp1:
                    update_res_table(conf, True, 'fp')
                else:
                    update_res_table(conf, True, 'tp')
            elif res1 == 'unsat':
                print(f"Something wrong......{log_file_0_conf}...............")

        elif res == 'unsat':
            log_file_0_conf = get_zero_conf_log_file(netname, ep, im)
            res1 = get_result(log_file_0_conf)
            if res1 == 'sat':
                is_fp = is_fp_log(im, log_file_0_conf, netname)
                cex = extract_ce(log_file_0_conf)
                if is_fp:
                    update_res_table(conf, False, 'tn')
                    update_res_table(conf, True, 'fp')
                else:
                    update_res_table(conf, False, 'fn')
                    update_res_table(conf, True, 'tp')
                    fn_file_name = f"{netname}+{conf}+{im}+{ep}.npy"
                    fn_log_dir = os.path.join(dump_fn_dir, str(conf), 'fn')
                    if not os.path.isdir(fn_log_dir):
                        os.makedirs(fn_log_dir)
                    
                    np.save(os.path.join(fn_log_dir, fn_file_name), cex)
                    

            elif res1 == 'unsat':
                update_res_table(conf, False, 'tn')
                update_res_table(conf, True, 'tn')
            else:
                update_res_table(conf, False, 'tn')


def analyse_dir():
    file_list = os.listdir(log_dir)
    for filename in file_list:
        log_file =  os.path.join(log_dir, filename)
        if os.path.isfile(log_file):
            analyse_log_file_count(log_file)



                                
if __name__ == '__main__':
    global orig_net_dir, oracle_net_dir, orcale_nets, log_dir, dump_fn_dir
    potential_datasets = [mnist_dataset, cifar10_dataset]
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    else:
        assert False, "Please provide the config file"

    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)

    is_test_data = config['is_test_data']
    dataset = config['dataset']
    assert dataset in potential_datasets, "Invalid dataset"
    set_images_labels(dataset, is_test_data)
    net_root_dir = config['net_root_dir']
    
    if dataset == mnist_dataset:
        orig_net_dir = os.path.join(net_root_dir, 'mnist', 'vnncomp')
    elif dataset == cifar10_dataset:
        orig_net_dir = os.path.join(net_root_dir, 'cifar10', 'vnncomp')
    
    oracle_net_dir = config['orcale_net_dir']
    orcale_nets = config['orcale_nets']
    log_dir = config['log_dir']
    dump_fn_dir = config['dump_fn_dir']

    analyse_dir()

    print(RES_TABLE)







