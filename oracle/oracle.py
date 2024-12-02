import sys
import os
import numpy as np
import onnxruntime as ort
import yaml
import matplotlib.pyplot as plt
from collections import Counter
import csv
# import pandas as pd
cwd = os.getcwd()
sys.path.append(f"{cwd}")

from extract_logs.logs_extract_abcrown import extract_ce, get_result, get_net_im_conf_ep_1, run_network
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

def set_images_labels_gan_with_oracle(image_csv, image_shape):
    global IMAGES, LABELS
    with open(image_csv, 'r') as f:
        selected_images, selected_labels = [], []
        csv_readers = csv.reader(f, delimiter=',')
        count = 0
        for row in csv_readers:
            label = int(row[0])
            image = np.array(row[1:])
            image = image.reshape(image_shape)
            image = image.astype(np.float32)
            # label1 = get_oracle_output(image, oracle_net_dir, orcale_nets)[0]
            selected_images.append(image)
            selected_labels.append(label)
            # if label != label1:
            #     print(count, label, label1)

            count += 1

        print(f"number of images: {count+1}")

    IMAGES = np.array(selected_images)
    LABELS = np.array(selected_labels)
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
    # print(pred_labels)  
    for val in pred_labels:
        count = label_dictionary.get(val, 0)
        count += 1
        label_dictionary[val] = count
    
    sorted_labeled_dictionary = dict(sorted(label_dictionary.items(), key=lambda item: item[1], reverse=True))
    # print(sorted_labeled_dictionary)
    multi_preds = [item[0] for item in sorted_labeled_dictionary.items() if item[1] >= 3]
    if len(multi_preds) == 0:
        # multi_preds.append([sorted_labeled_dictionary[0]])
        multi_preds.append(list(sorted_labeled_dictionary.keys())[0])
    # print(f"Oracle's output: {multi_preds}")
    return multi_preds

def get_oracle_output_for_logs(im:np.ndarray, net_dir, nets):
    im = im.reshape(-1,1,28,28)
    ret_str = None
    for net in nets:
        net_path = os.path.join(net_dir, net)
        session = ort.InferenceSession(net_path)
        input_name = session.get_inputs()[0].name
        output = session.run(None, {input_name: im})
        pred = np.argmax(output[0][0])
        softmax_output= softmax(output[0][0])
        pred_conf = softmax_output[pred]
        pred_conf = round(pred_conf * 100, 1)
        if ret_str == None:
            ret_str = f"{pred}:{pred_conf}"
        else:
            ret_str = ret_str+f" - {pred}:{pred_conf}"
    
    return ret_str   

def get_im_label(cex, net_path):
    # print(f"netpath: {net_path}")
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


def print_ce_oracle(log_file, output_file, net_dir):
    ce = extract_ce(log_file)
    netname, image_idx, conf, ep = get_net_im_conf_ep_1(log_file)
    orig_image = IMAGES[image_idx]
    # if is_gans:
    #     orig_image, _ = get_image_label_gans(image_idx)
    ce = ce.reshape(28,28)
    orig_image = orig_image.reshape(28,28)
    
    top_classes, top_confidences = run_network(os.path.join(net_dir, netname), orig_image, is_normalized=True)
    ce_top_classes, ce_top_confs = run_network(os.path.join(net_dir, netname), ce, is_normalized=True)
    # print(f"CEX class: {ce_top_classes[0]} Original class: {top_classes[0]}")
    fig, axes = plt.subplots(1, 3, figsize=(6, 6))
    top_k = 3
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
    plt.savefig(output_file)
    plt.close(fig)
    plt.clf()


def get_zero_conf_log_file(netname, ep, im):
    log_file_0_conf = f"{netname[:-5]}_0_{im}+prop_{im}_{ep}_0"
    log_file_0_conf = os.path.join(log_dir, log_file_0_conf)
    return log_file_0_conf

def is_fp_log(im, logfile, netname):
    # print(os.path.basename(logfile))
    net_path = os.path.join(orig_net_dir, netname)
    cex = extract_ce(logfile)
    # print(logfile)
    # print(cex.shape)
    cex_label = get_im_label(cex, net_path)
    print(f"cex label: {cex_label}, orig label: {LABELS[im]}")
    is_fp_1 = is_fp(cex_label, cex)
    if is_fp_1:
        print(f"FP: {os.path.basename(logfile)}, label: {cex_label}")
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

def dump_images(log_file, res, net_dir):
    # return
    netname, im, conf, ep = get_net_im_conf_ep_1(log_file)
    dump_file_name = f"{netname[:-5]}+{im}+{conf}+{ep}"
    if res == 'fp':
        dir_path = os.path.join(log_dir, 'cex', 'fp')
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)
    elif res == 'tp':
        dir_path = os.path.join(log_dir, 'cex', 'tp')
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)
    elif res == 'fn':
        dir_path = os.path.join(log_dir, 'cex', 'fn')
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)

        log_file = get_zero_conf_log_file(netname, ep, im)
        npy_path = os.path.join(dir_path, 'npy')
        if not os.path.isdir(npy_path):
            os.makedirs(npy_path)
        
        cex = extract_ce(log_file)
        oracle_label = get_oracle_output(cex, oracle_net_dir, orcale_nets)
        oracle_label = oracle_label[0]

        fn_file_name = f"{dump_file_name}+{oracle_label}.npy"        
        np.save(os.path.join(npy_path, fn_file_name), cex)  
    else: #tn
        dir_path = os.path.join(log_dir, 'cex', 'tn')
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)

        log_file = get_zero_conf_log_file(netname, ep, im)
    

    cex = extract_ce(log_file)
    oracle_label = get_oracle_output(cex, oracle_net_dir, orcale_nets)
    oracle_label = oracle_label[0]
    dump_file_name = f"{dump_file_name}+{oracle_label}.png"
    dump_cex_path = os.path.join(dir_path, dump_file_name)
    print_ce_oracle(log_file, dump_cex_path, net_dir=net_dir)

def write_to_csv_file(row):
    with open(result_csv, 'a') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(row)
        csv_file.close()



def dump_to_csv_file(log_file, res, res1, log_file1=None, conf1=-1):
    # return
    netname, im, conf, ep = get_net_im_conf_ep_1(log_file)
    net_path = os.path.join(orig_net_dir, netname)
    image = IMAGES[im]
    im_label_net = get_im_label(image, net_path)
    im_oracle_label = get_oracle_output_for_logs(image, oracle_net_dir, orcale_nets)
    cex_label_net = None
    cex_oracle_label = None
    tp_file_name = None
    tp_cex_label = None
    tp_cex_oracle_label = None
    if res1 == 'fp' or res1 == 'tp':
        cex = extract_ce(log_file)
        cex_label_net = get_im_label(cex, net_path)
        cex_oracle_label = get_oracle_output_for_logs(cex, oracle_net_dir, orcale_nets)
    elif log_file1 != None:
        tp_file_name = os.path.basename(log_file1)
        tp_cex = extract_ce(log_file1)
        tp_cex_label = get_im_label(tp_cex, net_path)
        tp_cex_oracle_label = get_oracle_output_for_logs(tp_cex, oracle_net_dir, orcale_nets)

    csv_row = [netname, im, conf,conf1, ep, im_label_net, im_oracle_label, res, res1, cex_label_net, cex_oracle_label, tp_file_name, tp_cex_label, tp_cex_oracle_label]
    
    write_to_csv_file(csv_row)     
    


def analyse_log_file_count(log_file):
    net_dir = orig_net_dir
    netname, im, conf, ep = get_net_im_conf_ep_1(log_file)
    res = get_result(log_file)
    if conf != 0.0 and conf == 60.0:
        if res == 'sat':
            is_fp = is_fp_log(im, log_file, netname)
            if is_fp: 
                res1 = 'fp'
                fp_conf.append(os.path.basename(log_file))
            else:
                res1 = 'tp'
            update_res_table(conf, False, res1)
            dump_images(log_file, res1, net_dir)
            dump_to_csv_file(log_file, res, res1)
            
            log_file_0_conf = get_zero_conf_log_file(netname, ep, im)
            res_0 = get_result(log_file_0_conf)
            if res_0 == 'sat':
                is_fp1 = is_fp_log(im, log_file_0_conf, netname)
                if is_fp1:
                    res1 = 'fp'
                    fp_zero.append(os.path.basename(log_file))
                else:
                    res1 = 'tp'

                update_res_table(conf, True, res1)
                dump_images(log_file_0_conf, res1, net_dir)
                dump_to_csv_file(log_file_0_conf, res_0, res1, conf1=conf)
            elif res_0 == 'unsat':
                print(f"Something wrong......{log_file_0_conf}...............")
            else:
                update_res_table(conf, True, 'timeout')
                dump_to_csv_file(log_file_0_conf, res_0, None, conf1=conf)


        elif res == 'unsat':
            log_file_0_conf = get_zero_conf_log_file(netname, ep, im)
            res_0 = get_result(log_file_0_conf)
            if res_0 == 'sat':
                is_fp = is_fp_log(im, log_file_0_conf, netname)
                if is_fp:
                    update_res_table(conf, False, 'tn')
                    dump_images(log_file, 'tn', net_dir)
                    dump_to_csv_file(log_file, res, 'tn', log_file_0_conf)

                    update_res_table(conf, True, 'fp')
                    dump_images(log_file_0_conf, 'fp', net_dir)
                    fp_zero.append(os.path.basename(log_file))
                    dump_to_csv_file(log_file_0_conf, res_0, 'fp', conf1=conf)
                else:
                    update_res_table(conf, False, 'fn')
                    dump_images(log_file, 'fn', net_dir)
                    dump_to_csv_file(log_file, res, 'fn', log_file_0_conf)

                    update_res_table(conf, True, 'tp')
                    dump_images(log_file_0_conf, 'tp', net_dir)
                    dump_to_csv_file(log_file_0_conf, res_0, 'tp', conf1=conf)
                    
            elif res_0 == 'unsat':
                update_res_table(conf, False, 'tn')
                dump_to_csv_file(log_file, res, 'tn')

                update_res_table(conf, True, 'tn')
                dump_to_csv_file(log_file_0_conf, res_0, 'tn', conf1=conf)
            else:
                update_res_table(conf, False, 'tn')
                dump_to_csv_file(log_file, res, 'tn')

                update_res_table(conf, True, 'timeout')
                dump_to_csv_file(log_file_0_conf, res_0, None, conf1=conf)

        else:
            update_res_table(conf, False, 'timeout')
            dump_to_csv_file(log_file, res, None)
            log_file_0_conf = get_zero_conf_log_file(netname, ep, im)
            res_0 = get_result(log_file_0_conf)
            if res_0 == 'sat':
                is_fp = is_fp_log(im, log_file_0_conf, netname)
                if is_fp:
                    update_res_table(conf, True, 'fp')
                    dump_images(log_file_0_conf, 'fp', net_dir)
                    fp_zero.append(os.path.basename(log_file))
                    dump_to_csv_file(log_file_0_conf, res_0, 'fp', conf1=conf)
                else:
                    update_res_table(conf, True, 'tp')
                    dump_images(log_file_0_conf, 'tp', net_dir)
                    dump_to_csv_file(log_file_0_conf, res_0, 'tp', conf1=conf)
                    
            elif res_0 == 'unsat':
                update_res_table(conf, True, 'tn')
                dump_to_csv_file(log_file_0_conf, res_0, 'tn', conf1=conf)
            else:
                update_res_table(conf, True, 'timeout')
                dump_to_csv_file(log_file_0_conf, res_0, None, conf1=conf)


def analyse_dir():
    res_csv_header = ['netname', 'image_index', 'confidence1', 'confidence2', 'ep', 'orig_image_label_net', 
    'orig_image_label_oracle', 'result', 'cex_label_net', 'cex_label_oracle', 'tp_file_name', 
    'tp_cex_label_net', 'tp_cex_label_oracle']
    write_to_csv_file(res_csv_header)
    file_list = os.listdir(log_dir)
    count = 0
    for filename in file_list:
        log_file =  os.path.join(log_dir, filename)
        if os.path.isfile(log_file) and not filename.startswith('res_') and not filename.startswith('script'):
            analyse_log_file_count(log_file)
            count += 1
            print(f"Processed file: {count}")



                                
if __name__ == '__main__':
    global orig_net_dir, oracle_net_dir, orcale_nets, log_dir, result_csv, fp_conf, fp_zero
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
    images_csv = config['images_csv_file']
    image_shape = config['image_shape']
    orig_net_dir = config['net_dir']
    oracle_net_dir = config['orcale_net_dir']
    orcale_nets = config['orcale_nets']
    log_dir = config['log_dir']
    result_csv = config['result_csv']
    # if os.path.exists(result_csv):
    #     os.remove(result_csv) 
    assert dataset in potential_datasets, "Invalid dataset"
    if is_gans_input:
        set_images_labels_gan_with_oracle(images_csv, image_shape)
    else:
        set_images_labels(dataset, is_test_data)    

    # fp_conf = []
    # fp_zero = []

    # analyse_dir()

    # print(RES_TABLE)

    # print(len(fp_zero))
    # # fp_zero = ['mnist-net_256x6_0_119+prop_119_0.06_0', 'mnist-net_256x4_0_52+prop_52_0.06_0', 'mnist-net_256x2_0_183+prop_183_0.06_0', 'mnist-net_256x4_0_119+prop_119_0.06_0', 'mnist-net_256x6_0_79+prop_79_0.06_0', 'mnist-net_256x4_0_114+prop_114_0.06_0', 'mnist-net_256x4_0_69+prop_69_0.06_0', 'mnist-net_256x6_0_183+prop_183_0.06_0', 'mnist-net_256x6_0_61+prop_61_0.06_0', 'mnist-net_256x2_0_29+prop_29_0.06_0', 'mnist-net_256x4_0_183+prop_183_0.06_0', 'mnist-net_256x6_0_69+prop_69_0.06_0', 'mnist-net_256x4_0_61+prop_61_0.06_0', 'mnist-net_256x4_0_49+prop_49_0.06_0']
    # print(len(fp_conf))
    # # fp_conf = ['mnist-net_256x6_70_119+prop_119_0.06_70', 'mnist-net_256x4_70_52+prop_52_0.06_70', 'mnist-net_256x2_70_183+prop_183_0.06_70', 'mnist-net_256x4_70_119+prop_119_0.06_70', 'mnist-net_256x6_70_2+prop_2_0.06_70', 'mnist-net_256x4_70_95+prop_95_0.06_70', 'mnist-net_256x6_70_79+prop_79_0.06_70', 'mnist-net_256x4_70_114+prop_114_0.06_70', 'mnist-net_256x4_70_69+prop_69_0.06_70', 'mnist-net_256x6_70_183+prop_183_0.06_70', 'mnist-net_256x4_70_79+prop_79_0.06_70', 'mnist-net_256x6_70_61+prop_61_0.06_70', 'mnist-net_256x4_70_183+prop_183_0.06_70', 'mnist-net_256x6_70_114+prop_114_0.06_70', 'mnist-net_256x4_70_2+prop_2_0.06_70', 'mnist-net_256x4_70_61+prop_61_0.06_70', 'mnist-net_256x4_70_29+prop_29_0.06_70', 'mnist-net_256x4_70_49+prop_49_0.06_70']

    # for f1 in fp_conf:
    #     f1l = f1.split('+')
    #     f1l_s = f1l[1].split('_')
    #     im1 = int(f1l_s[1])
    #     conf1 = float(f1l_s[3])
    #     net1_l = f1l[0].split('_')
    #     net1 = "_".join(net1_l[:-2])

    #     is_exist = False
    #     for f2 in fp_zero:
    #         f2l = f2.split('+')
    #         f2l_s = f2l[1].split('_')
    #         im2 = int(f2l_s[1])
    #         conf2 = float(f1l_s[3])
    #         net2_l = f2l[0].split('_')
    #         net2 = "_".join(net2_l[:-2])

    #         if net1 == net2 and im1 == im2:
    #             is_exist = True

    #     if not is_exist:
    #         print(f1)





    # df = pd.DataFrame(RES_TABLE)
    # print(df)







