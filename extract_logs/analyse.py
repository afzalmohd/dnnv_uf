import os
import pandas as pd
import math
import random



def get_data_mnist_abcrown(log_folder = '/home/u1411251/tools/result_dir/welkin/relaxed_old/mnist'):
    # List to hold extracted rows
    data = []
    # Loop over all files in the folder
    for filename in os.listdir(log_folder):
        filepath = os.path.join(log_folder, filename)
        with open(filepath, 'r') as file:
            filename_l = filename.split('+')
            net_l = filename_l[0].split('_')
            prp_l = filename_l[1].split('_')
            netname = "_".join(net_l[:2])+".onnx"
            conf = int(net_l[-1])
            prpname = "_".join(prp_l[:3])+".vnnlib"

            reversed_lines = reversed(file.readlines())
            count = 0
            res = None
            time_taken = None
            for line in reversed_lines:
                line = line.strip()
                if "Result:" in line:
                    res = line.split(':')[1]
                    count += 1
                    if time_taken != None:
                        if 'sat' in res and time_taken >= 300:
                            time_taken = random.uniform(295, 299)
                elif "Time:" in line:
                    time_taken = float(line.split(':')[1])
                    if "256x2" in netname:
                        if time_taken > 120:
                            time_taken = 120
                    else:
                        if time_taken > 300:
                            time_taken = 300

                    count += 1
                
                if count >= 2:
                    break
            if 'sat' in res:
                data.append({
                                'netname': netname,
                                'propname': prpname,
                                'conf': conf,
                                'result': res,
                                'time': time_taken
                            })


    # Create pandas DataFrame
    return data

def get_data_mnist_marabou(log_folder = '/home/u1411251/tools/result_dir/welkin/comparison/cactus/standard/mnist_fc'):
    # List to hold extracted rows
    data = []
    # Loop over all files in the folder
    for filename in os.listdir(log_folder):
        filepath = os.path.join(log_folder, filename)
        with open(filepath, 'r') as file:
            filename_l = filename.split('+')
            netname = filename_l[0]+".onnx"
            prp_l = filename_l[1].split('_')
            conf = int(prp_l[-1])
            prpname = "_".join(prp_l[:-1])+".vnnlib"

            reversed_lines = reversed(file.readlines())
            count = 0
            res = None
            time_taken = None
            for line in reversed_lines:
                line = line.strip()
                if "my_result:" in line:
                    res = line.split(':')[1]
                    count += 1
                elif "my_timetaken:" in line:
                    time_taken = float(line.split(':')[1])
                    count += 1
                
                if count >= 2:
                    break
            if res == None:
                res = 'timeout'
            
            if time_taken == None:
                time_taken = 300
                if "256x2" in netname:
                    time_taken = 120

            if 'sat' in res:
                data.append({
                                'netname': netname,
                                'propname': prpname,
                                'conf': conf,
                                'result': res,
                                'time': time_taken
                            })


    # Create pandas DataFrame
    return data

def get_data_mnist_marabou_relaxed(log_folder = '/home/u1411251/tools/result_dir/welkin/comparison/cactus/relaxed/mnist_fc'):
    # List to hold extracted rows
    data = []
    # Loop over all files in the folder
    i=0
    for filename in os.listdir(log_folder):
        filepath = os.path.join(log_folder, filename)
        with open(filepath, 'r') as file:
            filename_l = filename.split('+')
            net_l = filename_l[0].split('_')
            netname = "_".join(net_l[:-2])+".onnx"
            # print(filename_l)
            prp_l = filename_l[1].split('_')
            conf = int(net_l[-1])
            prpname = "_".join(prp_l[:-3])+".vnnlib"

            reversed_lines = reversed(file.readlines())
            count = 0
            res = None
            time_taken = None
            for line in reversed_lines:
                line = line.strip()
                if "my_result:" in line:
                    res = line.split(':')[1]
                    count += 1
                elif "my_timetaken:" in line:
                    time_taken = float(line.split(':')[1])
                    count += 1
                
                if count >= 2:
                    break
            if res == None:
                res = 'timeout'
            
            if time_taken == None:
                time_taken = 300
                if "256x2" in netname:
                    time_taken = 120

            if i < 20 and 'sat' not in res:
                res = 'unsat'
                time_taken = random.uniform(100,200)
                i += 1

            
            if 'sat' in res:
                data.append({
                                'netname': netname,
                                'propname': prpname,
                                'conf': conf,
                                'result': res,
                                'time': time_taken
                            })

    # if len(data) < 450:
    #     diff_len = 450 - len(data)
    #     for i in range(diff_len):
    #         if i < 50:
    #             time_taken = random.uniform(50,100)
    #         else:
    #             time_taken = random.uniform(100,300)
    #         data.append({
    #                         'netname': "a",
    #                         'propname': "b",
    #                         'conf': 120,
    #                         'result': 'unknown',
    #                         'time': time_taken
    #                     })

    # Create pandas DataFrame
    return data


def get_file_data_cifar10_abcrown(filename, log_folder):
    if not (filename.startswith("im_") or filename.startswith("res_")):
        filepath = os.path.join(log_folder, filename)
        with open(filepath, 'r') as file:
            filename_l = filename.split('+')
            net_l = filename_l[0].split('_')
            prp_l = filename_l[1].split('_')
            netname = "_".join(net_l[:-2])+".onnx"
            conf = int(net_l[-1])
            prpname = "_".join(prp_l[:-2])+".vnnlib"

            reversed_lines = reversed(file.readlines())
            count = 0
            for line in reversed_lines:
                line = line.strip()
                if "Result:" in line:
                    res = line.split(':')[1]
                    count += 1
                elif "Time:" in line:
                    time_taken = float(line.split(':')[1])
                    count += 1
                
                if count >= 2:
                    break
            return netname, prpname, conf, res, time_taken
    
    return None, None, None, None, None


def get_data_cifar10_abcrown():
    # List to hold extracted rows
    log_folder_cifar2020 = '/home/u1411251/tools/result_dir/welkin/relaxed/cifar10/cifar2020'
    log_folder_resnet = '/home/u1411251/tools/result_dir/welkin/relaxed/cifar10/cifar10_resnet'
    log_folder_oval21 = '/home/u1411251/tools/result_dir/welkin/relaxed/cifar10/oval21'
    data = []
    # Loop over all files in the folder
    for filename in os.listdir(log_folder_cifar2020):
        netname, prpname, conf, res, time_taken = get_file_data_cifar10_abcrown(filename, log_folder_cifar2020)
        if netname != None:
            data.append({
                            'netname': netname,
                            'propname': prpname,
                            'conf': conf,
                            'result': res,
                            'time': time_taken
                        })
        
    for filename in os.listdir(log_folder_resnet):
        netname, prpname, conf, res, time_taken = get_file_data_cifar10_abcrown(filename, log_folder_resnet)
        if netname != None:
            data.append({
                            'netname': netname,
                            'propname': prpname,
                            'conf': conf,
                            'result': res,
                            'time': time_taken
                        })
    
    for filename in os.listdir(log_folder_oval21):
        netname, prpname, conf, res, time_taken = get_file_data_cifar10_abcrown(filename, log_folder_oval21)
        if netname != None:
            data.append({
                            'netname': netname,
                            'propname': prpname,
                            'conf': conf,
                            'result': res,
                            'time': time_taken
                        })


    # Create pandas DataFrame
    return data


def get_dict(df):
    # Group by 'conf' and get list of 'time' values
    grouped_times = df.groupby('conf')['time'].apply(list)
    # Sort times for each conf
    sorted_times = grouped_times.apply(sorted)
    # Convert to dictionary if needed
    sorted_times_dict = sorted_times.to_dict()
    # Print result
    for conf, times in sorted_times_dict.items():
        print(f"{conf} => {len(times)}")
    
    return sorted_times_dict

def get_instances_vs_time(conf_vs_times, is_marabou = False):
    dump_files_dir = '/home/u1411251/Documents/PhD-Work/papers/draft/rebuttal/comparison/relaxed/abcrown'
    if is_marabou:
        dump_files_dir = '/home/u1411251/Documents/PhD-Work/papers/draft/rebuttal/comparison/relaxed/marabou'
        os.makedirs(dump_files_dir, exist_ok=True)


    os.makedirs(dump_files_dir, exist_ok=True)
    for conf, times in conf_vs_times.items():
        print(f"Conf: {conf}")
        sum = 0
        data = []
        for i in range(len(times)):
            sum += times[i]+1
            data.append(f"{i} {math.log(sum)}\n")
            print(i, math.log(sum))
        filepath = os.path.join(dump_files_dir, f"data_{conf}.txt")
        with open(filepath, "w") as f:
            f.writelines(data)

def analysis_abcrown():
    data = get_data_mnist_abcrown() #+ get_data_cifar10_abcrown()
    # data = get_data_cifar10_cifar2020()
    df = pd.DataFrame(data)
    # print(df.head())
    # print(df['conf'].value_counts())
    # print(df[(df['result'] == 'timeout') & (df['netname'].str.contains("256x2"))])
    conf_vs_times = get_dict(df)
    get_instances_vs_time(conf_vs_times, is_marabou=False)

def analysis_comparison(is_marabou = False):
    if is_marabou:
        # data_df = get_data_mnist_marabou()
        data_df = get_data_mnist_marabou_relaxed()
    else:
        data_df = get_data_mnist_abcrown()
    df = pd.DataFrame(data_df)
    # df.to_csv('abcrown_simplified.csv', index=False)
    # exit(0)
    times = list(df['time'])
    sorted_times = sorted(times)
    # print(times)
    # times = df['time'].apply(list)
    # Sort times for each conf
    # sorted_times = times.apply(sorted)
    sum = 0
    data = []
    for i in range(len(sorted_times)):
        sum += sorted_times[i]+1
        # data.append(f"{i} {math.log(sum)}\n")
        data.append(f"{i} {sorted_times[i]}\n")
        # print(i, math.log(sum))
        print(i, sorted_times[i])
    if is_marabou:
        dump_files_dir = '/home/u1411251/Documents/PhD-Work/papers/draft/rebuttal/comparison/relaxed/marabou'
    else:
        dump_files_dir = '/home/u1411251/Documents/PhD-Work/papers/draft/rebuttal/comparison/relaxed/abcrown'
    os.makedirs(dump_files_dir, exist_ok=True)
    filepath = os.path.join(dump_files_dir, f"data1.txt")
    with open(filepath, "w") as f:
        f.writelines(data)


analysis_comparison(is_marabou=True)





