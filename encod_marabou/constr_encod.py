'''
Disjunction Constraint Example
====================

Top contributors (to current version):
  - Haoze Andrew Wu

This file is part of the Marabou project.
Copyright (c) 2017-2024 by the authors listed in the file AUTHORS
in the top-level source directory) and their institutional affiliations.
All rights reserved. See the file COPYING in the top-level source
directory for licensing information.
'''

import sys
import os
print(sys.path)
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
print(sys.path)
import numpy as np
import math
import time
## %
# Path to Marabou folder if you did not export it

# sys.path.append('/home/USER/git/Marabou')
from maraboupy import Marabou
from maraboupy import MarabouUtils
from maraboupy import MarabouCore
from maraboupy.MarabouCore import StatisticsUnsignedAttribute
from get_bounds import extract_bounds_from_file, get_label_vnncomp_prp


def get_delta(conf):
    delta_th = -math.log((100/conf) - 1)
    delta_th = round(delta_th, 3)
    return delta_th
  

def mnist_encoding_standard(net_path, prp_path, num_workers):
    net1 = Marabou.read_onnx(net_path)
    input_vars = net1.inputVars[0].flatten()
    num_input_vars = input_vars.size
    outputVars = net1.outputVars[0].flatten()
    num_output_vars = outputVars.size
    lbs, ubs = extract_bounds_from_file(prp_path)
    label = get_label_vnncomp_prp(prp_path)
    print(f"Check: {input_vars.size} , {len(lbs)}")

    idx = 0
    for var in input_vars:
      net1.setLowerBound(var, lbs[idx])
      net1.setUpperBound(var, ubs[idx])
      idx += 1

    for i in range(num_output_vars):
        if i != label:
            eq = MarabouUtils.Equation(MarabouCore.Equation.GE)
            eq.addAddend(1, outputVars[i])
            eq.addAddend(1, outputVars[label])
            eq.setScalar(0.0)
      
    options = MarabouCore.Options()
    # options._numWorkers = num_workers
    start_time = time.time()
    exitCode1, vals1, stats1 = net1.solve(options=options)
    end_time = time.time()
    print(f"my_result:{exitCode1}")
    print(f"my_timetaken:{end_time-start_time}")

def mnist_encoding_relax(net_path, prp_path, conf, num_workers):
    if conf == 0.0:
        mnist_encoding_standard(net_path, prp_path, num_workers)
        return
    delta = get_delta(conf)
    net1 = Marabou.read_onnx(net_path)
    input_vars = net1.inputVars[0].flatten()
    num_input_vars = input_vars.size
    outputVars = net1.outputVars[0].flatten()
    num_output_vars = outputVars.size
    lbs, ubs = extract_bounds_from_file(prp_path)
    label = get_label_vnncomp_prp(prp_path)
    print(f"Check: {input_vars.size} , {len(lbs)}")

    idx = 0
    for var in input_vars:
      net1.setLowerBound(var, lbs[idx])
      net1.setUpperBound(var, ubs[idx])
      idx += 1

    dnf1 = []
    for i in range(num_output_vars):
        if i != label:
            conj = []
            for j in range(num_output_vars):
                if j != i:
                  eq = MarabouUtils.Equation(MarabouCore.Equation.GE)
                  eq.addAddend(1, outputVars[i])
                  eq.addAddend(-1, outputVars[j])
                  eq.setScalar(delta)
                  conj.append(eq)
            dnf1.append(conj)
      
    net1.addDisjunctionConstraint(dnf1)
    options = MarabouCore.Options()
    # options._numWorkers = num_workers
    start_time = time.time()
    exitCode1, vals1, stats1 = net1.solve(options=options)
    end_time = time.time()
    print(f"my_result:{exitCode1}")
    print(f"my_timetaken:{end_time-start_time}")


def mnist_encoding_relax_append_net(net_path, prp_path, num_workers):
    net1 = Marabou.read_onnx(net_path)
    input_vars = net1.inputVars[0].flatten()
    num_input_vars = input_vars.size
    outputVars = net1.outputVars[0].flatten()
    num_output_vars = outputVars.size
    lbs, ubs = extract_bounds_from_file(prp_path)
    print(f"Check: {input_vars.size} , {len(lbs)}")
    
    idx = 0
    for var in input_vars:
      net1.setLowerBound(var, lbs[idx])
      net1.setUpperBound(var, ubs[idx])
      idx += 1

    dnf1 = []
    for i in range(num_output_vars):
        eq = MarabouUtils.Equation(MarabouCore.Equation.GE)
        eq.addAddend(1, outputVars[i])
        eq.setScalar(-0.0001)
        dnf1.append([eq])

    net1.addDisjunctionConstraint(dnf1)

    options = MarabouCore.Options()
    # options._numWorkers = num_workers
    start_time = time.time()
    exitCode1, vals1, stats1 = net1.solve(options=options)
    end_time = time.time()
    print(f"my_result:{exitCode1}")
    print(f"my_timetaken:{end_time-start_time}")

def get_topk(netname, prp_name, conf_file):
    with open(conf_file, 'r') as f:
        Lines = f.readlines()
        for line in Lines:
            line = line.strip()
            line_l = line.split(',')
            net = line_l[0]
            prp = line_l[1]
            if (netname in net) and (prp_name in prp):
                print(f"Matched: {netname}, {net}, {prp_name}, {prp}")
                topk = [int(line_l[3]), int(line_l[4]), int(line_l[5])] 
                return topk

def mnist_encoding_topk(net_path, prp_path, conf_file):
    k = 2
    # if conf == 0.0:
    #     mnist_encoding_standard(net_path, prp_path, num_workers)
    # delta = get_delta(conf)
    net1 = Marabou.read_onnx(net_path)
    input_vars = net1.inputVars[0].flatten()
    num_input_vars = input_vars.size
    outputVars = net1.outputVars[0].flatten()
    num_output_vars = outputVars.size
    lbs, ubs = extract_bounds_from_file(prp_path)
    label = get_label_vnncomp_prp(prp_path)
    print(f"Check: {input_vars.size} , {len(lbs)}")
    top2_labels = get_topk(os.path.basename(net_path), os.path.basename(prp_path), conf_file)
    top2_labels = top2_labels[:k]
    idx = 0
    for var in input_vars:
      net1.setLowerBound(var, lbs[idx])
      net1.setUpperBound(var, ubs[idx])
      idx += 1

    dnf1 = []
    for i in range(num_output_vars):
        if i != label:
            eq = MarabouUtils.Equation(MarabouCore.Equation.GE)
            eq.addAddend(1, outputVars[i])
            eq.addAddend(-1, outputVars[label])
            eq.setScalar(0.0)
            dnf1.append([eq])
      
    net1.addDisjunctionConstraint(dnf1)

    dnf2 = []
    for i in range(num_output_vars):
        if i not in top2_labels:
            for j in top2_labels:
                eq = MarabouUtils.Equation(MarabouCore.Equation.GE)
                eq.addAddend(1, outputVars[i])
                eq.addAddend(-1, outputVars[j])
                eq.setScalar(0.0)
                dnf2.append([eq])

    net1.addDisjunctionConstraint(dnf2)
    options = MarabouCore.Options()
    # options._numWorkers = num_workers
    exitCode1, vals1, stats1 = net1.solve(options=options)
    print(f"my_result:{exitCode1}")


def mnist_encoding_appended(net_path, prp_path, conf_file):
    k = 2
    # if conf == 0.0:
    #     mnist_encoding_standard(net_path, prp_path, num_workers)
    # delta = get_delta(conf)
    net1 = Marabou.read_onnx(net_path)
    input_vars = net1.inputVars[0].flatten()
    num_input_vars = input_vars.size
    outputVars = net1.outputVars[0].flatten()
    num_output_vars = outputVars.size
    lbs, ubs = extract_bounds_from_file(prp_path)
    label = get_label_vnncomp_prp(prp_path)
    print(f"Check: {input_vars.size} , {len(lbs)}")
    top2_labels = get_topk(os.path.basename(net_path), os.path.basename(prp_path), conf_file)
    top2_labels = top2_labels[:k]
    idx = 0
    for var in input_vars:
      net1.setLowerBound(var, lbs[idx])
      net1.setUpperBound(var, ubs[idx])
      idx += 1

    dnf1 = []
    for i in range(num_output_vars):
        if i != label:
            eq = MarabouUtils.Equation(MarabouCore.Equation.GE)
            eq.addAddend(1, outputVars[i])
            eq.addAddend(-1, outputVars[label])
            eq.setScalar(0.0)
            dnf1.append([eq])
      
    net1.addDisjunctionConstraint(dnf1)

    dnf2 = []
    for i in range(num_output_vars):
        if i not in top2_labels:
            for j in top2_labels:
                eq = MarabouUtils.Equation(MarabouCore.Equation.GE)
                eq.addAddend(1, outputVars[i])
                eq.addAddend(-1, outputVars[j])
                eq.setScalar(0.0)
                dnf2.append([eq])

    net1.addDisjunctionConstraint(dnf2)
    options = MarabouCore.Options()
    # options._numWorkers = num_workers
    exitCode1, vals1, stats1 = net1.solve(options=options)
    print(f"my_result:{exitCode1}")


# def mnist_encoding_relax(net_path, prp_path, conf, num_workers):
#     delta = get_delta(conf)
#     net1 = Marabou.read_onnx(net_path)
#     input_vars = net1.inputVars[0].reshape(-1)
#     num_input_vars = input_vars.size
#     outputVars = net1.outputVars[0].flatten()
#     num_output_vars = outputVars.size
#     lbs, ubs = extract_bounds_from_file(prp_path)
#     label = get_label_vnncomp_prp(prp_path, is_less_than_output_prp=False)
#     print(f"Check: {input_vars.size} , {len(lbs)}")

#     idx = 0
#     for var in input_vars:
#       net1.setLowerBound(var, lbs[idx])
#       net1.setUpperBound(var, ubs[idx])
#       idx += 1

#     dnf1 = []
#     for i in range(num_output_vars):
#         if i != label:
#             conj = []
#             for j in range(num_output_vars):
#                 if j != i:
#                   eq = MarabouUtils.Equation(MarabouCore.Equation.GE)
#                   eq.addAddend(1, outputVars[i])
#                   eq.addAddend(-1, outputVars[j])
#                   eq.setScalar(delta)
#                   conj.append(eq)
#             dnf1.append(conj)
      
#     net1.addDisjunctionConstraint(dnf1)
#     options = MarabouCore.Options()
#     options._numWorkers = num_workers
#     exitCode1, vals1, stats1 = net1.solve(options=options)
#     print(f"my_result:{exitCode1}")



        
if __name__ == '__main__':
    if len(sys.argv) == 7:
        net_path = str(sys.argv[1])
        prp_path = str(sys.argv[2])
        conf = int(sys.argv[3])
        num_workers = int(sys.argv[4])
        prp_type = sys.argv[5]
        conf_file = sys.argv[6]

    # mnist_encoding_topk(net_path, prp_path, conf_file)
    # mnist_encoding_relax_append_net(net_path, prp_path, num_workers=num_workers)
    
    if prp_type == 'relaxed':
        mnist_encoding_relax_append_net(net_path, prp_path, num_workers=num_workers)
    else:
        mnist_encoding_relax(net_path, prp_path, conf = conf, num_workers=num_workers)








# Solve Marabou query
# exitCode1, vals1, stats1 = net1.solve()

# # %%
# # Example statistics
# stats1.getUnsignedAttribute(StatisticsUnsignedAttribute.NUM_SPLITS)
# stats1.getTotalTimeInMicro()


# # %%
# # Eval example
# #
# # Test that the satisfying assignment found is a real one.
# for i in range(784):
#     assert(abs(vals1[i] - 1) < 0.0000001 or abs(vals1[i]) < 0.0000001)
