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
  

    

def mnist_encoding_relax(net_path, prp_path, conf, num_workers):
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
    options._numWorkers = num_workers
    exitCode1, vals1, stats1 = net1.solve(options=options)
    print(f"my_result:{exitCode1}")
        
if __name__ == '__main__':
    if len(sys.argv) == 5:
        net_path = str(sys.argv[1])
        prp_path = str(sys.argv[2])
        conf = int(sys.argv[3])
        num_workers = int(sys.argv[4])
    	
    mnist_encoding_relax(net_path, prp_path, conf=conf, num_workers=num_workers)








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
