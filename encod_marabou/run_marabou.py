import time
import sys
## %
# Path to Marabou folder if you did not export it

# sys.path.append('/home/USER/git/Marabou')
from maraboupy import Marabou
from maraboupy import MarabouUtils
from maraboupy import MarabouCore
from maraboupy.MarabouCore import StatisticsUnsignedAttribute
from get_bounds import extract_bounds_from_file, get_label_vnncomp_prp


def mnist_encoding_standard(net_path, prp_path):
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

    dnf = []
    for i in range(num_output_vars):
        if i != label:
            eq = MarabouUtils.Equation(MarabouCore.Equation.GE)
            eq.addAddend(1, outputVars[i])
            eq.addAddend(-1, outputVars[label])
            eq.setScalar(0.0)
            dnf.append([eq])
      
    net1.addDisjunctionConstraint(dnf)

    options = MarabouCore.Options()
    start_time = time.time()
    exitCode1, vals1, stats1 = net1.solve(options=options)
    end_time = time.time()
    print(f"my_result:{exitCode1}")
    print(f"my_timetaken:{end_time-start_time}")

if __name__ == '__main__':
    if len(sys.argv) == 4:
        net_path = str(sys.argv[1])
        prp_path = str(sys.argv[2])
        prp_type = sys.argv[3]
    
    else:
        print('provide proper arguments')
        sys.exit(0)

    if prp_type == 'fp':
        pass
    else:
        mnist_encoding_standard(net_path=net_path, prp_path=prp_path)

    