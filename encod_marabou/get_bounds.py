import re
from collections import defaultdict

def get_label_vnncomp_prp(prp_file, is_less_than_output_prp=False, is_target_prop = False):
    label = None
    with open(prp_file, 'r') as file:
        lines = file.readlines()  # Read all lines into a list
        last_lines = lines[-10:]
        last_content = ''.join(last_lines)
        matches = re.findall(r'Y_(\d+)', last_content)
        if is_less_than_output_prp:
            if is_target_prop:
                label = int(matches[-1])
            else:
                label = int(matches[0])
        else:
            if is_target_prop:
                label = int(matches[0])
            else:
                label = int(matches[-1])

        return label

def extract_bounds_from_file(filename):
    bounds = defaultdict(lambda: {"lower": None, "upper": None})

    # Regular expressions to match constraints
    lower_bound_pattern = re.compile(r"\(assert \(>= (X_\d+) ([\d.eE+-]+)\)\)")
    upper_bound_pattern = re.compile(r"\(assert \(<= (X_\d+) ([\d.eE+-]+)\)\)")

    # Read the file line by line
    lbs, ubs = [], []
    with open(filename, "r") as file:
        for line in file:
            lower_match = lower_bound_pattern.match(line)
            upper_match = upper_bound_pattern.match(line)

            if lower_match:
                var, value = lower_match.groups()
                bounds[var]["lower"] = float(value)
                lbs.append(float(value))

            if upper_match:
                var, value = upper_match.groups()
                bounds[var]["upper"] = float(value)
                ubs.append(float(value))

    return lbs, ubs


# filename = '/home/u1411251/tools/vnncomp_benchmarks/cifar10/cifar2020/vnnlib/cifar10_spec_idx_0_eps_0.00784_n1.vnnlib'
# lbs, ubs = extract_bounds_from_file(filename)
# i=0
# for l,u in zip(lbs, ubs):
#     print(i,l,u)
#     i += 1
