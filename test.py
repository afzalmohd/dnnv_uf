import re
def get_label_vnncomp_prp(prp_file):
    # with open(prp_file, 'r') as file:
    #     first_line = file.readline().strip()
    #     # Extract the number after 'label: '
    #     if "property with label:" in first_line:
    #         label = first_line.split("label:")[-1].strip().rstrip('.')
    #         return int(label) 
        
    with open(prp_file, 'r') as file:
        lines = file.readlines()  # Read all lines into a list
        last_lines = lines[-10:]
        last_content = ''.join(last_lines)
        matches = re.findall(r'Y_(\d+)', last_content)
        return matches[-1]
    
prp_file = '/home/u1411251/tools/vnncomp_benchmarks/cifar10/cifar2020/vnnlib/cifar10_spec_idx_50_eps_0.03137_n1.vnnlib'

lb = get_label_vnncomp_prp(prp_file)
print(lb)