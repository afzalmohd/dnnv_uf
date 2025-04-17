from z3 import *

# Define input variables (2D input)
x1, x2 = Real('x1'), Real('x2')
x1_prime, x2_prime = Real('x1_prime'), Real('x2_prime')
delta = 0.04  # Small perturbation threshold

def network_constrs():
    # Define weight and bias variables for the hidden layer (3 neurons)
    w1 = [[1,1],  
        [1,-1],  
        [-1,1]]  

    b1 = [0,0,0]

    # Define weight and bias variables for the output layer (2D output)
    w2 = [[1,0.5,-0.5],  
        [0.5,1,1]]

    b2 = [0,0]

    # Define the ReLU function
    def relu(x):
        return If(x >= 0, x, 0)

    # Compute hidden layer activations
    h1 = [relu(w1[i][0] * x1 + w1[i][1] * x2 + b1[i]) for i in range(3)]

    # Compute output layer
    output = [w2[i][0] * h1[0] + w2[i][1] * h1[1] + w2[i][2] * h1[2] + b2[i] for i in range(2)]

    # Define perturbed inputs


    # Compute hidden layer activations for perturbed input
    h1_prime = [relu(w1[i][0] * x1_prime + w1[i][1] * x2_prime + b1[i]) for i in range(3)]

    # Compute output layer for perturbed input
    output_prime = [w2[i][0] * h1_prime[0] + w2[i][1] * h1_prime[1] + w2[i][2] * h1_prime[2] + b2[i] for i in range(2)]

    return output, output_prime

def network_constrs_local():
    # Define weight and bias variables for the hidden layer (3 neurons)
    w1 = [[1,1],  
        [1,-1],  
        [-1,1]]  

    b1 = [0,0,0]

    # Define weight and bias variables for the output layer (2D output)
    w2 = [[1,0.5,-0.5],  
        [0.5,1,1]]

    b2 = [0,0]

    # Define the ReLU function
    def relu(x):
        return If(x >= 0, x, 0)

    # Compute hidden layer activations
    h1 = [relu(w1[i][0] * x1 + w1[i][1] * x2 + b1[i]) for i in range(3)]

    # Compute output layer
    output = [w2[i][0] * h1[0] + w2[i][1] * h1[1] + w2[i][2] * h1[2] + b2[i] for i in range(2)]

    # Define perturbed inputs


    # # Compute hidden layer activations for perturbed input
    # h1_prime = [relu(w1[i][0] * x1_prime + w1[i][1] * x2_prime + b1[i]) for i in range(3)]

    # # Compute output layer for perturbed input
    # output_prime = [w2[i][0] * h1_prime[0] + w2[i][1] * h1_prime[1] + w2[i][2] * h1_prime[2] + b2[i] for i in range(2)]

    return output


def global_robsutness():

    output, output_prime = network_constrs()

    # Global robustness constraint: |output - output_prime| <= delta for all inputs
    s = Solver()
    s.add(And(x1 >= 0, x1 <= 1), And(x2 >= 0, x2 <= 1))
    s.add(And(x1_prime >= 0, x1_prime <= 1), And(x2_prime >= 0, x2_prime <= 1))
    pre_constr = And(Abs(x1 - x1_prime) <= delta, Abs(x2 - x2_prime) <= delta)
    # s.add(And(Abs(x1 - x1_prime) <= delta, Abs(x2 - x2_prime) <= delta))
    s.add(And(pre_constr, output[0] - output[1] > 0.1), output_prime[0] - output_prime[1] <= 0.0)

    if s.check() == sat:
        model = s.model()
        # Check if the property holds universally
        def get_val(var, prec=6):
            val_str = s.model()[var].as_decimal(prec)
            # Remove trailing '?' if present
            if val_str.endswith('?'):
                val_str = val_str[:-1]
            try:
                return float(val_str)
            except ValueError:
                return val_str
            
        x1_val = get_val(x1)
        x2_val = get_val(x2)
        x1p_val = get_val(x1_prime)
        x2p_val = get_val(x2_prime)

        print("Satisfiable solution found:")
        print(f"x1 = {x1_val:.6f}")
        print(f"x2 = {x2_val:.6f}")
        print(f"x1_prime = {x1p_val:.6f}")
        print(f"x2_prime = {x2p_val:.6f}")
    else:
        print("Unsatisfiable.")
       

def local_robustness():
    # Global robustness constraint: |output - output_prime| <= delta for all inputs
    output = network_constrs_local()
    s = Solver()
    s.add(And(x1 >= 0.5, x1 <= 1), And(x2 >= 0, x2 <= 1))
    # s.add(And(Abs(x1 - x1_prime) <= delta, Abs(x2 - x2_prime) <= delta))
    s.add(And(output[0] - output[1] < 0.0))

    if s.check() == sat:
        model = s.model()
        # Check if the property holds universally
        def get_val(var, prec=6):
            val_str = s.model()[var].as_decimal(prec)
            # Remove trailing '?' if present
            if val_str.endswith('?'):
                val_str = val_str[:-1]
            try:
                return float(val_str)
            except ValueError:
                return val_str
            
        x1_val = get_val(x1)
        x2_val = get_val(x2)

        print("Satisfiable solution found:")
        print(f"x1 = {x1_val:.6f}")
        print(f"x2 = {x2_val:.6f}")
    else:
        print("Unsatisfiable.")

local_robustness()


