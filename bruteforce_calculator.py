import itertools
import numpy as np
import time

def generate_feasible_assignments(K, num_pilots):
    # First UE is fixed to the first pilot
    fixed_assignment = [1]

    # Generate assignments for the remaining K-1 UEs
    remaining_UEs = itertools.product(range(1, num_pilots + 1), repeat=K - 1)

    feasible_assignments = []

    for assignment in remaining_UEs:
        # Combine fixed pilot with current assignment
        full_assignment = fixed_assignment + list(assignment)
        feasible_assignments.append(full_assignment)
    feasible_assignments = np.array(feasible_assignments)
    return feasible_assignments-1


# # Example usage:
# K = 10 # Number of UEs
# tau_p = 5
# BF_start = time.perf_counter()
# feasible_assignments = generate_feasible_assignments(K, tau_p)
# BF_end = time.perf_counter()
# feasible_assignments = feasible_assignments
# # for i, assignment in enumerate(feasible_assignments, 1):
# #     print(f"Assignment {i}: \t{assignment}")
#
# # Print the total number of assignments
# print(f"Total feasible assignments: {len(feasible_assignments)}")
# print(f'Total running time just for create bruteforce list: {BF_end-BF_start}')