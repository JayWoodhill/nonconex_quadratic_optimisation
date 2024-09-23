import random

# matrix generation validation
def generate_test_set(limit_n, number_rows):
    test_sets = []
    for _ in range(number_rows):
        n = random.randint(1, limit_n)
        u = random.randint(0, n)
        v = random.randint(0, n - u)
        test_sets.append((n, u, v))
    return test_sets


fp="C:\\Users\\JayWood\\Documents\\Data\\test_inputs.txt"

test_sets = generate_test_set(10,5)

with open(fp, "w") as file:
    for n, u, v in test_sets:
        file.write(f"{n} {u} {v}\n")

print(f"done {fp}")

#===========================================================


print("\nInitial matrix eigenvalue check:")
initial_check = matrix_checks(matrix)
print("Eigenvalues:", initial_check["eigenvalues"])
print("Is the matrix positive semidefinite?", initial_check["is_positive_semidefinite"])
print("Negative eigenvalue count:", initial_check["negative_eigenvalue_count"])

print("\nApplying rank reduction to remove the negative eigenvalue...")
reduction_results = iterative_rank_reduction(matrix)

final_matrix = reduction_results["matrices"][-1]

print("\nAttempting to solve the original non-convex matrix with Gurobi:")
original_solution, original_objective = solve_quadratic(matrix, matrix_name="Original Non-Convex Matrix")

print("\nSolving the convex matrix after rank reduction with Gurobi:")
transformed_solution, transformed_objective = solve_quadratic(final_matrix, matrix_name="Transformed Convex Matrix")

# Verify the solutions for both original and transformed matrices
print("\nVerifying the solution for the original matrix:")
verify_solution(matrix, original_solution, original_objective)

print("\nVerifying the solution for the transformed matrix:")
verify_solution(final_matrix, transformed_solution, transformed_objective)

#===========================================================