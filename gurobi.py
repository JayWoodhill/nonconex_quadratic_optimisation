import gurobipy as gp
import numpy as np
import numpy.linalg as linalg
import scipy
from gurobipy import GRB
from math3008_function import gen_real_constrained_matrix, matrix_checks, iterative_rank_reduction


def solve_quadratic(Q, matrix_name="matrix"):
    try:
        eigenvalues = np.linalg.eigvals(Q)

        '''if np.any(eigenvalues < 0):
            print(f"Matrix '{matrix_name}' is non-convex (contains negative eigenvalues), refusing to solve.")
            return None, None'''

        model = gp.Model("Quadratic_Optimization")
        n = Q.shape[0]
        x = model.addMVar(shape=n, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="x")
        obj = x @ Q @ x
        model.setObjective(obj, GRB.MINIMIZE)
        model.addConstr(x.sum() == 1, name="sum_constraint")
        model.setParam('TimeLimit', 10)
        model.optimize()

        if model.status == GRB.OPTIMAL:
            solution = x.X
            objective_value = model.objVal
            print(f"Optimal solution for {matrix_name}: {solution}")
            print(f"Objective value for {matrix_name}: {objective_value}")
            return solution, objective_value
        elif model.status == GRB.TIME_LIMIT:
            print(f"Solver timed out after 10 seconds for {matrix_name}.")
            return None, None
        else:
            print(f"No optimal solution found for {matrix_name}. Status: {model.status}")
            return None, None

    except gp.GurobiError as e:
        print(f"Gurobi error in solving {matrix_name}: {e}")
        return None, None
    except Exception as e:
        print(f"An error occurred in solving {matrix_name}: {e}")
        return None, None


def verify_solution(Q, solution, gurobi_obj_value):
    if solution is not None:
        manual_obj_value = solution @ Q @ solution
        print(f"Recomputed objective value (x^T Q x): {manual_obj_value}")
        print(f"Gurobi's objective value: {gurobi_obj_value}")
        if np.isclose(manual_obj_value, gurobi_obj_value, atol=1e-5):
            print("The solution is verified as correct.")
        else:
            print("The solution does not match the expected objective value.")
    else:
        print("No solution to verify.")


n = 7
u = 6
v = 1

matrix = gen_real_constrained_matrix(n, u, v)

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

if original_solution is not None and original_objective is not None:
    print("\nVerifying the solution for the original matrix:")
    verify_solution(matrix, original_solution, original_objective)
else:
    print("No solution for the original matrix.")

print("\nSolving the convex matrix after rank reduction with Gurobi:")
transformed_solution, transformed_objective = solve_quadratic(final_matrix, matrix_name="Transformed Convex Matrix")

if transformed_solution is not None and transformed_objective is not None:
    print("\nVerifying the solution for the transformed matrix:")
    verify_solution(final_matrix, transformed_solution, transformed_objective)
else:
    print("No solution for the transformed matrix.")