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
            return'''

        model = gp.Model("Qp")
        n = Q.shape[0]
        x = model.addMVar(shape=n, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="x")
        obj = x @ Q @ x
        model.setObjective(obj, GRB.MINIMIZE)
        model.addConstr(x.sum() == 1, name="sum_constraint")
        model.setParam('TimeLimit', 100)
        model.optimize()

        if model.status == GRB.OPTIMAL:
            solution = x.X
            print(f"Optimal solution for {matrix_name}: {solution}")
            print(f"Objective value for {matrix_name}: {model.objVal}")
        elif model.status == GRB.TIME_LIMIT:
            print(f"Solver timed out after 10 seconds for {matrix_name}.")
        else:
            print(f"No optimal solution found for {matrix_name}. Status: {model.status}")

    except gp.GurobiError as e:
        print(f"Gurobi error in solving {matrix_name}: {e}")
    except Exception as e:
        print(f"An error occurred in solving {matrix_name}: {e}")

def verify_solution(Q, solution):
    if solution is not None:
        result = solution @ Q @ solution
        print(f"Solution verification (x^T Q x) result: {result}")
    else:
        print("No solution to verify.")

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

Q = gen_real_constrained_matrix(n, u, v)
print(np.linalg.eig(Q)[0])

initial_check = matrix_checks(Q)

Q = iterative_rank_reduction(Q)["matrices"][-1]
print(np.linalg.eig(Q)[0])

Q = (Q + Q.T)
print(np.linalg.eig(Q)[0])
