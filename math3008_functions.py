import gurobipy as gp
import numpy as np
import numpy.linalg as  linalg
import scipy
from gurobipy import GRB

def gen_real_constrained_matrix(n, u, v, tol = 1e-5):
    # this function creates a square matrix 'n' with a rank of u + v, and count sums of positive/gative eigenvalues of u and v respectively
    # added optional tolerance threshold
    if u + v  > n:
        raise ValueError("excessive eigenvalue count")
    # sampling changed to half normal pos/neg
    eig_val_pos = np.abs(np.random.normal(0, 10, u))
    eig_val_neg = -np.abs(np.random.uniform(0, 10, v))
    eig_val_zero = np.zeros(n - u - v)
    eig_vals = np.concatenate([eig_val_pos, eig_val_neg, eig_val_zero])

    # potential infinite loop of qr decomps to get sufficient rank. probably need to fix
    while True:
        P, _ = np.linalg.qr(np.random.rand(n, n))
        if np.linalg.matrix_rank(P) == n:
            break

    D = np.diag(eig_vals)

    constrained_matrix = P @ D @ P.T

    if constrained_matrix.shape != (n, n):
        raise ValueError("output matrix dimension error")

    # check using linalg eig functions, count from list of eigenvalues with tolerance for imprecise 0 values
    # np.linalg.eigvalsh()
    output_eigenvalues = np.linalg.eigvalsh(constrained_matrix)

    # tolerance tol
    count_list = [np.sum(output_eigenvalues < -tol),
        np.sum(output_eigenvalues > tol),
        np.sum(np.isclose(output_eigenvalues, 0, atol = tol))
    ]

    # check count]
    if count_list[1] != u or count_list[0] != v or count_list[2] != (n - u - v):
        raise ValueError(
            "eigenvalue count error:\n"
            f"exp: {u}, {v}, {n - u - v}\n"
            f"actual:{count_list[1]}, {count_list[0]}, {count_list[2]}")
    else:
        print("working")

    return constrained_matrix
def matrix_checks(matrix):
    # checks eigenvalue characteristics of user input matrix, returns summary
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    eig_val_pos = np.sum(eigenvalues >0)
    eig_val_neg = np.sum(eigenvalues < 00)
    eig_val_zero = np.sum(np.isclose(eigenvalues,0))
    eig_val_complex = np.sum(np.iscomplex(eigenvalues))

    positive_semidefinite = np.all(eigenvalues >= 0)
    invertible = not np.isclose(np.linalg.det(matrix), 0)

    return {
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "positive_eigenvalue_count": eig_val_pos,
        "negative_eigenvalue_count": eig_val_neg,
        "zero_eigenvalue_count": eig_val_zero,
        "complex_eigenvalue_count": eig_val_complex,
        "is_positive_semidefinite": positive_semidefinite,
        "is_invertible": invertible
    }
def matrix_checks_and_output_eigenvalues(matrix):
    check_results = matrix_checks(matrix)

    eigenvalues = check_results["eigenvalues"]
    eigenvectors = check_results["eigenvectors"]
    eig_val_neg = check_results["negative_eigenvalue_count"]

    if eig_val_neg > 0:
        negative_eigenvalue_index = np.where(eigenvalues < 0)[0][0]
        negative_eigenvalue = eigenvalues[negative_eigenvalue_index]

        v1 = eigenvectors[:, negative_eigenvalue_index]
        u = np.ones(len(matrix))

        alpha = -negative_eigenvalue / np.dot(u, v1)
        matrix_prime = matrix + alpha * np.outer(u, v1)

        modified_eigenvalues, _ = np.linalg.eig(matrix_prime)

        return {
            "Q": matrix,
            "Q_eigenvalues": eigenvalues,
            "Q_prime": matrix_prime,
            "Q_prime_eigenvalues": modified_eigenvalues
        }

    return {
        "Q": matrix,
        "Q_eigenvalues": eigenvalues,
        "Q_prime": None,
        "Q_prime_eigenvalues": None
    }


def iterative_rank_reduction(q, mode='convex'):
    matrices = []
    eigenvalues_set = []
    eigenvectors_set = []
    level = 0

    current_matrix = q
    convex = (mode == 'convex')

    while True:
        check_results = matrix_checks(current_matrix)
        eigenvalues = check_results["eigenvalues"]
        eigenvectors = check_results["eigenvectors"]
        complex_eigenvalues = check_results["complex_eigenvalue_count"]

        matrices.append(np.round(current_matrix, 2))
        eigenvalues_set.append(np.round(eigenvalues, 2))
        eigenvectors_set.append(np.round(eigenvectors, 2))

        if complex_eigenvalues > 0:
            print(f"complexity after {level}")
            break
        if convex and np.all(eigenvalues >= 0):
            print(f"negatives removed after {level} levels")
            break
        if not convex and np.all(eigenvalues <= 0):
            print(f"positives removed after {level} levels")
            break

        if convex:
            if np.any(eigenvalues < 0):
                negative_eigenvalue_index = np.where(eigenvalues < 0)[0][0]
                negative_eigenvalue = eigenvalues[negative_eigenvalue_index]
                v1 = eigenvectors[:, negative_eigenvalue_index]
                u = np.ones(len(current_matrix))
                alpha = -negative_eigenvalue / np.dot(u, v1)
                current_matrix = current_matrix + alpha * np.outer(u, v1)
        else:
            if np.any(eigenvalues > 0):
                positive_eigenvalue_index = np.where(eigenvalues > 0)[0][0]
                positive_eigenvalue = eigenvalues[positive_eigenvalue_index]
                v1 = eigenvectors[:, positive_eigenvalue_index]
                u = np.ones(len(current_matrix))
                alpha = -positive_eigenvalue / np.dot(u, v1)
                current_matrix = current_matrix + alpha * np.outer(u, v1)

        level += 1

    print(f"levels rduced: {level}")
    print(f"eigenvalue: {eigenvalues_set[-1]}")

    return {
        "matrices": matrices,
        "eigenvalues_set": eigenvalues_set,
        "eigenvectors_set": eigenvectors_set,
        "levels_performed": level,
        "final_status": "complexity reached" if complex_eigenvalues > 0 else "unipolar set achieved"
    }

# sample test
Q = np.array([[3, 2, 1],
              [2, 4, 2],
              [1, 2, 3]])
eigenvalues, eigenvectors = np.linalg.eig(Q)

# test
def test_case_euclidean_opt(matrix):
    print("Q:")
    print(matrix)
    print("\nQ eigenvalue:")
    print(np.round(eigenvalues, 2))
    print("\nq eigenvector:")
    print(np.round(eigenvectors, 2))

#
# -1 (Automatic), 0 (Primal Simplex), 1 (Dual Simplex), 2 (Barrier), 3 (Concurrent), 4 (Deterministic Concurrent), 5 (Concurrent with Multiple Solvers)

#sample is currently rank7, with single negative eigenvalue
sample_matrix = np.array([[3.65343869e+00, -3.63643035e+00, -1.22365481e+00,
               2.23445874e-03, -1.34136466e+00, 4.31834447e-01,
               2.10979868e-01],
              [-3.63643035e+00, 3.57634805e+00, -6.86883584e-01,
               -7.21141937e-01, -2.03432810e+00, 1.65735656e+00,
               -6.09766959e-02],
              [-1.22365481e+00, -6.86883584e-01, 3.93904698e+00,
               -2.37344514e-02, -3.66910103e-01, 1.86375885e+00,
               -1.28546680e-01],
              [2.23445874e-03, -7.21141937e-01, -2.37344514e-02,
               4.24400364e+00, 6.84109392e-01, 1.05768926e+00,
               -2.26511390e-01],
              [-1.34136466e+00, -2.03432810e+00, -3.66910103e-01,
               6.84109392e-01, 5.52537750e+00, 5.05568064e-01,
               1.46393093e-01],
              [4.31834447e-01, 1.65735656e+00, 1.86375885e+00,
               1.05768926e+00, 5.05568064e-01, 4.59171547e+00,
               -8.52707541e-01],
              [2.10979868e-01, -6.09766959e-02, -1.28546680e-01,
               -2.26511390e-01, 1.46393093e-01, -8.52707541e-01,
               5.47006967e+00]])

#Q=sample_matrix
# -1 (Automatic), 0 (Primal Simplex), 1 (Dual Simplex), 2 (Barrier), 3 (Concurrent), 4 (Deterministic Concurrent), 5 (Concurrent with Multiple Solvers)
# https://www.gurobi.com/documentation/
#
Q = sample_matrix
Q = iterative_rank_reduction(Q)["matrices"][-1]
#print(Q)
#print(np.linalg.eig(Q))
# why does gurobi think this is non-convex
#
#print("count:"+str(np.sum(np.linalg.eig(Q)[0] < 0)))

matrix_test = gen_real_constrained_matrix(3,1,1)
print("matrix test:"+str(matrix_test))
print(np.linalg.eig(matrix_test))
Q = iterative_rank_reduction(matrix_test)["matrices"][-1]
print(Q)



'''model = gp.Model("Quadratic_Optimization")
n = Q.shape[0]
x = model.addMVar(shape=n, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="x")
obj = x @ Q @ x
model.setObjective(obj, GRB.MINIMIZE)
model.addConstr(x.sum() == 1, name="sum_constraint")
model.optimize()

if model.status == GRB.OPTIMAL:
    solution = x.X
    print("x:", solution)
    print("val:", model.objVal)
else:
    print("no solution")'''


#
#test_case_euclidean_optimization(Q)

#test
'''test_matrix = np.array([[4, 2, 0],
                        [2, -3, 0],
                        [0, 0, 1]])'''
#result = iterative_rank_reduction(test_matrix, mode='convex')

# tests
'''test_matrix = np.array([[4, 2, 0],
                        [2, -3, 0],
                        [0, 0, 1]])

result = matrix_checks_and_output_eigenvalues(test_matrix)
print(result)'''
# check all gurobi convex solvers
methods = [
    {"name": "prim sim", "method": 0},
    {"name": "dual sim", "method": 1},
    {"name": "barrier", "method": 2},
    {"name": "concurrent", "method": 3},
    {"name": "deterministic concurrent", "method": 4}
]
'''for solver in methods:
    print(f"\nsolver {solver['name']} method {solver['method']})")

    try:
        model = gp.Model("math3008_qp")
        # methods
        model.setParam("Method", solver['method'])
        n = Q.shape[0]
        x = model.addMVar(shape=n, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="x")
        obj = x @ Q @ x
        model.setParam("TimeLimit", 10)
        model.setObjective(obj, GRB.MINIMIZE)
        model.addConstr(x.sum() == 1, name="sum_constraint")
        model.optimize()
        if model.status == GRB.OPTIMAL:
            solution = x.X
            print(f"solution {solution}")
            print(f"value {model.objVal}")
        else:
            print(f"not found {model.status}")

    except gp.GurobiError as e:
        print(f"Gurobi error occurred: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")'''

# more precise eigenvaluecheck
'''eigenvalues = np.linalg.eigvals(Q)
np.set_printoptions(precision=16)  
print("Eigenvalues:", eigenvalues)'''



