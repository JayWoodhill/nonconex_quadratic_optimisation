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
def matrix_checks(matrix, tol = 1e-5):
    # checks eigenvalue characteristics of user input matrix, returns summary
    # input param. tol
    # add e-10 as an absolute value tolerance for 0 eigenvalues
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    eig_val_zero = np.sum(np.abs(eigenvalues) < tol)
    eig_val_pos = np.sum(eigenvalues > tol)
    eig_val_neg = np.sum(eigenvalues < tol)
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
    # q to -q to simplify convex/concave
    # 'level' iterator counter
    # check rounded np values
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
        # break loop for complexity or zero pos/neg eigval count
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
            # all other input treated as concave
            if np.any(eigenvalues > 0):
                positive_eigenvalue_index = np.where(eigenvalues > 0)[0][0]
                positive_eigenvalue = eigenvalues[positive_eigenvalue_index]
                v1 = eigenvectors[:, positive_eigenvalue_index]
                u = np.ones(len(current_matrix))
                alpha = -positive_eigenvalue / np.dot(u, v1)
                current_matrix = current_matrix + alpha * np.outer(u, v1)
                # create a list of linear terms indexed to the level of rank reduction
                # aggregate the appended terms from q' > q''... and store that as a variable to be passed to gurobi

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

# run numerical experiments to investigate conditions that imply capacity for iterations of rank reduction
# matrix characteristics: size, rank proportion, eigenvalue weighting, symmetry, invertibility, condition numbering (accuracy of a solution after approximation)
# generate symmetric matrix with uniform random distrib. [0,100] - each element with a value that is included conditional upon being less than a density parameter.
# above is populr for generating values for validating integer programming methods