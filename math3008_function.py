import gurobipy as gp
import numpy as np
import numpy.linalg as  linalg
import scipy
from gurobipy import GRB

def gen_real_constrained_matrix(n, u, v, tol = 1e-5):
    # this function creates a square matrix 'n' with a rank of u + v, and count sums of positive/gative eigenvalues of u and v respectively
    # added optional tolerance threshold
    # added max_attempts to avoid infinite looping of matrix generator
    if u + v  > n:
        raise ValueError("excessive eigenvalue count")

    eig_val_pos = np.abs(np.random.normal(0, 10, u))
    eig_val_neg = -np.abs(np.random.uniform(0, 10, v))
    eig_val_zero = np.zeros(n - u - v)
    eig_vals = np.concatenate([eig_val_pos, eig_val_neg, eig_val_zero])

    max_attempts = 100
    for attempt in range(max_attempts):
        P, _ = np.linalg.qr(np.random.rand(n, n))
        if np.linalg.matrix_rank(P) == n:
            break
    else:
        raise ValueError(f"Failed to generate a full-rank orthogonal matrix P after {max_attempts} attempts.")


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
    """
        Iteratively adjusts the input symmetric matrix `q` to make it positive semidefinite (`mode='convex'`)
        or negative semidefinite (`mode='concave'`) by eliminating undesired eigenvalues through rank-one updates.

        Parameters:
        - q (numpy.ndarray): The input symmetric matrix to be adjusted.
        - mode (str): 'convex' to make the matrix positive semidefinite, 'concave' for negative semidefinite.

        Returns:
        - dict: Contains the matrices, eigenvalues, eigenvectors at each iteration,
                the number of iterations performed, and the final status.

        Working notes 23/09:
        Adding tol
        Adding -q transformation for concave cases and identifier within function
        Protecting input q from mutability
        Fixed typos
        Cleaner presentation of results
        """
    matrices = []
    eigenvalues_set = []
    eigenvectors_set = []
    level = 0

    # q adjustment to simplify code
    if mode == 'concave':
        q = -q.copy()
        adjusted_for_concave = True
    else:
        q = q.copy()
        adjusted_for_concave = False

    current_matrix = q
    tol = 1e-10

    # Adjusted iterator based on pre-iteration conversion of q/-q
    while True:
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(current_matrix)

        # Store the current state
        matrices.append(current_matrix.copy())
        eigenvalues_set.append(eigenvalues.copy())
        eigenvectors_set.append(eigenvectors.copy())

        # Check for complex eigenvalues (shouldn't happen)
        if np.iscomplexobj(eigenvalues):
            final_status = f"Complex eigenvalues encountered after {level} iterations."
            break

        # Check for positive semidefinite
        if np.all(eigenvalues >= -tol):
            final_status = f"Desired definiteness achieved after {level} iterations."
            break

        negative_indices = np.where(eigenvalues < -tol)[0]
        if len(negative_indices) == 0:
            final_status = f"Fully reduced after {level} iterations."
            break

        # Get largest abs value eigenvalue
        idx = negative_indices[0]
        negative_eigenvalue = eigenvalues[idx]
        eigenvector = eigenvectors[:, idx]
        u = np.ones_like(eigenvector)

        # Calculate alpha
        denominator = np.dot(u, eigenvector)
        if np.abs(denominator) < tol:
            final_status = f"Numerical instability encountered at iteration {level}."
            break

        alpha = -negative_eigenvalue / denominator

        # Update matrix
        current_matrix += alpha * np.outer(u, eigenvector)

        level += 1

        # If adjusted for concave mode, revert matrix
        if adjusted_for_concave:
            matrices = [-M for M in matrices]
            current_matrix = -current_matrix
            eigenvalues_set = [-eigvals for eigvals in eigenvalues_set]
            final_status = final_status.replace("positive semidefinite", "negative semidefinite")

        # Results
        result = {
            "matrices": matrices,
            "eigenvalues_set": eigenvalues_set,
            "eigenvectors_set": eigenvectors_set,
            "levels_performed": level,
            "final_status": final_status
        }

        return result

# run numerical experiments to investigate conditions that imply capacity for iterations of rank reduction
# matrix characteristics: size, rank proportion, eigenvalue weighting, symmetry, invertibility, condition numbering (accuracy of a solution after approximation)
# generate symmetric matrix with uniform random distrib. [0,100] - each element with a value that is included conditional upon being less than a density parameter.
# above is populr for generating values for validating integer programming methods