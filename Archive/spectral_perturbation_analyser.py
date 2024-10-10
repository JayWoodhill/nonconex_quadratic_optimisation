import logging
import gurobipy as gp
import numpy as np
import numpy.linalg as linalg
import scipy
from gurobipy import GRB
import time
from scipy.stats import skew, kurtosis
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
'''
functions working notes 23/09
adding new characteristics to matrix checks
new dependencies: logging, time, scipy
'''

'''relationship between size (or other characteristics) and n(max steps)
    magnitude and strength of relationship between these variables added to this function
    
force input of b and pass this variable into the related functions
generate matrices of different sizes, record outputs, and calculate relationships between size, max rank reduction steps,...
fix erroneous cumulative delta values

use original code or new paper with standard method for generating matrices

-adjust functions: test solver, iterative rank reduction'''

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
    """
    Checks eigenvalue characteristics of the input matrix and returns a summary.

    Parameters:
    - matrix (numpy.ndarray): The matrix to check.
    - tol (float): Tolerance for considering an eigenvalue as zero.

    Returns:
    - dict: A dictionary containing eigenvalues, eigenvectors, counts, and various matrix properties.

    Dictionary of metric meanings
    "eigenvalues": "An array of the matrix's eigenvalues.",
    "eigenvectors": "A matrix whose columns are the eigenvectors corresponding to the eigenvalues.",
    "positive_eigenvalue_count": "The number of eigenvalues greater than the positive tolerance.",
    "negative_eigenvalue_count": "The number of eigenvalues less than the negative tolerance.",
    "zero_eigenvalue_count": "The number of eigenvalues considered zero within the specified tolerance.",
    "complex_eigenvalue_count": "The number of eigenvalues that are complex (should be zero for real symmetric matrices).",
    "is_positive_semidefinite": "A boolean indicating if the matrix is positive semidefinite (all eigenvalues ≥ -tolerance).",
    "is_invertible": "A boolean indicating if the matrix is invertible (determinant not close to zero).",
    "condition_number": "The condition number of the matrix (ratio of the largest to smallest singular value).",
    "reciprocal_condition_number": "The reciprocal of the condition number (1 / condition_number).",
    "density": "The proportion of non-zero elements in the matrix.",
    "frobenius_norm": "The Frobenius norm of the matrix (square root of the sum of the squares of all elements).",
    "spectral_norm": "The spectral (2) norm of the matrix (largest singular value).",
    "rank": "The numerical rank of the matrix (number of singular values greater than the tolerance).",
    "spectral_radius": "The maximum absolute value among the eigenvalues.",
    "trace": "The sum of the diagonal elements of the matrix (also the sum of the eigenvalues).",
    "determinant": "The determinant of the matrix (product of its eigenvalues).",
    "eigenvalue_mean": "The mean (average) of the eigenvalues.",
    "eigenvalue_variance": "The variance of the eigenvalues.",
    "eigenvalue_skewness": "The skewness of the eigenvalue distribution (measure of asymmetry).",
    "eigenvalue_kurtosis": "The kurtosis of the eigenvalue distribution (measure of 'tailedness').",
    "min_positive_eigenvalue": "The smallest positive eigenvalue (above the tolerance), if any.",
    "max_negative_eigenvalue": "The largest negative eigenvalue (below negative tolerance), if any.",
    "eigenvalue_gap": "The difference between the smallest positive and largest negative eigenvalues.",
    "largest_off_diagonal": "The maximum absolute value among the off-diagonal elements.",
    "energy": "The sum of the squares of all elements in the matrix (matrix energy)."

    """
    # Check characteristics of input matrix
    if not np.allclose(matrix, matrix.T, atol=tol):
        raise ValueError("Input matrix is not symmetric.")

    eigenvalues, eigenvectors = np.linalg.eig(matrix)

    # Eigenvalue counts
    eig_val_zero = np.sum(np.abs(eigenvalues) < tol)
    eig_val_pos = np.sum(eigenvalues > tol)
    eig_val_neg = np.sum(eigenvalues < -tol)
    eig_val_complex = np.sum(np.iscomplex(eigenvalues))

    # Additional matrix metrics
    positive_semidefinite = np.all(eigenvalues >= -tol)
    is_invertible = not np.isclose(np.linalg.det(matrix), 0, atol=tol)
    condition_number = np.linalg.cond(matrix)
    density = np.count_nonzero(matrix) / matrix.size
    frobenius_norm = np.linalg.norm(matrix, 'fro')
    spectral_norm = np.linalg.norm(matrix, 2)
    rank = np.linalg.matrix_rank(matrix, tol=tol)
    spectral_radius = np.max(np.abs(eigenvalues))
    trace = np.trace(matrix)
    determinant = np.linalg.det(matrix)
    eigenvalue_mean = np.mean(eigenvalues)
    eigenvalue_variance = np.var(eigenvalues)
    eigenvalue_skewness = skew(eigenvalues)
    eigenvalue_kurtosis = kurtosis(eigenvalues)
    sorted_eigenvalues = np.sort(eigenvalues)
    min_positive_eigenvalue = sorted_eigenvalues[eig_val_neg] if eig_val_neg < len(sorted_eigenvalues) else None
    max_negative_eigenvalue = sorted_eigenvalues[eig_val_neg - 1] if eig_val_neg > 0 else None
    eigenvalue_gap = min_positive_eigenvalue - max_negative_eigenvalue if (
                min_positive_eigenvalue is not None and max_negative_eigenvalue is not None) else None
    off_diagonal_elements = matrix - np.diag(np.diag(matrix))
    largest_off_diagonal = np.max(np.abs(off_diagonal_elements))
    rcond = 1 / condition_number if condition_number != 0 else None
    energy = np.sum(matrix ** 2)

    results = {
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "positive_eigenvalue_count": eig_val_pos,
        "negative_eigenvalue_count": eig_val_neg,
        "zero_eigenvalue_count": eig_val_zero,
        "complex_eigenvalue_count": eig_val_complex,
        "is_positive_semidefinite": positive_semidefinite,
        "is_invertible": is_invertible,
        "condition_number": condition_number,
        "reciprocal_condition_number": rcond,
        "density": density,
        "frobenius_norm": frobenius_norm,
        "spectral_norm": spectral_norm,
        "rank": rank,
        "spectral_radius": spectral_radius,
        "trace": trace,
        "determinant": determinant,
        "eigenvalue_mean": eigenvalue_mean,
        "eigenvalue_variance": eigenvalue_variance,
        "eigenvalue_skewness": eigenvalue_skewness,
        "eigenvalue_kurtosis": eigenvalue_kurtosis,
        "min_positive_eigenvalue": min_positive_eigenvalue,
        "max_negative_eigenvalue": max_negative_eigenvalue,
        "eigenvalue_gap": eigenvalue_gap,
        "largest_off_diagonal": largest_off_diagonal,
        "energy": energy
    }

    return results


def iterative_rank_reduction(q, c, A=None, b=None, mode='convex'):
    """
    Iteratively adjusts the input symmetric matrix `q` to make it positive semidefinite (`mode='convex'`)
    by eliminating undesired eigenvalues through symmetric rank-one updates,
    while adjusting the linear term `c` to compensate.

    Parameters:
    - q (numpy.ndarray): The input symmetric matrix to be adjusted.
    - c (numpy.ndarray): The linear coefficient vector.
    - A (numpy.ndarray, optional): The constraint matrix representing the problem's constraints.
    - b (numpy.ndarray or float, optional): The constraint value(s).
    - mode (str): 'convex' to make the matrix positive semidefinite.

    Returns:
    - dict: Contains the adjusted q and c, and additional information.
    """
    matrices = []
    eigenvalues_set = []
    eigenvectors_set = []
    delta_q_list = []  # List to store individual perturbations
    delta_c_list = []  # List to store adjustments to c
    level = 0
    alpha_list = []  # List of alpha values
    cumulative_delta_q = np.zeros_like(q, dtype=float)
    cumulative_delta_c = np.zeros_like(c, dtype=float)

    # q adjustment to simplify code
    q = q.copy()
    current_matrix = q.astype(float)
    current_c = c.copy()
    tol = 1e-10

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

        # Get the most negative eigenvalue and corresponding eigenvector
        idx = negative_indices[0]
        negative_eigenvalue = eigenvalues[idx]
        eigenvector = eigenvectors[:, idx]

        # Adjust u based on constraints
        if A is not None and b is not None:
            # Flatten A in case it's 2D with a single constraint
            A_flat = A.flatten()
            u = A_flat / np.linalg.norm(A_flat)
            b_value = b if np.isscalar(b) else b[0]
        else:
            u = np.ones_like(eigenvector)
            b_value = 1  # Default b value

        # Ensure that <u, eigenvector> != 0
        dot_uv = np.dot(u, eigenvector)
        if np.abs(dot_uv) < tol:
            final_status = f"Zero dot product between u and eigenvector at iteration {level}."
            break

        alpha = -negative_eigenvalue / dot_uv
        alpha_list.append(alpha)

        # Update Q symmetrically
        delta_q = alpha * (np.outer(u, eigenvector) + np.outer(eigenvector, u)) / 2
        cumulative_delta_q += delta_q
        delta_q_list.append(delta_q.copy())
        current_matrix += delta_q

        # Update c
        delta_c = -alpha * b_value * eigenvector
        cumulative_delta_c += delta_c
        delta_c_list.append(delta_c.copy())
        current_c += delta_c

        level += 1

    # Prepare the result dictionary
    result = {
        "adjusted_q": current_matrix,
        "adjusted_c": current_c,
        "matrices": matrices,
        "eigenvalues_set": eigenvalues_set,
        "eigenvectors_set": eigenvectors_set,
        "delta_q_list": delta_q_list,
        "delta_c_list": delta_c_list,
        "levels_performed": level,
        "final_status": final_status,
        "cumulative_delta_q": cumulative_delta_q,
        "cumulative_delta_c": cumulative_delta_c,
        "alpha_list": alpha_list
    }

    return result






def solve_qp_with_gurobi(Q, c, A=None, b=None, Aeq=None, beq=None, bounds=None, time_limit=60):
    """
    Solves a quadratic programming problem using Gurobi.

    The problem is defined as:
        minimize    (1/2) x^T Q x + c^T x
        subject to  A x <= b
                    Aeq x == beq
                    bounds on x

    Parameters:
    - Q (numpy.ndarray): The quadratic coefficient matrix.
    - c (numpy.ndarray): The linear coefficient vector.
    - A (numpy.ndarray, optional): The inequality constraint matrix.
    - b (numpy.ndarray, optional): The inequality constraint vector.
    - Aeq (numpy.ndarray, optional): The equality constraint matrix.
    - beq (numpy.ndarray, optional): The equality constraint vector.
    - bounds (list of tuples, optional): Bounds on variables [(lower, upper), ...].
    - time_limit (int, optional): Time limit for the solver in seconds.

    Returns:
    - result (dict): A dictionary containing solution status, objective value, variable values,
                     and computation time.

    **Gurobi Status Codes:**
    [List of status codes as before]
    """
    try:
        import time
        start_time = time.time()

        n = len(c)
        model = gp.Model()
        model.Params.OutputFlag = 0  # Suppress Gurobi output

        # Add variables with bounds
        if bounds is not None:
            lb = [bnd[0] if bnd[0] is not None else -GRB.INFINITY for bnd in bounds]
            ub = [bnd[1] if bnd[1] is not None else GRB.INFINITY for bnd in bounds]
        else:
            lb = [-GRB.INFINITY] * n
            ub = [GRB.INFINITY] * n

        if time_limit is not None:
            model.Params.TimeLimit = time_limit

        x = model.addMVar(shape=n, lb=lb, ub=ub, name="x")

        # Set objective
        obj = 0.5 * x @ Q @ x + c @ x
        model.setObjective(obj, GRB.MINIMIZE)

        # Add inequality constraints
        if A is not None and b is not None:
            model.addMConstr(A, x, sense=GRB.LESS_EQUAL, b=b, name="ineq_constraints")

        # Add equality constraints
        if Aeq is not None and beq is not None:
            model.addMConstr(Aeq, x, sense=GRB.EQUAL, b=beq, name="eq_constraints")

        # Optimize
        model.optimize()
        end_time = time.time()
        computation_time = end_time - start_time

        # Retrieve results
        status = model.Status
        if status == GRB.OPTIMAL:
            objective_value = model.ObjVal
            variable_values = x.X
            logger.info(f"Gurobi optimization successful. Objective value: {objective_value}")
        else:
            objective_value = None
            variable_values = None
            logger.warning(f"Gurobi optimization did not find an optimal solution. Status: {status}")

        result = {
            "status": status,
            "objective_value": objective_value,
            "variable_values": variable_values,
            "computation_time": computation_time
        }

        return result

    except gp.GurobiError as e:
        logger.exception("Gurobi Error during optimization.")
        return {
            "status": None,
            "objective_value": None,
            "variable_values": None,
            "computation_time": None,
            "error": str(e)
        }

