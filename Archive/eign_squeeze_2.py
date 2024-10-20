import numpy as np
import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import time
import logging

# Constants
FLOAT_TOL = 1e-10
RELATIVE_EIGEN_TOL = 1e-4
RELAXED_FLOAT_TOL = 1e-1
MAX_ITERATION = 9999

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

def get_status_message(status_code):
    """Map Gurobi status codes to status messages."""
    status_messages = {
        GRB.LOADED: "LOADED",
        GRB.OPTIMAL: "OPTIMAL",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.CUTOFF: "CUTOFF",
        GRB.ITERATION_LIMIT: "ITERATION_LIMIT",
        GRB.NODE_LIMIT: "NODE_LIMIT",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.SOLUTION_LIMIT: "SOLUTION_LIMIT",
        GRB.INTERRUPTED: "INTERRUPTED",
        GRB.NUMERIC: "NUMERIC",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
        GRB.INPROGRESS: "INPROGRESS",
        GRB.USER_OBJ_LIMIT: "USER_OBJ_LIMIT",
    }
    return status_messages.get(status_code, f"STATUS_{status_code}")

def nonzero_eigenvalues(evals):
    return np.where(np.abs(evals) > FLOAT_TOL)[0]

def relative_nonzero_eigenvalues(evals):
    relative_evals = np.abs(evals) / np.sum(np.abs(evals))
    return np.where(relative_evals > RELATIVE_EIGEN_TOL)[0]

def iterative_eigen_squeeze(
    mat: np.ndarray, linear_coeffs: np.ndarray, linear_rhs: float
):
    """
    Adjusts the input matrix to make it negative semidefinite.

    Returns:
        squeezed_mat: The adjusted matrix.
        offset: The offset vector.
        iterations: Number of iterations (steps of rank reduction).
        eigenvalues_history: List of eigenvalues at each iteration.
    """
    # Avoid mutating the original matrix
    mat = np.copy(mat).astype(float)
    n = mat.shape[0]

    # Initialize the offset vector
    offset = np.zeros_like(linear_coeffs).astype(float)

    # List to store eigenvalues at each iteration
    eigenvalues_history = []

    # Save previous state
    prev_mat = mat
    prev_offset = offset

    iterations = 1
    while iterations <= MAX_ITERATION:
        # Compute eigenvalues and eigenvectors
        evals, evecs = np.linalg.eig(mat)

        # Record the sorted real parts of eigenvalues
        eigenvalues_history.append(np.sort(evals.real))

        # Check for complex eigenvalues
        if np.any(np.iscomplex(evals)):
            return prev_mat, prev_offset, iterations - 1, eigenvalues_history

        # Get significant eigenvalues based on relative tolerance
        nonzero_evals_index = relative_nonzero_eigenvalues(evals)
        evals = evals[nonzero_evals_index]
        evecs = evecs[:, nonzero_evals_index]

        # Check if the matrix is already negative semidefinite
        if np.all(evals <= FLOAT_TOL):
            return mat, offset, iterations, eigenvalues_history

        # Identify the largest positive eigenvalue
        squeezing_eval_index = np.argmax(evals)
        squeezing_eval = evals[squeezing_eval_index]
        squeezing_evec = evecs[:, squeezing_eval_index]

        # Compute alpha to adjust the matrix
        _denominator = squeezing_evec.dot(linear_coeffs)
        if abs(_denominator) < FLOAT_TOL:
            return mat, offset, iterations, eigenvalues_history
        alpha = -squeezing_eval / _denominator

        # Save previous state
        prev_mat = np.copy(mat)
        prev_offset = np.copy(offset)

        # Update the matrix and offset
        mat += alpha * np.outer(squeezing_evec, linear_coeffs)
        offset -= alpha * linear_rhs * squeezing_evec

        iterations += 1

    # Return if maximum iterations are reached
    return mat, offset, iterations, eigenvalues_history

def gurobi_solve(
    matrix: np.ndarray,
    linear_objective: np.ndarray,
    linear_coeff: np.ndarray,
    linear_rhs: float,
    variable_type=GRB.BINARY,
    sense=GRB.MAXIMIZE,
):
    n = len(linear_coeff)
    mdl = gp.Model()
    x = mdl.addVars(n, vtype=variable_type, name="x")
    mdl.addConstr(gp.quicksum(x[i] * linear_coeff[i] for i in range(n)) == linear_rhs, name="LinearConstraint")
    mdl.setObjective(
        gp.quicksum(matrix[i, j] * x[i] * x[j] for i in range(n) for j in range(n))
        + gp.quicksum(linear_objective[i] * x[i] for i in range(n)),
        sense=sense,
    )

    # Suppress Gurobi output
    mdl.setParam("OutputFlag", 0)

    # Start timing
    start_time = time.time()
    mdl.optimize()
    end_time = time.time()
    runtime = end_time - start_time

    # Get the status
    status = mdl.Status
    status_message = get_status_message(status)

    if status != GRB.OPTIMAL:
        logger.warning("Model did not solve to optimality.")
        return None, runtime, status_message, None

    # Get the objective value and solution
    obj_val = mdl.ObjVal
    x_values = np.array([x[i].X for i in range(n)])

    return obj_val, runtime, status_message, x_values

def analyze_matrix(matrix: np.ndarray, linear_coeffs: np.ndarray, linear_rhs: float):
    n = matrix.shape[0]
    logger.info(f"\nAnalyzing a matrix of size {n}x{n}")

    # Calculate eigenvalues of the original matrix
    original_evals = np.linalg.eigvals(matrix)
    num_positive_eigenvalues = np.sum(original_evals > FLOAT_TOL)
    eigenvalue_density = num_positive_eigenvalues / n
    logger.info(f"Eigenvalue density (positive eigenvalues / size): {eigenvalue_density:.4f}")

    # Apply iterative_eigen_squeeze
    squeezed_matrix, offset, iterations, eigenvalues_history = iterative_eigen_squeeze(
        matrix, linear_coeffs, linear_rhs
    )
    logger.info(f"Number of steps of rank reduction: {iterations}")

    # Solve using Gurobi for original matrix
    obj_orig, runtime_orig, status_orig, x_orig = gurobi_solve(
        matrix, np.zeros(n), linear_coeffs, linear_rhs
    )
    logger.info(f"Gurobi Original - Status: {status_orig}, Runtime: {runtime_orig:.4f}s, Objective: {obj_orig}")

    # Solve using Gurobi for squeezed matrix
    obj_squeezed, runtime_squeezed, status_squeezed, x_squeezed = gurobi_solve(
        squeezed_matrix, offset, linear_coeffs, linear_rhs
    )
    logger.info(f"Gurobi Squeezed - Status: {status_squeezed}, Runtime: {runtime_squeezed:.4f}s, Objective: {obj_squeezed}")

    # Check if objective values are close
    if obj_orig is not None and obj_squeezed is not None:
        obj_diff = abs(obj_orig - obj_squeezed)
        logger.info(f"Difference in objective values: {obj_diff:.4e}")
    else:
        obj_diff = None

    # Collect data for analysis
    result = {
        "Size": n,
        "Eigenvalue Density": eigenvalue_density,
        "Rank Reduction Steps": iterations,
        "Original Objective": obj_orig,
        "Squeezed Objective": obj_squeezed,
        "Objective Difference": obj_diff,
        "Original Runtime": runtime_orig,
        "Squeezed Runtime": runtime_squeezed,
        "Original Status": status_orig,
        "Squeezed Status": status_squeezed,
    }

    return result

def test_matrices():
    results = []

    # Define different sizes
    sizes = [5, 10, 20, 50]  # Adjust sizes as needed

    for n in sizes:
        logger.info(f"\nTesting with matrix size: {n}x{n}")

        # Generate a random symmetric matrix
        random_matrix = np.random.randint(-10, 10, size=(n, n))
        random_matrix = (random_matrix + random_matrix.T) / 2  # Make it symmetric

        # Define linear coefficients and RHS
        linear_coeffs = np.ones(n)
        linear_rhs = np.random.randint(1, n)  # Ensure that linear_rhs is feasible

        # Analyze the matrix
        result = analyze_matrix(random_matrix, linear_coeffs, linear_rhs)
        results.append(result)

    # Convert results to a DataFrame
    df = pd.DataFrame(results)
    print("\nSummary of Results:")
    print(df)

if __name__ == "__main__":
    test_matrices()
