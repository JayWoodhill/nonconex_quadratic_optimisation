import numpy as np
import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import time
import logging
import matplotlib.pyplot as plt
import seaborn as sns

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

def generate_matrix_with_negative_eigenvalues(n, num_negative_eigenvalues):
    """
    Generate a symmetric matrix of size n x n with a specified number of negative eigenvalues.
    """
    eigenvalues = np.random.uniform(0.1, 1.0, size=n)
    # Set some eigenvalues to negative
    negative_indices = np.random.choice(n, num_negative_eigenvalues, replace=False)
    eigenvalues[negative_indices] *= -1

    # Generate a random orthogonal matrix
    Q = np.linalg.qr(np.random.randn(n, n))[0]
    # Construct the symmetric matrix
    matrix = Q @ np.diag(eigenvalues) @ Q.T
    # Ensure symmetry
    matrix = (matrix + matrix.T) / 2
    return matrix

def gurobi_solve(
    matrix: np.ndarray,
    linear_objective: np.ndarray,
    linear_coeff: np.ndarray,
    linear_rhs: float,
    variable_type=GRB.BINARY,
    sense=GRB.MAXIMIZE,
    method=None,
    presolve=None,
    time_limit=None,
    threads=None
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

    # Set Gurobi parameters
    if method is not None:
        mdl.setParam('Method', method)
    if presolve is not None:
        mdl.setParam('Presolve', presolve)
    if time_limit is not None:
        mdl.setParam('TimeLimit', time_limit)
    if threads is not None:
        mdl.setParam('Threads', threads)

    # Start timing
    start_time = time.time()
    mdl.optimize()
    end_time = time.time()
    runtime = end_time - start_time

    # Get the status
    status = mdl.Status
    status_message = get_status_message(status)

    # Get number of nodes explored
    nodes_explored = mdl.NodeCount

    if status != GRB.OPTIMAL:
        logger.warning(f"Model did not solve to optimality. Status: {status_message}")
        return None, runtime, status_message, None, nodes_explored

    # Get the objective value and solution
    obj_val = mdl.ObjVal
    x_values = np.array([x[i].X for i in range(n)])

    return obj_val, runtime, status_message, x_values, nodes_explored

def analyze_matrix(matrix: np.ndarray, linear_coeffs: np.ndarray, linear_rhs: float, **kwargs):
    n = matrix.shape[0]
    logger.info(f"\nAnalyzing a matrix of size {n}x{n}")

    # Calculate eigenvalues of the original matrix
    original_evals = np.linalg.eigvals(matrix)
    num_positive_eigenvalues = np.sum(original_evals > FLOAT_TOL)
    num_negative_eigenvalues = np.sum(original_evals < -FLOAT_TOL)
    eigenvalue_density = num_positive_eigenvalues / n
    logger.info(f"Eigenvalue density (positive eigenvalues / size): {eigenvalue_density:.4f}")
    logger.info(f"Number of negative eigenvalues: {num_negative_eigenvalues}")

    # Apply iterative_eigen_squeeze
    squeezed_matrix, offset, iterations, eigenvalues_history = iterative_eigen_squeeze(
        matrix, linear_coeffs, linear_rhs
    )
    logger.info(f"Number of steps of rank reduction: {iterations}")

    # Solve using Gurobi for original matrix
    obj_orig, runtime_orig, status_orig, x_orig, nodes_orig = gurobi_solve(
        matrix, np.zeros(n), linear_coeffs, linear_rhs, **kwargs
    )
    logger.info(f"Gurobi Original - Status: {status_orig}, Runtime: {runtime_orig:.4f}s, Objective: {obj_orig}, Nodes: {nodes_orig}")

    # Solve using Gurobi for squeezed matrix
    obj_squeezed, runtime_squeezed, status_squeezed, x_squeezed, nodes_squeezed = gurobi_solve(
        squeezed_matrix, offset, linear_coeffs, linear_rhs, **kwargs
    )
    logger.info(f"Gurobi Squeezed - Status: {status_squeezed}, Runtime: {runtime_squeezed:.4f}s, Objective: {obj_squeezed}, Nodes: {nodes_squeezed}")

    # Check if objective values are close
    if obj_orig is not None and obj_squeezed is not None:
        obj_diff = abs(obj_orig - obj_squeezed)
        logger.info(f"Difference in objective values: {obj_diff:.4e}")
    else:
        obj_diff = None

    # Collect data for analysis
    result = {
        "Size": n,
        "Num Positive Eigenvalues": num_positive_eigenvalues,
        "Num Negative Eigenvalues": num_negative_eigenvalues,
        "Eigenvalue Density": eigenvalue_density,
        "Rank Reduction Steps": iterations,
        "Original Objective": obj_orig,
        "Squeezed Objective": obj_squeezed,
        "Objective Difference": obj_diff,
        "Original Runtime (s)": runtime_orig,
        "Squeezed Runtime (s)": runtime_squeezed,
        "Original Nodes": nodes_orig,
        "Squeezed Nodes": nodes_squeezed,
        "Original Status": status_orig,
        "Squeezed Status": status_squeezed,
    }

    return result

def test_matrices():
    results = []

    # Define different sizes, numbers of negative eigenvalues, methods, and presolve options
    sizes = [10, 20]
    num_negative_eigenvalues_list = [0, 5, 10]
    methods = [0, 1, 2]  # Different Gurobi methods
    presolve_options = [0, 1, 2]

    for n in sizes:
        for num_negative_eigenvalues in num_negative_eigenvalues_list:
            for method in methods:
                for presolve in presolve_options:
                    logger.info(f"\nTesting with matrix size: {n}x{n}, negative eigenvalues: {num_negative_eigenvalues}, method: {method}, presolve: {presolve}")

                    matrix = generate_matrix_with_negative_eigenvalues(n, num_negative_eigenvalues)

                    # Define linear coefficients and RHS
                    linear_coeffs = np.ones(n)
                    linear_rhs = np.random.randint(1, n)

                    result = analyze_matrix(matrix, linear_coeffs, linear_rhs, method=method, presolve=presolve)
                    # Add additional data to the result
                    result['Num Negative Eigenvalues'] = num_negative_eigenvalues
                    result['Method'] = method
                    result['Presolve'] = presolve
                    results.append(result)

    df = pd.DataFrame(results)
    print("\nSummary of Results:")
    print(df)

    # Save to CSV
    df.to_csv("rank_reduction_analysis_results.csv", index=False)

    # Generate plots
    generate_plots(df)

def generate_plots(df: pd.DataFrame):
    """
    Generates plots to analyze the relationships between input and output metrics.

    Parameters:
        df (pd.DataFrame): DataFrame containing the analysis results.
    """
    sns.set(style="whitegrid")

    # Plot Original Runtime vs Number of Negative Eigenvalues
    plt.figure(figsize=(8, 6))
    sns.lineplot(x="Num Negative Eigenvalues", y="Original Runtime (s)", hue="Size", style="Method", data=df, markers=True)
    plt.title("Original Solver Runtime vs Number of Negative Eigenvalues")
    plt.ylabel("Runtime (seconds)")
    plt.xlabel("Number of Negative Eigenvalues")
    plt.legend(title="Size and Method")
    plt.tight_layout()
    plt.savefig("original_runtime_vs_negative_eigenvalues.png")
    plt.show()

    # Plot Squeezed Runtime vs Number of Negative Eigenvalues
    plt.figure(figsize=(8, 6))
    sns.lineplot(x="Num Negative Eigenvalues", y="Squeezed Runtime (s)", hue="Size", style="Method", data=df, markers=True)
    plt.title("Squeezed Solver Runtime vs Number of Negative Eigenvalues")
    plt.ylabel("Runtime (seconds)")
    plt.xlabel("Number of Negative Eigenvalues")
    plt.legend(title="Size and Method")
    plt.tight_layout()
    plt.savefig("squeezed_runtime_vs_negative_eigenvalues.png")
    plt.show()

    # Plot Rank Reduction Steps vs Number of Negative Eigenvalues
    plt.figure(figsize=(8, 6))
    sns.lineplot(x="Num Negative Eigenvalues", y="Rank Reduction Steps", hue="Size", data=df, marker="o")
    plt.title("Rank Reduction Steps vs Number of Negative Eigenvalues")
    plt.ylabel("Rank Reduction Steps")
    plt.xlabel("Number of Negative Eigenvalues")
    plt.tight_layout()
    plt.savefig("rank_reduction_steps_vs_negative_eigenvalues.png")
    plt.show()

    # Plot Objective Difference vs Rank Reduction Steps
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x="Rank Reduction Steps", y="Objective Difference", hue="Size", data=df)
    plt.title("Objective Difference vs Rank Reduction Steps")
    plt.ylabel("Objective Difference")
    plt.xlabel("Rank Reduction Steps")
    plt.tight_layout()
    plt.savefig("objective_difference_vs_rank_reduction_steps.png")
    plt.show()

    # Plot Nodes Explored vs Number of Negative Eigenvalues
    plt.figure(figsize=(8, 6))
    sns.lineplot(x="Num Negative Eigenvalues", y="Original Nodes", hue="Size", style="Method", data=df, markers=True)
    plt.title("Original Nodes Explored vs Number of Negative Eigenvalues")
    plt.ylabel("Nodes Explored")
    plt.xlabel("Number of Negative Eigenvalues")
    plt.legend(title="Size and Method")
    plt.tight_layout()
    plt.savefig("original_nodes_vs_negative_eigenvalues.png")
    plt.show()

    sns.lineplot(x="Num Negative Eigenvalues", y="Squeezed Nodes", hue="Size", style="Method", data=df, markers=True)
    plt.title("Squeezed Nodes Explored vs Number of Negative Eigenvalues")
    plt.ylabel("Nodes Explored")
    plt.xlabel("Number of Negative Eigenvalues")
    plt.legend(title="Size and Method")
    plt.tight_layout()
    plt.savefig("squeezed_nodes_vs_negative_eigenvalues.png")
    plt.show()

if __name__ == "__main__":
    test_matrices()
