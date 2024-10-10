import numpy as np
import logging
from gurobipy import GRB
import gurobipy as gp
import pandas as pd
import datetime
import time
import matplotlib.pyplot as plt
# Importing the original iterative_rank_reduction function
# from spectral_perturbation_analyser import (
#     solve_qp_with_gurobi,
#     iterative_rank_reduction
# )

'''advanced testing differentiates this from function unit testing. this file defines a set of functions to interact with gurobi solver functions using
q and q prime to consider the effects of applying the novel rank reduction technique.
to do:
- expand the results table to stratify by further characteristics than just rank
- improve solver output format
- extend advanced tests to call q/q-prime itself and log computation/time usage to make fairer performance comparisons
- explore gurobipy functions further

first attempt is integer incrementing n[10,20] with 5 tests, 3 eps vals
attempting with larger |eps| val because it seems more practical
noting that there are regular objective value difs of +-0.6'''

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def save_results_to_file(results, filename='results.txt'):
    """
    Save the results of the optimization to a text file with a timestamp.

    Parameters:
    - results (dict): A dictionary containing the solver results.
    - filename_prefix (str): The prefix for the output text file. The timestamp will be appended.
    """
    # Get the current system time and format it as a string (e.g., YYYYMMDD_HHMMSS)
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create the full filename with the timestamp
    filename = f"{filename}_{current_time}.txt"

    # Write the results to the file
    with open(filename, 'w') as f:
        f.write("Optimization Results:\n")
        for result in results:
            f.write(f"n = {result.get('n', 'N/A')}, epsilon = {result.get('epsilon', 'N/A')}\n")
            f.write(f"Status without rank reduction: {result.get('status_without', 'N/A')}\n")
            f.write(f"Computation time without rank reduction: {result.get('time_without', 'N/A')} seconds\n")
            f.write(f"Status with rank reduction: {result.get('status_with', 'N/A')}\n")
            f.write(f"Computation time with rank reduction: {result.get('time_with', 'N/A')} seconds\n")
            f.write(f"Improvement percentage: {result.get('improvement_percentage', 'N/A'):.2f}%\n")
            f.write(f"Objective without rank reduction: {result.get('objective_without', 'N/A')}\n")
            f.write(f"Objective with rank reduction: {result.get('objective_with', 'N/A')}\n")
            f.write("-" * 50 + "\n")

    print(f"Results saved to {filename}.")

def solve_qp_with_gurobi(Q, c, A=None, b=None, Aeq=None, beq=None, bounds=None, time_limit=60):
    try:
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
            num_constraints = A.shape[0]
            for i in range(num_constraints):
                expr = A[i, :] @ x
                model.addConstr(expr <= b[i], name=f"ineq_constraint_{i}")

        # Add equality constraints
        if Aeq is not None and beq is not None:
            num_constraints = Aeq.shape[0]
            for i in range(num_constraints):
                expr = Aeq[i, :] @ x
                model.addConstr(expr == beq[i], name=f"eq_constraint_{i}")

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
        logger.exception(f"Gurobi Error during optimization: {e}")
        return {
            "status": None,
            "objective_value": None,
            "variable_values": None,
            "computation_time": time.time() - start_time,
            "error": str(e)
        }

def generate_Q_with_small_negative_eigenvalue(n, epsilon):
    """Generates an n x n symmetric matrix with one small negative eigenvalue."""
    # Generate random positive definite matrix
    A = np.random.randn(n, n)
    Q_posdef = A.T @ A  # Ensures Q is positive semidefinite

    # Compute eigenvalues and eigenvectors
    eigvals, eigvecs = np.linalg.eigh(Q_posdef)

    # Introduce a small negative eigenvalue
    eigvals[0] = -epsilon  # Set the smallest eigenvalue to -epsilon

    # Reconstruct Q with the modified eigenvalues
    Q = eigvecs @ np.diag(eigvals) @ eigvecs.T

    # Ensure symmetry
    Q = (Q + Q.T) / 2

    return Q

def iterative_rank_reduction(q, A=None, mode='convex'):
    """
    Original iterative_rank_reduction function.
    """
    matrices = []
    eigenvalues_set = []
    eigenvectors_set = []
    delta_q_list = []  # List to store individual perturbations
    level = 0
    alpha_list = []  # List to store alpha values
    cumulative_delta_q = np.zeros_like(q, dtype=float)  # Initialize cumulative perturbation matrix

    # q adjustment to simplify code
    if mode == 'concave':
        q = -q.copy()
        adjusted_for_concave = True
    else:
        q = q.copy()
        adjusted_for_concave = False

    current_matrix = q.astype(float)
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
        if A is not None:
            # Flatten A in case it's 2D with a single constraint
            A_flat = A.flatten()
            u = A_flat / np.linalg.norm(A_flat)
        else:
            u = np.ones_like(eigenvector)

        # Calculate alpha
        denominator = np.dot(u, eigenvector)
        if np.abs(denominator) < tol:
            final_status = f"Numerical instability encountered at iteration {level}."
            break

        alpha = -negative_eigenvalue / denominator
        alpha_list.append(alpha)  # Store alpha value

        # Compute the perturbation matrix (delta_q)
        delta_q = alpha * np.outer(u, eigenvector)
        delta_q_list.append(delta_q.copy())  # Store individual perturbation
        cumulative_delta_q += delta_q  # Accumulate the perturbation

        # Update matrix
        current_matrix += delta_q

        level += 1

    # Adjust results for concave mode
    if adjusted_for_concave:
        matrices = [-M for M in matrices]
        current_matrix = -current_matrix
        cumulative_delta_q = -cumulative_delta_q
        delta_q_list = [-dq for dq in delta_q_list]
        eigenvalues_set = [-eigvals for eigvals in eigenvalues_set]
        final_status = final_status.replace("positive semidefinite", "negative semidefinite")

    # Prepare the result dictionary
    result = {
        "matrices": matrices,
        "eigenvalues_set": eigenvalues_set,
        "eigenvectors_set": eigenvectors_set,
        "delta_q_list": delta_q_list,
        "levels_performed": level,
        "final_status": final_status,
        "cumulative_delta_q": cumulative_delta_q,  # Cumulative perturbation matrix
        "alpha_list": alpha_list  # List of alpha values used
    }

    return result

def iterative_rank_reduction_with_c(q, c, A=None, b=None, mode='convex'):
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

def run_multiple_tests(num_tests, n_range, epsilon_range):
    '''New test function multi test'''
    results = []

    for n in range(n_range[0], n_range[1] + 1):
        for _ in range(num_tests):
            for epsilon in epsilon_range:
                # Generate problematic Q matrix with a small negative eigenvalue
                Q_problematic = generate_Q_with_small_negative_eigenvalue(n, epsilon)
                c = np.random.randn(n)
                A = np.random.randn(1, n)
                b = np.array([np.random.rand() * n])
                bounds = [(-1.0, 1.0) for _ in range(n)]

                # Solve without rank reduction
                result_without_reduction = solve_qp_with_gurobi(Q_problematic, c, A=A, b=b, bounds=bounds)
                status_without = result_without_reduction['status']
                time_without = result_without_reduction['computation_time'] if status_without == GRB.OPTIMAL else None
                objective_without = result_without_reduction['objective_value']

                # Apply rank reduction with c adjustment
                reduction_result = iterative_rank_reduction_with_c(Q_problematic, c, A=A, b=b, mode='convex')
                Q_psd = reduction_result['adjusted_q']
                c_adjusted = reduction_result['adjusted_c']

                # Solve with rank-reduced Q and adjusted c
                result_with_reduction = solve_qp_with_gurobi(Q_psd, c_adjusted, A=A, b=b, bounds=bounds)
                status_with = result_with_reduction['status']
                time_with = result_with_reduction['computation_time'] if status_with == GRB.OPTIMAL else None
                objective_with = result_with_reduction['objective_value']

                # Get rank before and after reduction
                initial_rank = np.linalg.matrix_rank(Q_problematic)
                reduced_rank = np.linalg.matrix_rank(Q_psd)

                # Store the results only if both solvers were successful
                if time_without is not None and time_with is not None:
                    results.append({
                        'n': n,
                        'epsilon': epsilon,
                        'initial_rank': initial_rank,
                        'reduced_rank': reduced_rank,
                        'time_without': time_without,
                        'time_with': time_with,
                        'improvement_percentage': ((time_without - time_with) / time_without) * 100,
                        'objective_without': objective_without,
                        'objective_with': objective_with,
                        'status_without': status_without,
                        'status_with': status_with
                    })

    return results

def plot_computation_time(results):
    """
    Plot computation time with and without rank reduction.

    Parameters:
    - results (dict): A dictionary containing the solver results.
    """
    df = pd.DataFrame(results)
    n_values = df['n'].unique()
    epsilon_values = df['epsilon'].unique()

    plt.figure(figsize=(10, 6))

    for epsilon in epsilon_values:
        subset = df[df['epsilon'] == epsilon]
        plt.plot(subset['n'], subset['time_without'], marker='o', label=f'Without Rank Reduction (epsilon={epsilon})')
        plt.plot(subset['n'], subset['time_with'], marker='x', label=f'With Rank Reduction (epsilon={epsilon})')

    plt.xlabel('Matrix Size (n)')
    plt.ylabel('Computation Time (seconds)')
    plt.title('Computation Time with and without Rank Reduction')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('computation_time_plot.png')  # Save the plot as an image
    plt.show()

def calculate_average_improvement_by_rank(results):
    """Calculate the average improvement filtered by rank.
    Relationship between size (or other characteristics) and n(max steps)
    Magnitude and strength of relationship between these variables added to this function"""
    df = pd.DataFrame(results)

    # Group by initial rank and calculate the mean improvement percentage
    avg_improvement_by_rank = df.groupby('initial_rank').agg(
        avg_improvement=('improvement_percentage', 'mean'),
        avg_time_without=('time_without', 'mean'),
        avg_time_with=('time_with', 'mean'),
        count=('n', 'count')
    ).reset_index()

    return avg_improvement_by_rank

if __name__ == '__main__':
    num_tests = 2  # Number of tests for each (n, epsilon) combination
    n_range = (3, 6)  # Range of matrix sizes n to test
    epsilon_range = [1e-1, 1]  # Range of epsilon values to test

    # Run multiple tests and collect results
    results = run_multiple_tests(num_tests, n_range, epsilon_range)

    # Save results to a text file
    save_results_to_file(results, filename='optimization_results.txt')

    # Plot computation time
    plot_computation_time(results)

    # Calculate and display the average improvement by rank
    avg_improvement_by_rank = calculate_average_improvement_by_rank(results)
    print(avg_improvement_by_rank)
