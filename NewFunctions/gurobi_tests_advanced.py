import numpy as np
import logging
import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import datetime
import time
import matplotlib.pyplot as plt
from spectral_perturbation_analyser import (
    generate_Q_with_small_negative_eigenvalue,
    iterative_rank_reduction_with_c
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def save_results_to_file(results, filename='results.txt'):
    """
    Save the results of the optimization to a text file with a timestamp.

    Parameters:
    - results (list of dict): A list containing the solver results.
    - filename (str): The prefix for the output text file. The timestamp will be appended.
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
    """
    Solves a quadratic programming problem using Gurobi optimizer.

    Parameters:
    - Q (numpy.ndarray): Quadratic coefficient matrix.
    - c (numpy.ndarray): Linear coefficient vector.
    - A (numpy.ndarray, optional): Inequality constraint matrix.
    - b (numpy.ndarray, optional): Inequality constraint vector.
    - Aeq (numpy.ndarray, optional): Equality constraint matrix.
    - beq (numpy.ndarray, optional): Equality constraint vector.
    - bounds (list of tuples, optional): Variable bounds as (lower, upper).
    - time_limit (float, optional): Time limit for the solver.

    Returns:
    - dict: Contains the optimization results.
    """
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

        s = model.addMVar(shape=n, lb=lb, ub=ub, name="s")  # Changed variable name to 's'

        # Set objective
        obj = 0.5 * s @ Q @ s + c @ s
        model.setObjective(obj, GRB.MINIMIZE)

        # Add inequality constraints
        if A is not None and b is not None:
            num_constraints = A.shape[0]
            for i in range(num_constraints):
                expr = A[i, :] @ s
                model.addConstr(expr <= b[i], name=f"ineq_constraint_{i}")

        # Add equality constraints
        if Aeq is not None and beq is not None:
            num_constraints = Aeq.shape[0]
            for i in range(num_constraints):
                expr = Aeq[i, :] @ s
                model.addConstr(expr == beq[i], name=f"eq_constraint_{i}")

        # Optimize
        model.optimize()
        end_time = time.time()
        computation_time = end_time - start_time

        # Retrieve results
        status = model.Status
        if status == GRB.OPTIMAL:
            objective_value = model.ObjVal
            variable_values = s.X
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

def run_multiple_tests(num_tests, n_range, epsilon_range):
    """
    Runs multiple tests to compare the optimization performance with and without rank reduction.

    Parameters:
    - num_tests (int): Number of tests for each (n, epsilon) combination.
    - n_range (tuple): Range of matrix sizes to test (start, end).
    - epsilon_range (list): List of epsilon values to test.

    Returns:
    - list of dict: A list containing the results of each test.
    """
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
    - results (list of dict): A list containing the solver results.
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
    """
    Calculate the average improvement filtered by rank.

    Parameters:
    - results (list of dict): A list containing the solver results.

    Returns:
    - pandas.DataFrame: DataFrame containing average improvements by rank.
    """
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

    num_tests = 2
    n_range = (4, 8)
    epsilon_range = [1e-1, 1, 2]

    # Run multiple tests and collect results
    results = run_multiple_tests(num_tests, n_range, epsilon_range)

    # Save results to a text file
    save_results_to_file(results, filename='optimization_results.txt')

    # Plot computation time
    plot_computation_time(results)

    # Calculate and display the average improvement by rank
    avg_improvement_by_rank = calculate_average_improvement_by_rank(results)
    print(avg_improvement_by_rank)
