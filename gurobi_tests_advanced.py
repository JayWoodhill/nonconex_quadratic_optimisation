import numpy as np
import logging
from gurobipy import GRB
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from spectral_perturbation_analyser import (
    solve_qp_with_gurobi,
    iterative_rank_reduction
)

'''advanced testing differentiates this from function unit testing. this file defines a set of functions to interact with with gurobi solver functions using
q and q prime to consider the effects of applying the novel rank reeduction technique.
to do:
- expand the results table to stratify by further characteristics than just rank
- improve solver output format
- extend advanced tests to call q/q-prime itself and log computation/time usage to make fairer performance comparisons
- explore gurobipy functions further

first attempt is integer incrementing n[10,20] with 5 tests, 3 eps vals
attempting with lrger |eps| val because it seems more practical
noting that there are regular objective value difs of +-0.6'''

#adding results, compute time plot

'''adding descriptive results'''



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

def test_solver_with_problematic_eigenvalue(n, epsilon):
    # Generate Q with a small negative eigenvalue
    Q_problematic = generate_Q_with_small_negative_eigenvalue(n, epsilon)
    c = np.random.randn(n)
    A = np.random.randn(1, n)
    b = np.array([np.random.rand() * n])
    bounds = [(-1.0, 1.0) for _ in range(n)]

    # Solve without rank reduction
    logger.info("Solving without rank reduction...")
    result_without_reduction = solve_qp_with_gurobi(Q_problematic, c, A=A, b=b, bounds=bounds)

    # Check if solver was successful
    status_without = result_without_reduction['status']
    if status_without == GRB.OPTIMAL:
        logger.info("Solver found an optimal solution without rank reduction.")
    else:
        logger.warning(f"Solver did not find an optimal solution without rank reduction. Status: {status_without}")

    # Apply rank reduction
    reduction_result = iterative_rank_reduction(Q_problematic, mode='convex')
    Q_psd = reduction_result['matrices'][-1]

    # Solve with rank-reduced Q
    logger.info("Solving with rank reduction...")
    result_with_reduction = solve_qp_with_gurobi(Q_psd, c, A=A, b=b, bounds=bounds)

    # Check if solver was successful
    status_with = result_with_reduction['status']
    if status_with == GRB.OPTIMAL:
        logger.info("Solver found an optimal solution with rank reduction.")
    else:
        logger.warning(f"Solver did not find an optimal solution with rank reduction. Status: {status_with}")

    # Compare computation times
    time_without = result_without_reduction['computation_time']
    time_with = result_with_reduction['computation_time']

    logger.info(f"Computation time without rank reduction: {time_without:.6f} seconds")
    logger.info(f"Computation time with rank reduction: {time_with:.6f} seconds")

    # Return results for further analysis
    return {
        'n': n,
        'epsilon': epsilon,
        'status_without': status_without,
        'status_with': status_with,
        'time_without': time_without,
        'time_with': time_with,
        'objective_without': result_without_reduction['objective_value'],
        'objective_with': result_with_reduction['objective_value']
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

    return Q

def run_multiple_tests(num_tests, n_range, epsilon_range):
    '''new test function multi test'''
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

                # Apply rank reduction
                reduction_result = iterative_rank_reduction(Q_problematic, mode='convex')
                Q_psd = reduction_result['matrices'][-1]

                # Solve with rank-reduced Q
                result_with_reduction = solve_qp_with_gurobi(Q_psd, c, A=A, b=b, bounds=bounds)
                status_with = result_with_reduction['status']
                time_with = result_with_reduction['computation_time'] if status_with == GRB.OPTIMAL else None

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
                        'improvement_percentage': ((time_without - time_with) / time_without) * 100
                    })

    return results


def plot_computation_time(results):
    """
    Plot computation time with and without rank reduction.

    Parameters:
    - results (dict): A dictionary containing the solver results.
    """
    n_values = [result['n'] for result in results]
    time_without = [result['time_without'] for result in results]
    time_with = [result['time_with'] for result in results]

    plt.figure(figsize=(10, 6))

    # Plotting computation time
    plt.bar(n_values, time_without, width=0.4, label='Without Rank Reduction', align='center')
    plt.bar(n_values, time_with, width=0.4, label='With Rank Reduction', align='edge')

    plt.xlabel('Matrix Size (n)')
    plt.ylabel('Computation Time (seconds)')
    plt.title('Computation Time with and without Rank Reduction')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('computation_time_plot.png')  # Save the plot as an image
    plt.show()


def calculate_average_improvement_by_rank(results):
    """Calculate the average improvement filtered by rank."""
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
    num_tests = 3  # Number of tests for each (n, epsilon) combination
    n_range = (3,8)  # Range of matrix sizes n to test
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

'''if __name__ == '__main__':
    # param
    num_tests = 5  # Number of tests for each (n, epsilon) combination
    n_range = (5, 10)
    epsilon_range = [1e-2, 1e-1, 1]

    results = run_multiple_tests(num_tests, n_range, epsilon_range)
    avg_improvement_by_rank = calculate_average_improvement_by_rank(results)
    print(avg_improvement_by_rank)'''

'''if __name__ == '__main__':
    # Example usage
    epsilon_values = [1e-6, 1e-4, 1e-2]
    for epsilon in epsilon_values:
        logger.info(f"\nTesting with epsilon = {epsilon}")
        result = test_solver_with_problematic_eigenvalue(n=10, epsilon=epsilon)
        print(result)'''
