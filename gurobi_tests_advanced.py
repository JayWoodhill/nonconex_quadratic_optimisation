import numpy as np
import logging
from gurobipy import GRB
from spectral_perturbation_analyser import (
    solve_qp_with_gurobi,
    iterative_rank_reduction
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

if __name__ == '__main__':
    # Example usage
    epsilon_values = [1e-6, 1e-4, 1e-2]
    for epsilon in epsilon_values:
        logger.info(f"\nTesting with epsilon = {epsilon}")
        result = test_solver_with_problematic_eigenvalue(n=10, epsilon=epsilon)
        print(result)
