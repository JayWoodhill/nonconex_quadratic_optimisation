import numpy as np
import time
import logging
from gurobipy import GRB
from spectral_perturbation_analyser import (
    solve_qp_with_gurobi,
    iterative_rank_reduction
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_solver_performance(n):
    # Generate an indefinite Q
    Q_indefinite = generate_indefinite_Q(n)
    c = np.random.randn(n)
    A = np.random.randn(1, n)
    b = np.array([np.random.rand() * n])
    bounds = [(-1.0, 1.0) for _ in range(n)]

    # Solve without rank reduction
    logger.info("Solving without rank reduction...")
    result_without_reduction = solve_qp_with_gurobi(Q_indefinite, c, A=A, b=b, bounds=bounds)

    # Check if solver was successful
    status_without = result_without_reduction['status']
    if status_without == GRB.OPTIMAL:
        logger.info("Solver found an optimal solution without rank reduction.")
    else:
        logger.warning(f"Solver did not find an optimal solution without rank reduction. Status: {status_without}")

    # Apply rank reduction
    reduction_result = iterative_rank_reduction(Q_indefinite, mode='convex')
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
        'status_without': status_without,
        'status_with': status_with,
        'time_without': time_without,
        'time_with': time_with,
        'objective_without': result_without_reduction['objective_value'],
        'objective_with': result_with_reduction['objective_value']
    }

def generate_indefinite_Q(n):
    """Generates an n x n indefinite symmetric matrix."""
    # Generate random symmetric matrix
    A = np.random.randn(n, n)
    Q = (A + A.T) / 2
    # Ensure Q is indefinite by modifying eigenvalues
    eigvals, eigvecs = np.linalg.eigh(Q)
    # Set half of the eigenvalues to negative
    eigvals[int(n/2):] = -np.abs(eigvals[int(n/2):]) - 1e-2  # Slightly negative
    Q_indefinite = eigvecs @ np.diag(eigvals) @ eigvecs.T
    return Q_indefinite

if __name__ == '__main__':
    # Example for n = 10
    result = test_solver_performance(10)
    print(result)


'''this is going terribly'''