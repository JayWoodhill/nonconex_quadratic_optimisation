import numpy as np
import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import time
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from eigen_squeeze import (
    nonzero_eigenvalues,
    relative_nonzero_eigenvalues,
    iterative_eigen_squeeze,
    analyse_matrix
)

FLOAT_TOL = 1e-10
RELATIVE_EIGEN_TOL = 1e-4
RELAXED_FLOAT_TOL = 1e-1
MAX_ITERATION = 9999

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

def get_status_message(status_code):
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

def gurobi_solve(
    matrix: np.ndarray,
    linear_objective: np.ndarray,
    linear_coeff: np.ndarray,
    linear_rhs: float,
    variable_type=GRB.BINARY,
    sense=GRB.MAXIMIZE,
    time_limit=60  # Enforce 60-second timeout
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

    # Set time limit
    mdl.setParam("TimeLimit", time_limit)

    # Start timing
    start_time = time.time()
    mdl.optimize()
    end_time = time.time()
    runtime = end_time - start_time

    # Get the status
    status = mdl.Status
    status_message = get_status_message(status)

    if status != GRB.OPTIMAL:
        logger.warning(f"Model did not solve to optimality. Status: {status_message}")
        return None, runtime, status_message, None

    # Get the objective value and solution
    obj_val = mdl.ObjVal
    x_values = np.array([x[i].X for i in range(n)])

    return obj_val, runtime, status_message, x_values

def analyse_matrix(matrix: np.ndarray, linear_coeffs: np.ndarray, linear_rhs: float):
    n = matrix.shape[0]
    logger.info(f"\nAnalysing a matrix of size {n}x{n}")

    # Solve using Gurobi for the original matrix
    obj_orig, runtime_orig, status_orig, x_orig = gurobi_solve(
        matrix, np.zeros(n), linear_coeffs, linear_rhs
    )
    logger.info(f"Gurobi Original - Status: {status_orig}, Runtime: {runtime_orig:.4f}s, Objective: {obj_orig}")

    # Apply the iterative eigenvalue squeezing to process the matrix
    squeezed_matrix, offset, iterations, eigenvalues_history = iterative_eigen_squeeze(
        matrix, linear_coeffs, linear_rhs
    )
    logger.info(f"Number of steps of rank reduction: {iterations}")

    # Solve using Gurobi for the squeezed matrix
    obj_squeezed, runtime_squeezed, status_squeezed, x_squeezed = gurobi_solve(
        squeezed_matrix, offset, linear_coeffs, linear_rhs
    )
    logger.info(f"Gurobi Squeezed - Status: {status_squeezed}, Runtime: {runtime_squeezed:.4f}s, Objective: {obj_squeezed}")

    # Compare results: objectives, runtimes, and solver status
    if obj_orig is not None and obj_squeezed is not None:
        obj_diff = abs(obj_orig - obj_squeezed)
        logger.info(f"Difference in objective values: {obj_diff:.4e}")
    else:
        obj_diff = None

    # Collect data for analysis
    result = {
        "Size": n,
        "Eigenvalue Density": np.sum(np.linalg.eigvals(matrix) > FLOAT_TOL) / n,
        "Rank Reduction Steps": iterations,
        "Original Objective": obj_orig,
        "Squeezed Objective": obj_squeezed,
        "Objective Difference": obj_diff,
        "Original Runtime (s)": runtime_orig,
        "Squeezed Runtime (s)": runtime_squeezed,
        "Original Status": status_orig,
        "Squeezed Status": status_squeezed,
    }

    return result

def test_matrices():
    results = []

    # Test sizes from 5 to 50, incrementing by 1, repeated 3 times
    sizes_small = list(range(5, 51))  # 5 to 50 inclusive
    repeats_small = 3

    # Test sizes from 10 to 500, incrementing by 10, repeated once
    sizes_large = list(range(10, 501, 10))  # 10 to 500 inclusive
    repeats_large = 1

    # Testing matrices from sizes_small with repeats
    for n in sizes_small:
        for repeat in range(repeats_small):
            logger.info(f"\nTesting with matrix size: {n}x{n}, repeat {repeat + 1}/{repeats_small}")
            random_matrix = np.random.randint(-10, 10, size=(n, n))
            random_matrix = (random_matrix + random_matrix.T) / 2
            linear_coeffs = np.ones(n)
            linear_rhs = np.random.randint(1, n)

            result = analyse_matrix(random_matrix, linear_coeffs, linear_rhs)
            result['Repeat'] = repeat + 1  # Add repeat number to the result
            results.append(result)

    # Testing matrices from sizes_large with single repeat
    for n in sizes_large:
        logger.info(f"\nTesting with matrix size: {n}x{n}")
        random_matrix = np.random.randint(-10, 10, size=(n, n))
        random_matrix = (random_matrix + random_matrix.T) / 2
        linear_coeffs = np.ones(n)
        linear_rhs = np.random.randint(1, n)

        result = analyse_matrix(random_matrix, linear_coeffs, linear_rhs)
        result['Repeat'] = 1  # Only one repeat
        results.append(result)

    df = pd.DataFrame(results)

    # Convert columns to numeric types
    numeric_columns = ['Size', 'Eigenvalue Density', 'Rank Reduction Steps',
                       'Original Objective', 'Squeezed Objective', 'Objective Difference',
                       'Original Runtime (s)', 'Squeezed Runtime (s)']
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

    # Compute % Solve Time Difference
    # Only compute if both runs reached optimality
    def compute_percentage_difference(row):
        if row['Original Status'] == 'OPTIMAL' and row['Squeezed Status'] == 'OPTIMAL':
            return ((row['Squeezed Runtime (s)'] - row['Original Runtime (s)']) / row['Original Runtime (s)']) * 100
        else:
            return np.nan  # Cannot compute meaningful percentage difference

    df['% Solve Time Difference'] = df.apply(compute_percentage_difference, axis=1)

    # Handle infinite and NaN values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # We may choose to keep NaN values to show where timeouts occurred

    print("\nSummary of Results:")
    print(df)

    # CSV output
    df.to_csv("rank_reduction_analysis_results.csv", index=False)

    # Generate plots
    generate_plots(df)

def generate_plots(df: pd.DataFrame):
    sns.set(style="whitegrid")

    # Average over repeats where both runs reached optimality
    df_filtered = df[(df['Original Status'] == 'OPTIMAL') & (df['Squeezed Status'] == 'OPTIMAL')]
    df_grouped = df_filtered.groupby('Size').mean().reset_index()

    # Plot Eigenvalue Density vs Matrix Size
    plt.figure(figsize=(12, 6))
    sns.lineplot(x="Size", y="Eigenvalue Density", data=df_grouped)
    plt.title("Average Eigenvalue Density by Matrix Size")
    plt.ylabel("Eigenvalue Density")
    plt.xlabel("Matrix Size (n)")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig("eigenvalue_density.png")
    plt.show()

    # Plot Rank Reduction Steps vs Matrix Size
    plt.figure(figsize=(12, 6))
    sns.lineplot(x="Size", y="Rank Reduction Steps", data=df_grouped)
    plt.title("Average Rank Reduction Steps by Matrix Size")
    plt.ylabel("Number of Iterations")
    plt.xlabel("Matrix Size (n)")
    plt.tight_layout()
    plt.savefig("rank_reduction_steps.png")
    plt.show()

    # Plot Objective Difference vs Matrix Size
    plt.figure(figsize=(12, 6))
    sns.lineplot(x="Size", y="Objective Difference", data=df_grouped)
    plt.title("Average Objective Difference by Matrix Size")
    plt.ylabel("Absolute Difference")
    plt.xlabel("Matrix Size (n)")
    plt.tight_layout()
    plt.savefig("objective_difference.png")
    plt.show()

    # Plot % Solve Time Difference vs Matrix Size
    plt.figure(figsize=(12, 6))
    sns.lineplot(x="Size", y="% Solve Time Difference", data=df_grouped)
    plt.title("Average % Solve Time Difference by Matrix Size")
    plt.ylabel("Percentage Solve Time Difference (%)")
    plt.xlabel("Matrix Size (n)")
    plt.tight_layout()
    plt.savefig("solve_time_difference.png")
    plt.show()

    # Plot Runtime Comparison
    plt.figure(figsize=(12, 6))
    plt.plot(df_grouped['Size'], df_grouped['Original Runtime (s)'], label='Original', color='skyblue')
    plt.plot(df_grouped['Size'], df_grouped['Squeezed Runtime (s)'], label='Squeezed', color='salmon')
    plt.xlabel('Matrix Size (n)')
    plt.ylabel('Average Runtime (seconds)')
    plt.title('Average Solver Runtime Comparison')
    plt.legend()
    plt.tight_layout()
    plt.savefig("solver_runtime_comparison.png")
    plt.show()

if __name__ == "__main__":
    test_matrices()
