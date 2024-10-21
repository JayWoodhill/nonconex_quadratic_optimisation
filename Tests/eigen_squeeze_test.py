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
    iterative_eigen_squeeze
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
    time_limit=30  # Enforce 60-second timeout
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

    # Calculate density before rank reduction
    total_elements = n * n
    nonzero_elements_before = np.count_nonzero(matrix)
    density_before = nonzero_elements_before / total_elements
    logger.info(f"Density before rank reduction: {density_before:.4f}")

    # Solve using Gurobi for the original matrix
    obj_orig, runtime_orig, status_orig, x_orig = gurobi_solve(
        matrix, np.zeros(n), linear_coeffs, linear_rhs
    )
    logger.info(f"Gurobi Original - Status: {status_orig}, Runtime: {runtime_orig:.4f}s, Objective: {obj_orig}")
    solvable_orig = status_orig == "OPTIMAL"

    # Apply the iterative eigenvalue squeezing to process the matrix
    squeezed_matrix, offset, iterations, eigenvalues_history = iterative_eigen_squeeze(
        matrix, linear_coeffs, linear_rhs
    )
    logger.info(f"Number of steps of rank reduction: {iterations}")

    # Calculate density after rank reduction
    nonzero_elements_after = np.count_nonzero(squeezed_matrix)
    density_after = nonzero_elements_after / total_elements
    logger.info(f"Density after rank reduction: {density_after:.4f}")

    # Calculate change in density
    density_change = density_after - density_before
    logger.info(f"Change in density: {density_change:.4f}")

    # Solve using Gurobi for the squeezed matrix
    obj_squeezed, runtime_squeezed, status_squeezed, x_squeezed = gurobi_solve(
        squeezed_matrix, offset, linear_coeffs, linear_rhs
    )
    logger.info(f"Gurobi Squeezed - Status: {status_squeezed}, Runtime: {runtime_squeezed:.4f}s, Objective: {obj_squeezed}")
    solvable_squeezed = status_squeezed == "OPTIMAL"

    # Compare objectives
    if obj_orig is not None and obj_squeezed is not None:
        obj_diff = abs(obj_orig - obj_squeezed)
        objectives_differ = obj_diff > FLOAT_TOL  # Or use a more appropriate tolerance
        logger.info(f"Difference in objective values: {obj_diff:.4e}")
    else:
        obj_diff = None
        objectives_differ = None

    # Determine if preprocessing led to faster solve times
    if solvable_orig and solvable_squeezed:
        faster_with_preprocessing = runtime_squeezed < runtime_orig
        runtime_difference = runtime_orig - runtime_squeezed
        percent_solve_time_difference = ((runtime_squeezed - runtime_orig) / runtime_orig) * 100 if runtime_orig > 0 else 0
    else:
        faster_with_preprocessing = None
        runtime_difference = None
        percent_solve_time_difference = None

    # Collect data for analysis
    result = {
        "Size": n,
        "Eigenvalue Density": np.sum(np.linalg.eigvals(matrix) > FLOAT_TOL) / n,
        "Rank Reduction Steps": iterations,
        "Original Objective": obj_orig,
        "Squeezed Objective": obj_squeezed,
        "Objective Difference": obj_diff,
        "Objectives Differ": objectives_differ,
        "Original Runtime (s)": runtime_orig,
        "Squeezed Runtime (s)": runtime_squeezed,
        "Runtime Difference": runtime_difference,
        "% Solve Time Difference": percent_solve_time_difference,
        "Faster with Preprocessing": faster_with_preprocessing,
        "Original Status": status_orig,
        "Squeezed Status": status_squeezed,
        "Solvable Original": solvable_orig,
        "Solvable Squeezed": solvable_squeezed,
        "Density Before": density_before,
        "Density After": density_after,
        "Density Change": density_change,
    }

    return result


def test_matrices():
    results = []

    # Test sizes from 5 to 50, incrementing by 1, repeated 3 times
    sizes_small = list(range(5, 25))  # 5 to 50 inclusive
    repeats_small = 5

    ''''# Test sizes from 10 to 500, incrementing by 10, repeated once
    sizes_large = list(range(10, 30, 10))  # 10 to 500 inclusive
    repeats_large = 1'''

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
    '''for n in sizes_large:
        logger.info(f"\nTesting with matrix size: {n}x{n}")
        random_matrix = np.random.randint(-10, 10, size=(n, n))
        random_matrix = (random_matrix + random_matrix.T) / 2
        linear_coeffs = np.ones(n)
        linear_rhs = np.random.randint(1, n)

        result = analyse_matrix(random_matrix, linear_coeffs, linear_rhs)
        result['Repeat'] = 1  # Only one repeat
        results.append(result)'''

    df = pd.DataFrame(results)

    # Convert columns to numeric types
    numeric_columns = ['Size', 'Eigenvalue Density', 'Rank Reduction Steps',
                       'Original Objective', 'Squeezed Objective', 'Objective Difference',
                       'Original Runtime (s)', 'Squeezed Runtime (s)']
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

    df = pd.DataFrame(results)
    print(df['Size'])
    print('test1')
    # Ensure 'Size' is a numeric scalar
    df['Size'] = pd.to_numeric(df['Size'], errors='coerce')

    # Check for any NaN values in 'Size'
    if df['Size'].isnull().any():
        print("Warning: Some 'Size' values are NaN after conversion.")
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

    # Filter the DataFrame to include only optimal solutions
    df_filtered = df[
        (df['Original Status'] == 'OPTIMAL') & (df['Squeezed Status'] == 'OPTIMAL')
        ]

    # Select numeric columns for aggregation, excluding 'Size'
    numeric_columns = df_filtered.select_dtypes(include=[np.number]).columns.tolist()
    numeric_columns = [col for col in numeric_columns if col != 'Size']

    # Include 'Size' in the list of columns to be used
    columns_to_use = ['Size'] + numeric_columns

    # Group by 'Size' and compute the mean of numeric columns without setting 'Size' as index
    df_grouped = df_filtered[columns_to_use].groupby('Size', as_index=False).mean()

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

    # Scatter Plot: Density Change vs Runtime Difference
    plt.figure(figsize=(8, 6))
    # Only include cases where both runtimes are available
    valid_cases = df_filtered.dropna(subset=["Density Change", "Runtime Difference"])
    sns.scatterplot(x="Density Change", y="Runtime Difference", hue="Size", data=valid_cases)
    plt.title("Density Change vs Runtime Difference")
    plt.xlabel("Density Change (After - Before)")
    plt.ylabel("Runtime Difference (Original - Squeezed)")
    plt.legend(title="Matrix Size")
    plt.axhline(0, color='grey', linestyle='--')
    plt.tight_layout()
    plt.savefig("density_change_vs_runtime_difference.png")
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
