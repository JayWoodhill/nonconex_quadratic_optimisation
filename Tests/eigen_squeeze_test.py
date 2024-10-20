import numpy as np
import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import time
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from eigen_squeeze import (nonzero_eigenvalues,
    relative_nonzero_eigenvalues,
    iterative_eigen_squeeze,
    analyse_matrix

)

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
def test_matrices():
    results = []
    #sizes = [5, 10, 21, 22, 23, 24, 25, 26, 27, 28, 39, 30]  # Adjust sizes as needed
    sizes = list(range(5, 15, 1))

    for n in sizes:
        logger.info(f"\nTesting with matrix size: {n}x{n}")
        random_matrix = np.random.randint(-10, 10, size=(n, n))
        random_matrix = (random_matrix + random_matrix.T) / 2
        linear_coeffs = np.ones(n)
        linear_rhs = np.random.randint(1, n)

        result = analyse_matrix(random_matrix, linear_coeffs, linear_rhs)
        results.append(result)


    df = pd.DataFrame(results)
    print("\nSummary of Results:")
    print(df)

    # csv outputs
    df.to_csv("rank_reduction_analysis_results.csv", index=False)

    # plots
    generate_plots(df)

def generate_plots(df: pd.DataFrame):
    sns.set(style="whitegrid")

    # Plot Eigenvalue Density vs Matrix Size
    plt.figure(figsize=(8, 6))
    sns.barplot(x="Size", y="Eigenvalue Density", data=df, palette="viridis")
    plt.title("Eigenvalue Density by Matrix Size")
    plt.ylabel("Eigenvalue Density")
    plt.xlabel("Matrix Size (n)")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig("eigenvalue_density.png")
    plt.show()

    # Plot Rank Reduction Steps vs Matrix Size
    plt.figure(figsize=(8, 6))
    sns.barplot(x="Size", y="Rank Reduction Steps", data=df, palette="magma")
    plt.title("Rank Reduction Steps by Matrix Size")
    plt.ylabel("Number of Iterations")
    plt.xlabel("Matrix Size (n)")
    plt.tight_layout()
    plt.savefig("rank_reduction_steps.png")
    plt.show()

    # Plot Original vs Squeezed Objective Values
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x="Original Objective", y="Squeezed Objective", hue="Size", data=df, palette="deep", s=100)
    plt.title("Original vs Squeezed Objective Values")
    plt.xlabel("Original Objective")
    plt.ylabel("Squeezed Objective")
    plt.legend(title="Matrix Size")
    plt.tight_layout()
    plt.savefig("objective_values_comparison.png")
    plt.show()

    # Plot Objective Difference vs Matrix Size
    plt.figure(figsize=(8, 6))
    sns.barplot(x="Size", y="Objective Difference", data=df, palette="coolwarm")
    plt.title("Objective Difference by Matrix Size")
    plt.ylabel("Absolute Difference")
    plt.xlabel("Matrix Size (n)")
    plt.tight_layout()
    plt.savefig("objective_difference.png")
    plt.show()

    # Plot Runtime Comparison
    plt.figure(figsize=(10, 6))
    width = 0.35  # Width of the bars
    x = np.arange(len(df))
    plt.bar(x - width/2, df["Original Runtime (s)"], width, label='Original', color='skyblue')
    plt.bar(x + width/2, df["Squeezed Runtime (s)"], width, label='Squeezed', color='salmon')
    plt.xlabel('Matrix Size (n)')
    plt.ylabel('Runtime (seconds)')
    plt.title('Solver Runtime Comparison')
    plt.xticks(x, df["Size"])
    plt.legend()
    plt.tight_layout()
    plt.savefig("solver_runtime_comparison.png")
    plt.show()

    # Heatmap of Objective Differences
    plt.figure(figsize=(8, 6))
    sns.heatmap(df.pivot("Size", "Rank Reduction Steps", "Objective Difference"), annot=True, fmt=".2e", cmap="YlGnBu")
    plt.title("Objective Differences Heatmap")
    plt.xlabel("Rank Reduction Steps")
    plt.ylabel("Matrix Size (n)")
    plt.tight_layout()
    plt.savefig("objective_differences_heatmap.png")
    plt.show()

if __name__ == "__main__":
    test_matrices()