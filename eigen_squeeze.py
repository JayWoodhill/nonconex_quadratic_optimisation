import numpy as np
import gurobipy as gp
from gurobipy import GRB
import logging
import time


FLOAT_TOL = 1e-10
RELATIVE_EIGEN_TOL = 1e-4
RELAXED_FLOAT_TOL = 1e-1
MAX_ITERATION = 9999

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

def nonzero_eigenvalues(evals):
    return np.where(np.abs(evals) > FLOAT_TOL)[0]

def relative_nonzero_eigenvalues(evals):
    relative_evals = np.abs(evals) / np.sum(np.abs(evals))
    return np.where(relative_evals > RELATIVE_EIGEN_TOL)[0]


def iterative_eigen_squeeze(
    mat: np.ndarray, linear_coeffs: np.ndarray, linear_rhs: float
):
    # Avoid mutating mat
    mat = np.copy(mat).astype(float)

    # Initialize offset
    offset = np.zeros_like(linear_coeffs).astype(float)

    # Initialize eigenvalues history
    eigenvalues_history = []

    # Save previous state
    prev_mat = mat
    prev_offset = offset

    iterations = 1
    while iterations <= MAX_ITERATION:
        # Get eigenvalues and eigenvectors
        evals, evecs = np.linalg.eig(mat)

        # Record eigenvalues history
        eigenvalues_history.append(np.sort(evals.real))

        # Check for complex eigenvalues
        if np.any(np.iscomplex(evals)):
            return prev_mat, prev_offset, iterations - 1, eigenvalues_history

        # Get significant eigenvalues
        nonzero_evals_index = relative_nonzero_eigenvalues(evals)
        evals = evals[nonzero_evals_index]
        evecs = evecs[:, nonzero_evals_index]

        # Check if already concave (negative semidefinite)
        if np.all(evals <= FLOAT_TOL):
            return mat, offset, iterations, eigenvalues_history

        # Identify the largest positive eigenvalue
        squeezing_eval_index = np.argmax(evals)
        squeezing_eval = evals[squeezing_eval_index]
        squeezing_evec = evecs[:, squeezing_eval_index]

        # Determine alpha
        _denominator = squeezing_evec.dot(linear_coeffs)
        if abs(_denominator) < FLOAT_TOL:
            return mat, offset, iterations, eigenvalues_history

        alpha = -squeezing_eval / _denominator

        # Save previous
        prev_mat = np.copy(mat)
        prev_offset = np.copy(offset)

        # Update matrix
        mat += alpha * np.outer(squeezing_evec, linear_coeffs)

        # Update offset
        offset -= alpha * linear_rhs * squeezing_evec

        iterations += 1

    # If maximum iterations reached, return current state
    return mat, offset, iterations, eigenvalues_history
1

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
        logger.warning(f"Model did not solve to optimality. Status: {status_message}")
        return None, runtime, status_message, None

    # Get the objective value and solution
    obj_val = mdl.ObjVal
    x_values = np.array([x[i].X for i in range(n)])

    return obj_val, runtime, status_message, x_values

def analyse_matrix(matrix: np.ndarray, linear_coeffs: np.ndarray, linear_rhs: float):
    n = matrix.shape[0]
    logger.info(f"\nAnalysing a matrix of size {n}x{n}")

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
        "Original Runtime (s)": runtime_orig,
        "Squeezed Runtime (s)": runtime_squeezed,
        "Original Status": status_orig,
        "Squeezed Status": status_squeezed,
    }

    return result

'''def gurobi_solve(
    matrix: np.ndarray,
    linear_objective: np.ndarray,
    linear_coeff: np.ndarray,
    linear_rhs: np.ndarray,
):
    n = len(linear_coeff)
    mdl = gp.Model()
    x = [mdl.addVar(vtype=GRB.BINARY) for _ in range(n)]
    mdl.addConstr(sum(x[i] * linear_coeff[i] for i in range(n)) == linear_rhs)
    mdl.setObjective(
        sum(matrix[i, j] * x[i] * x[j] for i in range(n) for j in range(n))
        + sum(linear_objective[i] * x[i] for i in range(n)),
        gp.GRB.MAXIMIZE,
    )

    # solve and return
    mdl.setParam("OutputFlag", 0)
    mdl.optimize()

    if mdl.status != gp.GRB.OPTIMAL:
        raise Exception("Something went wrong, model did not solve")

    return mdl.objVal, mdl.Runtime'''


'''def test_iterative_eigen_squeeze(verbose=False):
    # generate a random symetric matrix
    for _ in range(100):
        # Setup parameters
        n = 100
        original_matrix = np.random.randint(-100, 100, size=(n, n))
        original_matrix += original_matrix.T  # (make it symetric)
        linear_coeffs = np.random.randint(-100, 100, size=n)
        linear_rhs = np.random.randint(-100, 100)

        # Find squeezed matrix
        squeezed_matrix, offset, iterations = iterative_eigen_squeeze(
            original_matrix, linear_coeffs, linear_rhs
        )

        # Check it operated as expected on eigenvalues
        original_evals = relative_nonzero_eigenvalues(
            np.linalg.eig(original_matrix).eigenvalues
        )
        num_original_problematic_eigens = np.sum(original_evals > FLOAT_TOL)
        squeezed_evals = relative_nonzero_eigenvalues(
            np.linalg.eig(squeezed_matrix).eigenvalues
        )
        num_squeezed_problematic_eigens = np.sum(squeezed_evals > FLOAT_TOL)

        if verbose:
            print(
                f"Remove {num_original_problematic_eigens - num_squeezed_problematic_eigens} eigenvalues"
            )

        assert iterations > 1
        if iterations > 2:
            # If we have hit 2 iterations, MUST have removed at least one problematic
            assert num_original_problematic_eigens > num_squeezed_problematic_eigens
        else:
            assert num_original_problematic_eigens >= num_squeezed_problematic_eigens

        # Check that function values match
        for _ in range(100):
            # Generate a random x that satisfies our constraint
            x = np.random.randint(0, 100, size=n).astype(float)
            # Find a component with nonzero entry
            nonzero_coeff_index = np.where(linear_coeffs)[0][0]
            # update respective x component to make sure sums match
            x[nonzero_coeff_index] = 0
            x[nonzero_coeff_index] = (
                linear_rhs - x.dot(linear_coeffs)
            ) / linear_coeffs[nonzero_coeff_index]
            assert abs(x.dot(linear_coeffs) - linear_rhs) < FLOAT_TOL

            # Now we know x satisfies our constraint, check its fn value matches
            original_value = x @ original_matrix @ x
            squeezed_value = x @ squeezed_matrix @ x + x @ offset
            assert abs(original_value - squeezed_value) < RELAXED_FLOAT_TOL'''


'''def test_gurobi_solve():
    for _ in range(10):
        n = 25
        original_matrix = np.random.randint(-100, 100, size=(n, n))
        original_matrix += original_matrix.T  # (make it symetric)
        linear_coeffs = np.ones(n)
        linear_rhs = np.random.randint(2, n - 1)
        # Find squeezed matrix
        squeezed_matrix, offset, iterations = iterative_eigen_squeeze(
            original_matrix, linear_coeffs, linear_rhs
        )
        obj_orig, runtime_orig = gurobi_solve(
            original_matrix, np.zeros(n), linear_coeffs, linear_rhs
        )
        obj_squeezed, runtime_squeezed = gurobi_solve(
            squeezed_matrix, offset, linear_coeffs, linear_rhs
        )
        assert abs(obj_orig - obj_squeezed) < RELAXED_FLOAT_TOL
        print(f"orig : {runtime_orig}\tsqueezed : {runtime_squeezed}")'''


'''if __name__ == "__main__":
    test_iterative_eigen_squeeze()
    test_gurobi_solve()'''
