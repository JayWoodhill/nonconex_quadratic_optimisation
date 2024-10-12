import numpy as np
import gurobipy as gp
from gurobipy import GRB

FLOAT_TOL = 1e-10
RELATIVE_EIGEN_TOL = 1e-4
RELAXED_FLOAT_TOL = 1e-1
MAX_ITERATION = 9999


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

    # Save out offset
    offset = np.zeros_like(linear_coeffs).astype(float)

    # Fake save previous for now
    prev_mat = mat
    prev_offset = offset

    iterations = 1
    while iterations <= MAX_ITERATION:
        # Get eigevalues and eigenvectors
        evals, evecs = np.linalg.eig(mat)

        # Check if complex?
        if np.any(np.iscomplex(evals)):
            return prev_mat, prev_offset, iterations - 1

        # Get nonzeros
        nonzero_evals_index = relative_nonzero_eigenvalues(evals)
        evals = evals[nonzero_evals_index]
        evecs = evecs[:, nonzero_evals_index]

        # Check if already concave?
        if np.all(evals <= FLOAT_TOL):
            return mat, offset, iterations

        # Get our eigenvector of interest
        squeezing_eval_index = np.argmax(evals)
        squeezing_eval = evals[squeezing_eval_index]
        squeezing_evec = evecs[:, squeezing_eval_index]

        # Determine $\alpha$ (provided we actually can)
        _denominator = squeezing_evec.dot(linear_coeffs)
        if abs(_denominator) < FLOAT_TOL:
            return mat, offset, iterations
        alpha = -squeezing_eval / _denominator

        # Save previous
        prev_mat = np.copy(mat)
        prev_offset = np.copy(offset)

        # Update matrix
        mat += alpha * np.outer(squeezing_evec, linear_coeffs)

        # Update offset
        offset -= alpha * linear_rhs * squeezing_evec

        iterations += 1


def gurobi_solve(
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

    return mdl.objVal, mdl.Runtime


def test_iterative_eigen_squeeze(verbose=False):
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
            assert abs(original_value - squeezed_value) < RELAXED_FLOAT_TOL


def test_gurobi_solve():
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
        print(f"orig : {runtime_orig}\tsqueezed : {runtime_squeezed}")


if __name__ == "__main__":
    test_iterative_eigen_squeeze()
    test_gurobi_solve()
