import numpy as np
import logging

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




