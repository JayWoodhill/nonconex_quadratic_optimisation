import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_Q_with_small_negative_eigenvalue(n, epsilon):
    """
    Generates an n x n symmetric matrix with one small negative eigenvalue.

    Parameters:
    - n (int): Size of the matrix.
    - epsilon (float): The magnitude of the small negative eigenvalue.

    Returns:
    - Q (numpy.ndarray): The generated symmetric matrix.
    """
    # Generate random positive definite matrix
    A = np.random.randn(n, n)
    Q_posdef = A.T @ A  # Ensures Q is positive semidefinite

    # Compute eigenvalues and eigenvectors
    eigvals, eigvecs = np.linalg.eigh(Q_posdef)

    # Introduce a small negative eigenvalue
    eigvals[0] = -epsilon  # Set the smallest eigenvalue to -epsilon

    # Reconstruct Q with the modified eigenvalues
    Q = eigvecs @ np.diag(eigvals) @ eigvecs.T

    # Ensure symmetry
    Q = (Q + Q.T) / 2

    return Q

def iterative_rank_reduction_with_c(Q, c, A=None, b=None, mode='convex', max_iterations=100, tol=1e-8):
    """
    Iteratively adjusts the input symmetric matrix `Q` to make it positive semidefinite (`mode='convex'`)
    by eliminating undesired eigenvalues through symmetric rank-one updates,
    while adjusting the linear term `c` to compensate and preserve the QP problem's equivalence.

    Parameters:
    - Q (numpy.ndarray): The input symmetric matrix to be adjusted.
    - c (numpy.ndarray): The linear coefficient vector.
    - A (numpy.ndarray, optional): The constraint matrix representing the problem's constraints.
    - b (numpy.ndarray or float, optional): The constraint value(s).
    - mode (str): 'convex' to make the matrix positive semidefinite, 'concave' for negative semidefinite.
    - max_iterations (int): Maximum number of iterations to prevent infinite loops.
    - tol (float): Tolerance for numerical comparisons.

    Returns:
    - dict: Contains the adjusted Q and c, and additional information.
    """
    logger = logging.getLogger(__name__)

    matrices = []
    eigenvalues_set = []
    eigenvectors_set = []
    delta_q_list = []
    delta_c_list = []
    alpha_list = []
    level = 0
    cumulative_delta_q = np.zeros_like(Q, dtype=float)
    cumulative_delta_c = np.zeros_like(c, dtype=float) if c is not None else None

    # check a and b
    if (A is None and (c is not None or b is not None)) or (A is not None and (c is None or b is None)):
        raise ValueError("When adjusting 'c', both A and b must be provided.")

    Q = Q.copy().astype(float)
    c = c.copy().astype(float) if c is not None else None
    current_matrix = Q.copy()
    current_c = c.copy() if c is not None else None

    if A is not None and b is not None:
        # Solve A x = b for x_star (minimum norm solution)
        try:
            x_star = np.linalg.lstsq(A, b, rcond=None)[0]
        except np.linalg.LinAlgError:
            raise ValueError("Cannot find a feasible solution x_star for the constraints A x = b.")
    else:
        x_star = None  # No feasible solution to preserve

    while level < max_iterations:
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(current_matrix)

        # Store
        matrices.append(current_matrix.copy())
        eigenvalues_set.append(eigenvalues.copy())
        eigenvectors_set.append(eigenvectors.copy())

        # Check for complex eigenvalues
        if np.iscomplexobj(eigenvalues):
            final_status = f"Complex eigenvalues encountered after {level} iterations."
            logger.warning(final_status)
            break

        # undesired eigenvalues
        if mode == 'convex':
            undesired_indices = np.where(eigenvalues < -tol)[0]
            desired_condition = np.all(eigenvalues >= -tol)
            final_definiteness = "positive semidefinite"
        elif mode == 'concave':
            undesired_indices = np.where(eigenvalues > tol)[0]
            desired_condition = np.all(eigenvalues <= tol)
            final_definiteness = "negative semidefinite"
        else:
            raise ValueError("Invalid mode. Use 'convex' or 'concave'.")

        if desired_condition:
            final_status = f"Desired definiteness achieved after {level} iterations."
            logger.info(final_status)
            break

        if len(undesired_indices) == 0:
            final_status = f"No further adjustments needed after {level} iterations."
            logger.info(final_status)
            break

        # Select the most negative (or positive) eigenvalue to adjust
        idx = undesired_indices[0]
        undesired_eigenvalue = eigenvalues[idx]
        eigenvector = eigenvectors[:, idx]


        if mode == 'convex' and x_star is not None:
            # Compute the adjustment alpha
            dot_product = np.dot(A, eigenvector)
            if A.ndim == 1:
                dot_product = np.dot(A, eigenvector)
            else:
                dot_product = np.dot(A, eigenvector)

            # Ensure the dot product is not zero to avoid division by zero
            # maybe this needs a tolerance check for stability
            if np.abs(dot_product) < tol:
                final_status = f"Zero dot product between A and eigenvector at iteration {level}."
                logger.warning(final_status)
                break


            alpha = -undesired_eigenvalue / dot_product
            alpha_list.append(alpha)

            # Compute delta_q
            delta_q = alpha * np.outer(eigenvector, A.flatten())
            delta_q = (delta_q + delta_q.T) / 2  # Ensure symmetry

            # Update Q
            current_matrix += delta_q
            cumulative_delta_q += delta_q
            delta_q_list.append(delta_q.copy())

            # Compute delta_c to preserve equivalence: delta_c = - delta_q x_star
            if x_star is not None:
                delta_c = -delta_q @ x_star
                current_c += delta_c
                cumulative_delta_c += delta_c
                delta_c_list.append(delta_c.copy())
        else:
            # If not adjusting for a QP problem, proceed as usual
            u = np.ones_like(eigenvector)
            dot_product = np.dot(u, eigenvector)
            if np.abs(dot_product) < tol:
                final_status = f"Zero dot product between u and eigenvector at iteration {level}."
                logger.warning(final_status)
                break

            alpha = -undesired_eigenvalue / dot_product
            alpha_list.append(alpha)

            # Compute delta_q symmetrically
            delta_q = alpha * (np.outer(u, eigenvector) + np.outer(eigenvector, u)) / 2
            current_matrix += delta_q
            cumulative_delta_q += delta_q
            delta_q_list.append(delta_q.copy())

            # No adjustment to c since it's not part of the problem
            delta_c = None

        # Debugging output
        '''logger.info(f"Iteration {level}:")
        logger.info(f"  Undesired Eigenvalue: {undesired_eigenvalue}")
        logger.info(f"  Alpha: {alpha}")
        if mode == 'convex' and x_star is not None:
            logger.info(f"  Norm of delta_q: {np.linalg.norm(delta_q)}")
            logger.info(f"  Norm of delta_c: {np.linalg.norm(delta_c)}")
        else:
            logger.info(f"  Norm of delta_q: {np.linalg.norm(delta_q)}")'''
        # Log the minimum eigenvalue after adjustment
        min_eigenvalue = np.min(np.linalg.eigvalsh(current_matrix))
        logger.info(f"  Min Eigenvalue of current_matrix: {min_eigenvalue}")
        logger.info("-" * 50)

        level += 1

    else:
        final_status = f"Maximum iterations ({max_iterations}) reached without achieving desired definiteness."
        logger.warning(final_status)

    # Adjust results back if concave mode was used
    if mode == 'concave':
        matrices = [-M for M in matrices]
        current_matrix = -current_matrix
        eigenvalues_set = [-eigvals for eigvals in eigenvalues_set]
        delta_q_list = [-dq for dq in delta_q_list]
        if delta_c_list is not None:
            delta_c_list = [-dc for dc in delta_c_list]
        final_status = final_status.replace("positive semidefinite", "negative semidefinite")

    # results
    result = {
        "adjusted_q": current_matrix,
        "adjusted_c": current_c,
        "matrices": matrices,
        "eigenvalues_set": eigenvalues_set,
        "eigenvectors_set": eigenvectors_set,
        "delta_q_list": delta_q_list,
        "delta_c_list": delta_c_list,
        "levels_performed": level,
        "final_status": final_status,
        "cumulative_delta_q": cumulative_delta_q,
        "cumulative_delta_c": cumulative_delta_c,
        "alpha_list": alpha_list
    }

    return result

