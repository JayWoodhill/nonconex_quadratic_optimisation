import unittest
import numpy as np
import numpy.testing as npt
import gurobipy as gp
from gurobipy import GRB
import logging
from spectral_perturbation_analyser import solve_qp_with_gurobi, gen_real_constrained_matrix, matrix_checks, iterative_rank_reduction

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class TestSolveQPWithGurobi(unittest.TestCase):
    def test_simple_unconstrained_qp(self):
        """Test solving a simple unconstrained quadratic program."""
        Q = np.array([[2, 0], [0, 2]])
        c = np.array([-4, -6])
        result = solve_qp_with_gurobi(Q, c)
        x_expected = np.array([2.0, 3.0])
        objective_expected = -13.0

        x_solution = result['variable_values']
        objective_value = result['objective_value']
        status = result['status']

        # Check if the optimization was successful
        self.assertEqual(status, GRB.OPTIMAL)
        # Check if the solution is as expected
        npt.assert_allclose(x_solution, x_expected, rtol=1e-5)
        # Check if the objective value is as expected
        self.assertAlmostEqual(objective_value, objective_expected, places=5)

    def test_qp_with_bounds(self):
        """Test solving a quadratic program with variable bounds."""
        Q = np.array([[2, 0], [0, 2]])
        c = np.array([-4, -6])
        bounds = [(0, None), (0, None)]  # x >= 0
        result = solve_qp_with_gurobi(Q, c, bounds=bounds)
        x_expected = np.array([2.0, 3.0])
        objective_expected = -13.0

        x_solution = result['variable_values']
        objective_value = result['objective_value']
        status = result['status']

        self.assertEqual(status, GRB.OPTIMAL)
        npt.assert_allclose(x_solution, x_expected, rtol=1e-5)
        self.assertAlmostEqual(objective_value, objective_expected, places=5)

    def test_qp_with_inequality_constraints(self):
        """Test solving a quadratic program with inequality constraints."""
        Q = np.array([[2, 0], [0, 2]])
        c = np.array([-4, -6])
        A = np.array([[1, 1]])
        b = np.array([4])  # Constraint: x1 + x2 <= 4
        result = solve_qp_with_gurobi(Q, c, A=A, b=b)
        x_expected = np.array([1.5, 2.5])
        objective_expected = -12.5

        x_solution = result['variable_values']
        objective_value = result['objective_value']
        status = result['status']

        self.assertEqual(status, GRB.OPTIMAL)
        npt.assert_allclose(x_solution, x_expected, rtol=1e-5)
        self.assertAlmostEqual(objective_value, objective_expected, places=5)

    def test_non_positive_definite_Q(self):
        """Test solving a quadratic program where Q is not positive semidefinite."""
        Q = np.array([[0, 0], [0, -1]])
        c = np.array([0, 0])

        try:
            result = solve_qp_with_gurobi(Q, c)
            status = result['status']
            self.assertNotEqual(status, GRB.OPTIMAL)
        except gp.GurobiError as e:
            logger.info(f"GurobiError as expected: {e}")

    def test_error_handling_incorrect_dimensions(self):
        """Test error handling when dimensions of Q and c do not match."""
        Q = np.array([[2, 0], [0, 2]])
        c = np.array([-4, -6, -8])  # Incorrect dimension

        with self.assertRaises(ValueError):
            solve_qp_with_gurobi(Q, c)

    def test_variable_bounds(self):
        """Test variable upper bounds."""
        Q = np.array([[2, 0], [0, 2]])
        c = np.array([-4, -6])
        bounds = [(None, 2), (None, 2)]  # x1 <= 2, x2 <= 2
        result = solve_qp_with_gurobi(Q, c, bounds=bounds)
        x_expected = np.array([2.0, 2.0])
        objective_expected = -12.0

        x_solution = result['variable_values']
        objective_value = result['objective_value']
        status = result['status']

        self.assertEqual(status, GRB.OPTIMAL)
        npt.assert_allclose(x_solution, x_expected, rtol=1e-5)
        self.assertAlmostEqual(objective_value, objective_expected, places=5)

class TestGenRealConstrainedMatrix(unittest.TestCase):
    def test_matrix_generation(self):
        """Test the generation of a real constrained matrix."""
        n = 5
        u = 2  # Number of positive eigenvalues
        v = 2  # Number of negative eigenvalues
        matrix = gen_real_constrained_matrix(n, u, v)
        eigenvalues = np.linalg.eigvalsh(matrix)

        # Counts
        positive_count = np.sum(eigenvalues > 1e-5)
        negative_count = np.sum(eigenvalues < -1e-5)
        zero_count = n - positive_count - negative_count

        self.assertEqual(positive_count, u)
        self.assertEqual(negative_count, v)
        self.assertEqual(zero_count, n - u - v)

class TestMatrixChecks(unittest.TestCase):
    def test_symmetric_matrix_checks(self):
        """Test the matrix_checks function with a symmetric matrix."""
        matrix = np.array([[2, 1], [1, 2]])
        results = matrix_checks(matrix)
        self.assertTrue(results['is_positive_semidefinite'])
        self.assertEqual(results['rank'], 2)

class TestIterativeRankReduction(unittest.TestCase):
    def test_rank_reduction_convex(self):
        """Test the iterative rank reduction in convex mode."""
        q = np.array([[1, 2], [2, -3]])
        result = iterative_rank_reduction(q, mode='convex')
        final_matrix = result['matrices'][-1]
        eigenvalues = np.linalg.eigvalsh(final_matrix)
        self.assertTrue(np.all(eigenvalues >= -1e-10))

    def test_rank_reduction_concave(self):
        """Test the iterative rank reduction in concave mode."""
        q = np.array([[1, 2], [2, -3]])
        result = iterative_rank_reduction(q, mode='concave')
        final_matrix = result['matrices'][-1]
        eigenvalues = np.linalg.eigvalsh(final_matrix)
        self.assertTrue(np.all(eigenvalues <= 1e-10))

if __name__ == '__main__':
    unittest.main()
