The project aims to:

- Implement an iterative rank reduction technique to adjust symmetric matrices to be positive semidefinite (convex) or negative semidefinite (concave).
- Integrate with Gurobi to solve quadratic programming problems using both original and rank-reduced matrices.
- Record and analyse matrix characteristics, such as density, condition number, and eigenvalue distribution.
- Compare solver performance (e.g., computation time, solution quality) before and after applying rank reduction.
- Conduct experiments to identify relationships between matrix characteristics and the effectiveness of the rank reduction technique.

---
Features

- Matrix Generation: Create symmetric matrices with specified eigenvalue distributions.
- Matrix Analysis: Compute various metrics and characteristics of matrices.
- Iterative Rank Reduction: Adjust matrices to desired definiteness through iterative updates.
- Quadratic Programming Solver: Solve QP problems using Gurobi, with and without rank reduction.
- Experiment Automation: Run experiments across multiple configurations and trials.
- Logging and Reporting: Detailed logging of computational steps and results, with options for customization.
- Results Analysis: Save results to CSV for further analysis and visualization.

---
Prerequisites

Python 3.6 or higher
Gurobi Optimizer (with a valid license)

---
Dependencies

- Git
git clone https://github.com/yourusername/iterative-rank-reduction.git
cd iterative-rank-reduction

- Python
pip install -r requirements.txt

- Venv
python -m venv venv
source venv/bin/activate    



