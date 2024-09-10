import numpy as np
import numpy.linalg as  linalg

'''n = 5

m = np.random.rand(n,n)

print(linalg.eig(m))

# create function with input metrics for eigenvalue counts'''

#eig = [1,2,-5]

# setting P to a random square invertible matrix with dimension len(eigenvalues)
# identity $A = PDP^-1$ means that if the diagonal matrix D (xi ** I) is defiined, a random invertible matrix P can be set and used arbitrarily to caulculate some matrix A with the desired eigenvalue set:

#input = eig

def gen_eigenvector(input):
# gen_eigenvector is a helper function to generate a matrix corresponding to the input list of eigenvalues
# it does so using a randomly generated matrix P and the identity A = PDP^-1, where D = X %% I
    D = np.diag(input)
    P = np.random.rand(len(input), len(input))
    A = np.dot(np.dot(P, D), np.linalg.inv(P))
    print(A)
    print(linalg.eig(A))
    return A

#gen_eigenvector(eig)

def gen_constrained_matrix(n, u, v):
    # this function creates a square matrix 'n' with a rank of u + v, and count sums of positive/gative eigenvalues of u and v respectively
    # check rank constraint of inputs
    if u + v  > n:
        raise ValueError("excessive eigenvalue count")

    # generate & concat random eigenvalues [integer, max val. 10)
    eig_val_pos = np.random.uniform(1, 10, u)
    eig_val_neg = -np.random.uniform(1, 10, v)
    eig_val_zero = np.zeros(n - u - v)
    eig_vals = np.concatenate([eig_val_pos, eig_val_neg, eig_val_zero])

    while True:
        P = np.random.rand(n, n)
        if np.linalg.matrix_rank(P) == n:
            break

    D = np.diag(eig_vals)
    constrained_matrix = P @ D @ np.linalg.inv(P)

    return constrained_matrix

def gen_real_constrained_matrix(n, u, v, tol = 1e-5):
    # added optional tolerance threshold
    if u + v  > n:
        raise ValueError("excessive eigenvalue count")
    # sampling changed to half normal pos/neg
    eig_val_pos = np.abs(np.random.normal(0, 10, u))
    eig_val_neg = -np.abs(np.random.uniform(0, 10, v))
    eig_val_zero = np.zeros(n - u - v)
    eig_vals = np.concatenate([eig_val_pos, eig_val_neg, eig_val_zero])

    # potential infinite loop of qr decomps to get sufficient rank. probably need to fix
    while True:
        P, _ = np.linalg.qr(np.random.rand(n, n))
        if np.linalg.matrix_rank(P) == n:
            break

    D = np.diag(eig_vals)

    constrained_matrix = P @ D @ P.T

    if constrained_matrix.shape != (n, n):
        raise ValueError("output matrix dimension error")

    # check using linalg eig functions, count from list of eigenvalues with tolerance for imprecise 0 values
    # np.linalg.eigvalsh()
    output_eigenvalues = np.linalg.eigvalsh(constrained_matrix)

    # tolerance tol
    count_list = [np.sum(output_eigenvalues < -tol),
        np.sum(output_eigenvalues > tol),
        np.sum(np.isclose(output_eigenvalues, 0, atol = tol))
    ]

    # check count]
    if count_list[1] != u or count_list[0] != v or count_list[2] != (n - u - v):
        raise ValueError(
            "eigenvalue count error:\n"
            f"exp: {u}, {v}, {n - u - v}\n"
            f"actual:{count_list[1]}, {count_list[0]}, {count_list[2]}")
    else:
        print("working")

    return constrained_matrix

#matrix = gen_real_constrained_matrix(3, 2, 1)

def matrix_checks(matrix):
    #checks eigenvalue characteristics of user input matrix, returns summary
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    eig_val_pos = np.sum(eigenvalues >0)
    eig_val_neg = np.sum(eigenvalues < 00)
    eig_val_zero = np.sum(np.isclose(eigenvalues,0))
    eig_val_complex = np.sum(np.iscomplex(eigenvalues))

    positive_semidefinite = np.all(eigenvalues >= 0)
    invertible = not np.isclose(np.linalg.det(matrix), 0)

    return {
        "eigenvalues": np.round(eigenvalues, 2),
        "eigenvectors": np.round(eigenvectors, 2),
        "positive_eigenvalue_count": eig_val_pos,
        "negative_eigenvalue_count": eig_val_neg,
        "zero_eigenvalue_count": eig_val_zero,
        "complex_eigenvalue_count": eig_val_complex,
        "is_positive_semidefinite": positive_semidefinite,
        "is_invertible": invertible
    }


def matrix_checks_and_zero_negative(matrix):
    check_results = matrix_checks(matrix)

    eigenvalues = check_results["eigenvalues"]
    eigenvectors = check_results["eigenvectors"]
    eig_val_neg = check_results["negative_eigenvalue_count"]

    if eig_val_neg > 0:
        negative_eigenvalue_index = np.where(eigenvalues < 0)[0][0]
        negative_eigenvalue = eigenvalues[negative_eigenvalue_index]
        v1 = eigenvectors[:, negative_eigenvalue_index]
        u = np.ones(len(matrix))
        alpha = -negative_eigenvalue / np.dot(u, v1)
        matrix_prime = matrix + alpha * np.outer(u, v1)
        modified_eigenvalues, _ = np.linalg.eig(matrix_prime)

        return {
            "eigenvalue_summary": check_results,
            "modified_matrix": np.round(matrix_prime, 2),
            "modified_eigenvalues": np.round(modified_eigenvalues, 2)
        }

    return {
        "eigenvalue_summary": check_results,
        "modified_matrix": None,
        "modified_eigenvalues": None
    }

test_matrix = np.array([[4, 2, 0],
                        [2, -3, 0],
                        [0, 0, 1]])

result = matrix_checks_and_zero_negative(test_matrix)
print(result)


#print("matrix:\n", matrix)
#print("eigenvalues:", np.linalg.eigvals(matrix))

#test file
#fp = "C:\\Users\\JayWood\\Documents\\Data\\test_inputs.txt"

'''with open(fp, "r") as f:
    for line in f:
        n, u, v = map(int, line.split())
        print(f"\nn={n}, u={u}, v={v}")
        try:
            matrix = gen_real_constrained_matrix(n, u, v)
            #print(matrix)
            #print("eigval", np.linalg.eigvals(matrix))
        except ValueError as e:
            print("Error:", e)'''
