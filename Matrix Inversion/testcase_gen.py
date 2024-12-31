import numpy as np

def generate_non_singular_matrix(n, decimals=2):
    matrix = np.random.uniform(low=-10.0, high=10.0, size=(n, n))
    matrix = np.round(matrix, decimals)
    matrix += np.eye(n) * np.random.uniform(0.1, 1.0)
    return matrix

def save_matrix_to_file(filename, matrix):
    with open(filename, 'w') as f:
        f.write(f"{matrix.shape[0]}\n")
        np.savetxt(f, matrix, fmt=f'%.{2}f')

if __name__ == "__main__":
    n = 1000
    matrix = generate_non_singular_matrix(n)
    save_matrix_to_file("inp_file.txt", matrix)
    print(f"{n}x{n} non-singular matrix saved.")
