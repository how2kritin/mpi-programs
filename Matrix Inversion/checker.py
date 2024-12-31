import numpy as np

def load_matrix_from_file(filename):
    with open(filename, 'r') as f:
        n = int(f.readline().strip())
        matrix = np.loadtxt(f)
    return matrix

def invert_matrix(matrix):
    return np.linalg.inv(matrix)

def save_matrix_to_file(filename, matrix):
    np.savetxt(filename, matrix, fmt=f'%.{2}f')

if __name__ == "__main__":
    input_filename = "inp_file.txt"
    output_filename = "inverted_matrix.txt"

    matrix = load_matrix_from_file(input_filename)
    inverted_matrix = invert_matrix(matrix)
    save_matrix_to_file(output_filename, inverted_matrix)

    print(f"Inverted matrix saved to '{output_filename}'.")
