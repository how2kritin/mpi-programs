import random


def save_large_test_case(filename, n, lower_bound=-1000.0, upper_bound=1000.0):
    with open(filename, 'w') as f:
        f.write(f"{n}\n")
        for _ in range(int(n)):
            value = round(random.uniform(lower_bound, upper_bound), 2)
            f.write(f"{value} ")


if __name__ == "__main__":
    n = int(1e6)
    save_large_test_case("inp_file.txt", n)
    print(f"Test case with {int(n)} values saved.")
