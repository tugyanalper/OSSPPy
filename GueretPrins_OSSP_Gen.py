import numpy as np


def generator(n, m):
    K = 1000
    P = np.zeros(shape=(n, m))
    for i in range(m):
        for j in range(n):
            P[i, j] = K // m
        P[i, i] += K % m
    return P


def main():
    P = generator(3, 3)
    print(P)


if __name__ == '__main__':
    main()
