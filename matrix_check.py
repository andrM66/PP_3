import numpy as np

PATH = ["matrix_list_100.txt", "matrix_list200.txt", "matrix_list400.txt", "matrix_list1000.txt"]
SIZES = [100, 200, 400, 1000]
RES_PATH = ["result100.txt", "result200.txt", "result400.txt", "result1000.txt"]


def read_matrixes(path, size):
    arr = []
    f = open(path, "r")
    tmp = []
    lines = f.readlines()
    i = 1
    for line in lines:
        tmp.append(list(map(int, line.rstrip().split())))
        if i % size == 0:
            tmp = np.array((tmp))
            arr.append(tmp)
            tmp = []
        i += 1
    f.close()
    return arr


def compare_matrix(matrix1: np.ndarray, matrix2: np.ndarray, size: int):
    for i in range(size):
        for j in range(size):
            if matrix1[i][j] != matrix2[i][j]:
                return False
    return True


if __name__ == "__main__":
    check_flag = True
    for i in range(len(PATH)):
        arr = read_matrixes(PATH[i], SIZES[i])
        arr_res = read_matrixes(RES_PATH[i], SIZES[i])
        res_check = []
        for j in range(len(arr) - 1):
            res_check.append(arr[j].dot(arr[j + 1]))
        for j in range(len(res_check)):
            if not compare_matrix(res_check[j], arr_res[j], SIZES[i]):
                check_flag = False
    if check_flag:
        print("All checks passed")
    else:
        print("Not all passed")
