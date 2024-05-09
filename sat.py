import scipy.stats as sts
import numpy as np
import matplotlib.pyplot as plt
import statistics as st


DATA_PATH = ["time_res100.txt", "time_res200.txt", "time_res400.txt", "time_res1000.txt"]
GAMMA = 0.9
FIG_PATH = "figure.png"
STAT_PATH = ["statistic100.txt","statistic200.txt", "statistic400.txt", "statistic1000.txt"]
SIZES = [100, 200, 400, 1000]


def find_expectation_inter_no_dis(sample: np.ndarray, gamma: float) -> tuple:
    t = sts.t.ppf(gamma/2 + 0.5, len(sample) - 1)
    s = np.sqrt(st.pvariance(sample))
    delta = t * s / np.sqrt(len(sample))
    x_l = sample.mean() - delta
    x_r = sample.mean() + delta
    return x_l, x_r


def read_time(path):
    arr = []
    f = open(path, "r")
    line = f.readline()
    arr.append(list(map(float, line.rstrip().split())))
    arr = np.array(arr[0])
    f.close()
    return arr


def make_statistic(array: np.ndarray, path: str):
    f = open(path, "w")
    f.write(f'математическое ожидание: {array.mean()}\n')
    f.write(f'интервальная оценка при gamma = {GAMMA}: ({find_expectation_inter_no_dis(array, GAMMA)})\n')
    f.write(f'Количество операций: {array.size}\n')
    f.close()


def plot_dependence_time_of_size(mean_arr, size_array, path):
    fig = plt.figure()
    plt.plot(size_array, mean_arr)
    plt.grid(True)
    plt.title("Зависимость времени от размера квадратной матрицы, при 6 ядрах")
    plt.xlabel("Размер матрицы")
    plt.ylabel("Время одной операции")
    fig.savefig(path)


if __name__ == "__main__":
    mean_arr = []
    for i in range(len(DATA_PATH)):
        arr = read_time(DATA_PATH[i])
        mean_arr.append(arr.mean())
        make_statistic(arr, STAT_PATH[i])
    plot_dependence_time_of_size(mean_arr, SIZES, FIG_PATH)