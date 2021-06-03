import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
import os


def get_cmap(n, name='Dark2_r'):
    return plt.cm.get_cmap(name, n)


def get_cdf(data):
    sorted_data = sorted(data)
    p = 100. * np.arange(len(sorted_data)) / (len(sorted_data) - 1)
    return sorted_data, p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, help='input file path')
    parser.add_argument('--output_prefix', default='result', type=str, help='output file prefix')
    parser.add_argument('--label_name', default='label', type=str)
    args = parser.parse_args()
    print(args)

    df = pd.read_csv(args.input_file)
    pending_overhead_list = df[args.label_name].to_list()
    x, y = get_cdf(pending_overhead_list)
    plt.plot(x, y, color='red')

    # plt.legend(loc='lower right')
    plt.xlabel('GPU time: utilized/deserved')
    plt.xlim(0, 5000)
    plt.ylim(0, 100)
    plt.ylabel('CDF')
    plt.grid(alpha=.3, linestyle='--')
    # plt.savefig(output_prefix + '-fairness.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
