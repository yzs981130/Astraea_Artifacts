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


def get_value_from_filename(name: str) -> str:
    return name[name.rfind('-', 0, name.rfind('-') - 1) + 1:name.rfind('-')]


def get_value_from_kth(name: str, k: int) -> str:
    pos1 = pos2 = len(name)
    cnt = 0
    for i in range(len(name)-1, -1, -1):
        if name[i] == '-':
            cnt += 1
            if cnt == k:
                pos2 = i
            elif cnt == k+1:
                pos1 = i
                break
    return name[pos1 + 1:pos2]


def plot_pending_overhead(input_files, output_prefix, label, k):
    cmap = get_cmap(len(input_files))
    for i, file in enumerate(input_files):
        df = pd.read_csv(file)
        pending_overhead_list = df['pending_overhead'].to_list()
        x, y = get_cdf(pending_overhead_list)
        plt.plot(x, y, label='%s %s' % (label, get_value_from_kth(file, k)), color=cmap(i))
    plt.legend(loc='lower right')
    plt.xlabel('(RunningTime+PendingTime)/RunningTime')
    plt.xlim(1, 25)
    plt.ylim(0, 100)
    plt.ylabel('CDF')
    plt.grid(alpha=.3, linestyle='--')
    plt.savefig(output_prefix + '-pending_overhead.png', dpi=300)
    plt.close('all')


def plot_fairness(input_files, output_prefix, label, k):
    cmap = get_cmap(len(input_files))
    for i, file in enumerate(input_files):
        df = pd.read_csv(file)
        pending_overhead_list = df['fairness'].to_list()
        x, y = get_cdf(pending_overhead_list)
        plt.plot(x, y, label='%s %s' % (label, get_value_from_kth(file, k)), color=cmap(i))
    plt.legend(loc='lower right')
    plt.xlabel('GPU time: utilized/deserved')
    plt.xlim(0, 8)
    plt.ylim(0, 100)
    plt.ylabel('CDF')
    plt.grid(alpha=.3, linestyle='--')
    plt.savefig(output_prefix + '-fairness.png', dpi=300)
    plt.close('all')


def print_average_jct(input_files, output_prefix, k):
    JCT = list()
    index = list()
    for i, file in enumerate(input_files):
        df = pd.read_csv(file)
        aJCT = df['total_life_time'].mean()
        JCT.append(aJCT)
        index.append(get_value_from_kth(file, k))
    df_new = pd.DataFrame(JCT, index = index,columns=['JCT'])
    fig, ax = plt.subplots()
    p1 = df_new['JCT'].plot(kind="bar", colormap='Dark2', title="Average Job Complete Time", ax=ax)
    plt.savefig(output_prefix + '-average-JCT.png', dpi=300)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_files', nargs='+', type=str, help='input file path')
    parser.add_argument('--output_prefix', default='result', type=str, help='output file prefix')
    parser.add_argument('--label_name', default='label', type=str)
    parser.add_argument('--last_k', type=int, default=1, help='use last k th "-%s-" pattern as label value')
    args = parser.parse_args()
    if not args.input_files:
        args.input_files = ['fairness.csv']
    # args.input_files.sort(key=lambda e: int(get_value_from_filename(e)))
    print(args)

    plot_pending_overhead(args.input_files, args.output_prefix, args.label_name, args.last_k)
    plot_fairness(args.input_files, args.output_prefix, args.label_name, args.last_k)

    print_average_jct(args.input_files, args.output_prefix, args.last_k)


if __name__ == "__main__":
    main()
