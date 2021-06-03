import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
import os


def get_cmap(n, name='Dark2_r'):
    return plt.cm.get_cmap(name, n)


users = ['VI_BI_1080TI', 'MediaA', 'HA', 'VI_IPS_1080TI', 'VI_OP_1080TI',
       'VI_Face_1080TI', 'MIA', 'AD', 'Test', 'ha_det', 'SenseVideo5',
       'VI_AIC_1080TI', 'VI_MODEL_1080TI', 'HA-HAND', 'SenseMap']

users = ['VI_BI_1080TI', 'VI_OP_1080TI', 'VI_IPS_1080TI', 'VI_Face_1080TI', 'VI_MODEL_1080TI', 'MIA']


def print_user_share(filename):
    df = pd.read_csv(filename)
    cmap = get_cmap(len(users))
    for i, user in enumerate(users):
        x_name, y_name = user + '-used_quota', user + '-total_quota'
        y_df = df[x_name].cumsum() / df[y_name].cumsum()
        y_list = y_df.to_list()
        # plt.plot(y_df.index.to_list(), y_list, label='%s %s' % ("user", user), color=cmap(i))
        plt.plot(y_df.index.to_list(), df[user + '-fairness'], label='%s %s' % ("user", user), color=cmap(i), linestyle='--')
    # plt.plot(df.index.to_list(), df['occupied_gpu'] / df['total_gpu'], label='%s' % "pending_gpu", color=cmap(len(users) + 1), linestyle='-')

    plt.legend(loc='lower right')
    plt.xlabel('user accumulated share')
    plt.xlim(8640 * 0, 8640 * 12)
    # plt.xlim(20000, 30000)
    plt.ylim(0, 2)
    plt.ylabel('utilize/weight')
    plt.grid(alpha=.3, linestyle='--')
    plt.savefig('user_share.png', dpi=300)
    plt.close('all')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_files', nargs='+', type=str, help='input file path')
    parser.add_argument('--output_prefix', default='result', type=str, help='output file prefix')
    parser.add_argument('--label_name', default='label', type=str)
    parser.add_argument('--last_k', type=int, default=1, help='use last k th "-%s-" pattern as label value')
    args = parser.parse_args()
    if not args.input_files:
        args.input_files = ['de94e51-lease-lease-consolidate-job_reward-fair-900-metrics.csv']
        args.input_files = ['de94e51-yarn-lease-consolidate-fifo-static-10000000-metrics.csv']
    # args.input_files.sort(key=lambda e: int(get_value_from_filename(e)))
    print(args)
    print_user_share(args.input_files[0])


if __name__ == "__main__":
    main()
