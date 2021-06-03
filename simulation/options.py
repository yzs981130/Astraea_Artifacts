import argparse
import ast

class Options:
    def __init__(self, ):
        parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
        parser.add_argument('--placement', default='random', choices=['random', 'consolidate', 'gandiva', 'local_search'], type=str, help='placement policy')
        parser.add_argument('--schedule', default='time-aware', type=str, choices=['fifo', 'dlas', 'gittins', 'gandiva', 'time-aware', 'time-aware-with-lease', 'lease', 'fairness'], help='schedue policy')
        parser.add_argument('--profile', default=False, type=ast.literal_eval, help='trace path')
        parser.add_argument('--num_switch', default=4, type=int, help='number of switch')
        parser.add_argument('--num_node_p_switch', default=8, type=int, help='number of num_node_p_switch')
        parser.add_argument('--num_gpu_p_node', default=8, type=int, help='number of num_gpu_p_node')
        parser.add_argument('--num_cpu_p_node', default=64, type=int, help='number of num_cpu_p_node')
        parser.add_argument('--mem_p_node', default=256, type=int, help='number of mem_p_node')
        parser.add_argument('--check_time_interval', default=10, type=int, help='number of mem_p_node')
        parser.add_argument('--trace', default='data/fake_job.csv', type=str, help='trace path')
        parser.add_argument('--lease_term_interval', default=30*60, type=int, help='lease specific: lease term interval')
        parser.add_argument('--disc_priority_k', default=5, type=int,
                            help='lease specific: how many discrete priority class')
        parser.add_argument('--stavation_threshold', default=4*3600, type=int, help='stavation time threshold (second)')
        parser.add_argument('--delay_threshold', default=0, type=int, help='delay time threshold (second)')
        parser.add_argument('--job_selection', default='random',
                            choices=['random', 'fifo', 'smallestfirst', '2das', 'shortestremainfirst', 'fairness', 'disc_fairness', 'job_reward', 'themis'],
                            type=str, help='job selection policy')
        parser.add_argument('--user_quota', default='fair', choices=['dynamic', 'static', 'fair'], type=str, help='user quota policy')
        parser.add_argument('--block_scheduling', default=0, choices=[0, 1], type=int, help='block scheduling')
        parser.add_argument('--dynamic_quota_ratio', default=1.2, type=float, help='overcommit ratio of dynamic quota policy')
        parser.add_argument('--log_level', default='INFO', choices=['INFO', 'DEBUG', 'ERROR'],
                            type=str, help='log level')
        parser.add_argument('--share', default='data/share.csv', type=str, help='user share path')
        parser.add_argument('--name_list', default='data/name.lst', type=str, help='user name list')
        parser.add_argument('--fairness_output', default='fairness.csv', type=str, help='fairness.csv')
        parser.add_argument('--numgpu_fallback_threshold', default=5, type=int, help='numgpu fallback length threshold')
        parser.add_argument('--dist_trace_path', default='', type=str, help='distribution file path')
        parser.add_argument('--metrics', default=False, type=bool, help="log metrics")
        parser.add_argument('--metrics_path', default='metrics.csv', type=str, help="log metrics destination")
        parser.add_argument('--skip_big_job_lease', default=False, type=bool, help="set big job lease term to infinite")
        self.args = parser.parse_args()
    
    def init(self, ):
        return self.args

Singleton = Options()


_allowed_symbols = [
    'Singleton'
]
