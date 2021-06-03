
import bisect
import os, sys, random

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.basename(__file__) + os.sep + '..' + os.sep + '..')
from .base import BaseScheduler

CHECKPOINT_OVERHEAD = 5
RESUME_OVERHEAD = 20

class Vividict(dict):
    def __missing__(self, key):
        value = self[key] = type(self)()
        return value


def is_job_finished(job):
    return job['total_executed_time'] >= job['duration']


def is_job_lease_expires(job, lease_term):
    return job['executed_time_in_last_lease'] >= lease_term


def get_job_progress(job):
    return 1. * job['total_executed_time'] / job['duration']


def get_job_pending_overhead(job):
    if job['total_executed_time'] == 0:
        return 1
    return 1. * (job['total_pending_time'] + job['total_executed_time']) / job['total_executed_time']


def get_job_weighted_share(job, jobs):
    if len(jobs) == 0:
        return float('inf')
    if 'weight' not in job:
        return 1. / len(jobs)
    return 1. * job['weight'] / sum(job['weight'] for job in jobs)
    # return 1. * job['num_gpu'] / sum(job['num_gpu'] for job in jobs)


class LeaseScheduler(BaseScheduler):
    def __init__(self, JOBS, CLUSTER, placement, name, logger, **kwargs):
        super(LeaseScheduler, self).__init__(JOBS=JOBS, CLUSTER=CLUSTER, placement=placement, name=name, logger=logger)
        assert self.name == 'lease'
        self.pending_jobs = JOBS.pending_jobs
        self.running_jobs = JOBS.running_jobs
        self.event_jobs = JOBS.job_events
        self.end_jobs = list()
        self.check_time_interval = kwargs.get('check_time_interval', 10)

        # custom
        self.lease_term_interval = kwargs.get('lease_term_interval', 30 * 60)
        self.replacement = kwargs.get('replacement', False)
        # self.last_round_running_job_cnt = 0
        self.gpunum_job_dist = {}
        self.gpunum_job_dist_tlb = Vividict()
        self.disc_priority_k = kwargs.get('disc_priority_k', 5)

        self.job_selection_str = kwargs.get('job_selection', 'random')
        self.user_quota_policy = kwargs.get('user_quota', 'fair')
        self.dynamic_quota_ratio = kwargs.get('dynamic_quota_ratio', 1.5)
        self.stavation_threshold = kwargs.get('stavation_threshold', 4*3600)
        self.block_scheduling = kwargs.get('block_scheduling', 0)
        self.delay_threshold = kwargs.get('delay_threshold', 0)

        self.user_share = {}
        self.user_deserved = {}
        self.user_utilized = {}
        self.user_fairness = {}
        self.user_deserved_temp = {}
        self.user_deserved_history = {}
        self.user_utilized_temp = {}
        self.user_utilized_history = {}
        self.user_weighted = {}
        self.user_weighted_history = {}
        self.user_weighted_temp = {}
        self.user_U_divide_W = {}
        self.user_quota = {}
        self.user_used_quota = {}
        self.user_resource_ratio = {}
        self.user_stave_jobs = {}
        self.user_delay_jobs = {}

        self.name2user_dict = {}
        self.USERS = kwargs.get('USERS')

        self.share_path = kwargs.get('share', 'data/share.csv')

        self.fairness_output_path = kwargs.get('fairness_output', 'fairness.csv')

        self.numgpu_fallback_threshold = kwargs.get('numgpu_fallback_threshold', 5)
        self.job_dist = []
        self.job_dist_tlb = {}

        self.dist_trace_path = kwargs.get('dist_trace_path')

        self.metrics = kwargs.get('metrics')
        self.metrics_path = kwargs.get('metrics_path')

        self.job_has_resume_overhead = {}

        self.skip_big_job_lease = kwargs.get('skip_big_job_lease')

        self.long_term_interval_cnt = kwargs.get('long_term_interval_cnt', 24)

    # abstract
    def check_resource(self, **kwargs):
        resource_name = kwargs.get('resource', 'gpu')
        assert isinstance(resource_name, str)
        if 'gpu' == resource_name:
            return self.CLUSTER.check_free_gpus()
        if 'cpu' == resource_name:
            return self.CLUSTER.check_free_cpus()

        raise NotImplementedError

    def remove_by_status(self, job, status):
        user = job['user']
        if status == 'EVENT':
            # self.event_jobs.remove(job)
            pass
        elif status == 'PENDING':
            self.pending_jobs.remove(job)
            user.pending_jobs.remove(job)
        elif status == 'RUNNING':
            self.running_jobs.remove(job)
            user.running_jobs.remove(job)
        elif status == 'END':
            self.running_jobs.remove(job)
            user.running_jobs.remove(job)

        else:
            raise NotImplementedError

    def add_by_status(self, job, status):
        user = job['user']
        if status == 'EVENT':
            self.event_jobs.append(job)
        elif status == 'PENDING':
            self.pending_jobs.append(job)
            user.pending_jobs.append(job)
        elif status == 'RUNNING':
            self.running_jobs.append(job)
            user.running_jobs.append(job)
        elif status == 'END':
            self.end_jobs.append(job)
        else:
            raise NotImplementedError

    def switch_job_status(self, job, prev=None, cur=None, cur_time=None):
        assert prev is not None and cur is not None
        assert job['status'] == prev
        self.remove_by_status(job, prev)
        self.add_by_status(job, cur)
        job['status'] = cur

    def finish_all_jobs(self):
        return len(self.event_jobs) + len(self.pending_jobs) + len(self.running_jobs) == 0

    def place_jobs(self, jobs):
        for job in jobs:
            self.placement(job)

    def release_job_resource(self, job, status='UNKNOWN'):
        if self.placement.name == 'gandiva':
            ret = self.CLUSTER.release_gandiva_job_resource(job, status)
        else:
            ret = self.CLUSTER.release_job_resource(job, status)
        return ret

    def try_allocate_resource(self, job):
        if self.placement.name in ['random', 'consolidate', 'gandiva', 'local_search']:
            ret = self.placement.place_jobs(job)
        else:
            raise NotImplementedError
        return ret

    # add time_diff to pending jobs for last term pending
    # total_life_time for calculating deserved service, record since arrival
    def update_pending_jobs(self, cur_time):
        for user in self.USERS:
             self.user_stave_jobs[user] = list()

        for job in self.pending_jobs:
            time_diff = int(cur_time - job['last_check_time'])
            job['total_pending_time'] += time_diff
            job['last_pending_time'] += time_diff
            job['total_life_time'] += time_diff
            job['last_check_time'] = cur_time
            # add job to starvation queue if it staves, otherwise remove it from the queue
            if (job['last_pending_time'] >= self.stavation_threshold):
                self.user_stave_jobs[job['user']].append(job)

        for user in self.USERS:
             self.user_stave_jobs[user].sort(key=lambda e: e.__getitem__('submit_time'))

    # finish job; release resource; update metrics
    def finish_job(self, job):
        assert self.release_job_resource(job) == True
        self.switch_job_status(job, prev="RUNNING", cur="END")
        #TODO: the fairness value may be not accurate, add scheduling fairnees to address this issue
        job['utilized'] = job['num_gpu'] * job['total_executed_time'] 
        job['fairness'] = job['utilized'] / job['deserved']
        job['disc_fairness'] = self.fairness2disc(job['fairness'])

        job['end_time'] = job['last_check_time'] + job['duration'] - job['total_executed_time']
        job['total_executed_time'] = job['duration']
        if job['job_id'] in self.job_has_resume_overhead:
            if self.job_has_resume_overhead[job['job_id']]:
                job['resume_cnt'] += 1
            self.job_has_resume_overhead[job['job_id']] = False
        self.logger.info('complete job {} at time {}'.format(job['job_id'], job['end_time']))

    # job lease expires; change job status to pending; update job metrics
    def lease_expire_job(self, job, cur_time):
        # reclaim resource
        assert self.release_job_resource(job) == True
        # change job status to pending
        self.switch_job_status(job, prev="RUNNING", cur="PENDING")
        # update lease term related metrics
        job['lease_expiry_cnt'] = job['lease_expiry_cnt'] + 1
        job['last_pending_time'] = 0
        job['last_checkpoint_time'] = cur_time
        job['checkpoint_cnt'] += 1

        if job['job_id'] in self.job_has_resume_overhead:
            if self.job_has_resume_overhead[job['job_id']]:
                job['resume_cnt'] += 1
            self.job_has_resume_overhead[job['job_id']] = False

        self.logger.info('job {} lease expires'.format(job['job_id']))

    def update_running_jobs(self, cur_time):
        for job in self.running_jobs:
            # update all running jobs
            time_diff = int(cur_time - job['last_check_time'])
            job['total_executed_time'] += time_diff
            job['total_life_time'] += time_diff
            job['last_check_time'] = cur_time
            job['executed_time_in_last_lease'] += time_diff

    def check_terminated_jobs(self, cur_time):
        for job in self.running_jobs:
            # if running->finished: finish job; release job resource
            if is_job_finished(job):
                self.finish_job(job)
            # if running->expiry: change job status to pending
            elif is_job_lease_expires(job, self.lease_term_interval):
                if self.skip_big_job_lease and job['num_gpu'] > 32:
                    continue
                self.lease_expire_job(job, cur_time)

    def update_job_fairness(self):
        # deserved: min {share, demand}
        for job in self.pending_jobs + self.running_jobs:
            weighted_share = get_job_weighted_share(job, job['user'].pending_jobs + job['user'].running_jobs)
            share = weighted_share * self.user_quota[job['user']]
            job['deserved'] += min(job['num_gpu'], share) * self.check_time_interval
            # add computing time gained in last lease term if running
            if job['status'] == "RUNNING":
                job['utilized'] = job['num_gpu'] * job['total_executed_time']
            # calculate job fairness index
            job['fairness'] = job['utilized'] / job['deserved']
            # calculate job disc_fairness
            job['disc_fairness'] = self.fairness2disc(job['fairness'])

    def is_delay_job(self, job):
        if job['lease_expiry_cnt'] > 0 and job['executed_time_in_last_lease'] < self.delay_threshold:
            return True
        if job['lease_expiry_cnt'] == 0 and job['executed_time_in_last_lease'] >= 120 and \
            job['executed_time_in_last_lease'] < self.delay_threshold:
            return True
        return False

    def check_delay_jobs(self):
        for user in self.USERS:
            self.user_delay_jobs[user] = list()

        for job in self.running_jobs:
            if self.is_delay_job(job):
                job['utilized'] = job['num_gpu'] * (job['total_executed_time'] - job['executed_time_in_last_lease'])
                job['fairness'] = job['utilized'] / job['deserved']
                job['disc_fairness'] = self.fairness2disc(job['fairness'])
                self.user_delay_jobs[job['user']].append(job)

    # update visible jobs
    # change all visible jobs status from EVENT to PENDING
    def dispatch_new_job(self, prev_time, cur_time):
        event_list = list()
        for event in self.event_jobs:
            if event['time'] <= cur_time:
                assert event['time'] >= prev_time
                event_list.append(event)

        for event in event_list:
            for start_job in event['start_jobs']:
                #pass jobs without valid user share
                if start_job['user'] not in self.user_share:
                    self.JOBS.job_list.remove(start_job)
                    continue
                if start_job['num_gpu'] > self.user_quota[start_job['user']]:
                    self.JOBS.job_list.remove(start_job)
                    continue

                # set initial job metrics
                self.switch_job_status(start_job, prev='EVENT', cur='PENDING', cur_time=cur_time)
                start_job['last_check_time'] = cur_time
                start_job['pending_time'] = cur_time - start_job['submit_time']

                start_job['first_seen_time'] = cur_time

                # set fairness
                start_job['fairness'] = 0
                start_job['disc_fairness'] = self.fairness2disc(0)
                # lease related metrics
                start_job['lease_expiry_cnt'] = 0
                start_job['total_pending_time'] = start_job['pending_time']
                start_job['total_life_time'] = start_job['pending_time']
                start_job['last_pending_time'] = start_job['pending_time']
                # TODO: dynamic lease term
                start_job['last_lease_term_interval'] = self.lease_term_interval
                # calculate fairness by
                start_job['deserved'] = 0
                start_job['utilized'] = 0
                # set checkpoint and resume cnt
                start_job['checkpoint_cnt'] = 0
                start_job['resume_cnt'] = 0
                start_job['last_checkpoint_time'] = cur_time

            self.event_jobs.remove(event)

    def job_selection(self, user):
        pending_jobs = user.pending_jobs.copy()
        delay_jobs = self.user_delay_jobs[user].copy()
        empty_jobs = list()
        
        # remove starvation jobs in job selection, since such jobs in a seperate queue
        for u in self.USERS:
            for j in self.user_stave_jobs[u]:
                if (j in pending_jobs):
                    pending_jobs.remove(j)            
                    
        if self.job_selection_str == 'random':
            return random.sample(pending_jobs, k=len(pending_jobs)), empty_jobs

        if self.job_selection_str == 'fifo':
            pending_jobs.sort(key=lambda e: e.__getitem__('submit_time'))
            return pending_jobs, empty_jobs

        if self.job_selection_str == 'smallestfirst':
            pending_jobs.sort(key=lambda \
                e: (e.__getitem__('num_gpu'),
                    e.__getitem__('submit_time')))
            return pending_jobs, empty_jobs

        if self.job_selection_str == 'shortestremainfirst':
            pending_jobs.sort(key=lambda \
                e: (e.__getitem__('duration') - e.__getitem__('total_executed_time'),
                    e.__getitem__('submit_time')))
            return pending_jobs, empty_jobs

        if self.job_selection_str == '2das':
            pending_jobs.sort(key=lambda \
                e: (e.__getitem__('total_executed_time'),
                    e.__getitem__('submit_time')))
            return pending_jobs, empty_jobs

        if self.job_selection_str == 'fairness':
            pending_jobs.sort(key=lambda \
                e: (e.__getitem__('fairness'), \
                    e.__getitem__('submit_time')))
            delay_jobs.sort(key=lambda \
                e: (e.__getitem__('fairness') * -1., \
                    e.__getitem__('submit_time') * -1.))
            return pending_jobs, delay_jobs

        if self.job_selection_str == 'disc_fairness':
            pending_jobs.sort(key=lambda \
                e: (e.__getitem__('disc_fairness'), \
                    e.__getitem__('submit_time')))
            delay_jobs.sort(key=lambda \
                e: (e.__getitem__('disc_fairness') * -1., \
                    e.__getitem__('submit_time') * -1.))
            return pending_jobs, delay_jobs

        #To Do: Should we calc job reward based on user
        if self.job_selection_str == 'job_reward':
            for job in pending_jobs:
                job['job_reward'] = float(self.get_job_reward(job))
            pending_jobs.sort(key=lambda \
                e: (e.__getitem__('disc_fairness'), \
                    e.__getitem__('job_reward'), \
                    e.__getitem__('submit_time')))
            delay_jobs.sort(key=lambda \
                e: (e.__getitem__('disc_fairness') * -1., \
                    e.__getitem__('job_reward') * -1., \
                    e.__getitem__('submit_time') * -1.))
            return pending_jobs, delay_jobs

        if self.job_selection_str == 'themis':
            for job in pending_jobs:
                job['finish_time_fairness'] = float(self.get_finish_time_fairness(job))
            pending_jobs.sort(key=lambda \
                    e: e.__getitem__('finish_time_fairness') * -1)
            delay_jobs.sort(key=lambda \
                    e: e.__getitem__('finish_time_fairness'))
            return pending_jobs, delay_jobs

    def launch_job(self, job, cur_time):
        job['last_check_time'] = cur_time
        job['executed_time_in_last_lease'] = 0
        job['last_pending_time'] = 0
        if job['lease_expiry_cnt'] > 0:
            if not job['last_checkpoint_time'] == cur_time:
                self.job_has_resume_overhead[job['job_id']] = True
        self.switch_job_status(job, prev="PENDING", cur="RUNNING")
        self.logger.info('----job [%d] starts from pending' % job['job_id'])

    def lease_summary(self):
        """
        data_ = pd.DataFrame()
        for job in self.JOBS.job_list:
            data = pd.DataFrame(np.array([job['job_id'], job['pending_time'],
                                job['fairness'], job['disc_fairness'],
                                                       job['lease_expiry_cnt'], job['total_pending_time'],
                                                       job['total_life_time'], job['duration']]), dtype=int)
            data_ = data_.append(data.T)
        data_.columns = ['job_id', 'pending_time', 'fairness', 'disc_fairness', 'lease_expiry_cnt',
                                         'total_pending_time', 'total_life_time', 'duration']
        """
        sharing_benefit_cnt = sharing_loss_cnt = 0
        sharing_fair_cnt = sharing_unfair_cnt = 0
        with open(self.fairness_output_path, 'w+') as f:
            print("job_id,pending_time,fairness,disc_fairness,preempt_cnt,checkpoint_cnt,resume_cnt,"
                  "lease_expiry_cnt,total_pending_time,total_life_time,duration,pending_overhead,first_seen_time,submit_time,end_time,num_gpu", file=f)
            for job in self.JOBS.job_list:
                print("%d,%d,%.4f,%.4f,%d,%d,%d,%d,%d,%d,%d,%.4f,%d,%d,%d,%d" % (job['job_id'], job['pending_time'],
                                                            job['fairness'], job['disc_fairness'],job['preempt'],
                                                            job['checkpoint_cnt'], job['resume_cnt'],
                                                            job['lease_expiry_cnt'], job['total_pending_time'],
                                                            job['total_life_time'], job['duration'],
                                                            get_job_pending_overhead(job), job['first_seen_time'],
                                                                     job['submit_time'], job['end_time'], job['num_gpu']), file=f)
                if ((job['fairness'] <= 1 and job['fairness'] >= 0.95) or \
                    (1/job['fairness'] <= 1 and 1/job['fairness'] >= 0.95)):
                    sharing_fair_cnt += 1
                else:
                    sharing_unfair_cnt += 1
                if (job['fairness'] < 0.95):
                    sharing_loss_cnt += 1
                if (job['fairness'] > 1.05):
                    sharing_benefit_cnt += 1

        print("sharing_fair_cnt: %d/%d, sharing_unfair_cnt: %d/%d" %
              (sharing_fair_cnt, len(self.JOBS.job_list), sharing_unfair_cnt, len(self.JOBS.job_list)))
        print("sharing_loss_cnt: %d/%d, sharing_benefit_cnt: %d/%d" %
              (sharing_loss_cnt, len(self.JOBS.job_list), sharing_benefit_cnt, len(self.JOBS.job_list)))

        for user in self.user_fairness:
            print("user %s, share %.4f" % (user.name, self.user_fairness[user]))

    def get_disc_distribution_index(self, n):
        return n // self.lease_term_interval

    def get_disc_distribution(self, duration_list: list) -> list:
        # get list length
        max_duration = 0
        for duration in duration_list:
            if duration > max_duration:
                max_duration = duration
        max_length = self.get_disc_distribution_index(max_duration) + 1
        result_list = [0] * max_length
        # count
        for duration in duration_list:
            index = self.get_disc_distribution_index(duration)
            result_list[index] += 1
        # prob
        for i in range(max_length):
            result_list[i] = 1. * result_list[i] / len(duration_list)
        return result_list

    def prepare_job_distribution(self):
        # not set, use job trace
        if self.dist_trace_path == '':
            data = pd.DataFrame([[job['num_gpu'], job['duration']] for job in self.JOBS.job_list],
                                columns=['num_gpu', 'duration'])
        else:
            data = pd.read_csv(self.dist_trace_path, usecols=['num_gpu', 'duration'])
        for gpu_num, job_group in data.groupby(['num_gpu']):
            self.gpunum_job_dist[gpu_num] = self.get_disc_distribution(job_group['duration'].tolist())
        self.job_dist = self.get_disc_distribution(data['duration'].tolist())

    def get_lifetime_prob(self, job):
        t1 = job['total_executed_time']
        t2 = t1 + self.lease_term_interval
        # if not exists or less than threshold, use whole job distribution to predict
        if job['num_gpu'] not in self.gpunum_job_dist or \
                len(self.gpunum_job_dist[job['num_gpu']]) < self.numgpu_fallback_threshold:
            job_group = self.job_dist
            tlb = self.job_dist_tlb
        else:
            job_group = self.gpunum_job_dist[job['num_gpu']]
            tlb = self.gpunum_job_dist_tlb[job['num_gpu']]

        i1 = self.get_disc_distribution_index(t1)
        if i1 not in tlb:
            if i1 >= len(job_group):
                return 1
            p1 = sum(job_group[i1:])
            p2 = p1 - job_group[i1]
            p = 1 - p2 / p1
            tlb[i1] = p
        return tlb[i1]

    def absolute_lifetime_prob(self, job):
        t1 = job['total_executed_time']
        return t1 + self.lease_term_interval >= job['duration']

    def fairness2disc(self, x):
        # TODO: add discrete priority
        if x == 0:
            return 0

        f_index = -1. * np.log(x) / np.sqrt(1 + np.log(x) * np.log(x))
        index = self.disc_priority_k - ((f_index + 1) / 2) // (1. / self.disc_priority_k)
        return int(index)

    def get_job_reward(self, job):
        # return 1. * (job['duration'] - job['total_executed_time'])
        return -1. * self.get_lifetime_prob(job) / job['num_gpu']
        # return -1. * self.get_lifetime_prob(job)
        # return 1 - self.absolute_lifetime_prob(job)

    def get_finish_time_fairness(self, job):
        if job['deserved'] == 0:
            return 1
        t_remaining = job['duration'] - job['total_executed_time']
        t_isolated = job['total_executed_time'] * (job['utilized'] / job['deserved']) + (job['total_life_time'] - job['total_executed_time'])
        return job['total_life_time'] / (t_isolated + t_remaining)

    def name2user(self, name):
        if name in self.name2user_dict:
            return self.name2user_dict[name]
        for user in self.USERS.user_list:
            if user.name == name:
                self.name2user_dict[name] = user
                return user
        # TODO: handle never met user
        raise RuntimeError

    def prepare_user_share(self):
        share = pd.read_csv(self.share_path)
        for index, row in share.iterrows():
            self.user_share[self.name2user(row['name'])] = float(row['share'])

        user_share_sum = sum(self.user_share.values())
        total_capacity = self.CLUSTER.check_total_gpus()
        for user in self.USERS:
            ratio = 1. * self.user_share[user] / user_share_sum
            self.user_resource_ratio[user] = ratio
            self.user_quota[user] = int(total_capacity * ratio)
            self.user_used_quota[user] = 0
            self.user_stave_jobs[user] = list()
            self.user_delay_jobs[user] = list()

    def update_user_used_quota(self):
        for user in self.USERS:
            self.user_used_quota[user] = 0.
            for job in user.running_jobs:
                self.user_used_quota[user] += job['num_gpu']

    # TODO: optimization using update for total_job_demand update rather than loop
    def update_user_fairness(self, cur_time):
        for user in self.user_share:
            weighted = self.user_quota[user] * self.check_time_interval
            demand = utilized = 0

            for job in user.running_jobs:
                job_gpu_time = job['num_gpu'] * self.check_time_interval
                utilized += job_gpu_time
                demand += job_gpu_time
            for job in user.pending_jobs:
                demand += job['num_gpu'] * self.check_time_interval

            deserved = min(demand, weighted)
            self.user_utilized_temp[user] += utilized
            self.user_weighted_temp[user] += weighted
            self.user_deserved_temp[user] += deserved

            self.user_utilized[user] = sum(self.user_utilized_history[user][-min(len(self.user_utilized_history[user]), self.long_term_interval_cnt):])
            self.user_weighted[user] = sum(self.user_weighted_history[user][-min(len(self.user_weighted_history[user]), self.long_term_interval_cnt):])
            self.user_deserved[user] = sum(self.user_deserved_history[user][-min(len(self.user_deserved_history[user]), self.long_term_interval_cnt):])

            if self.user_weighted[user] == 0:
                self.user_U_divide_W[user] = 1
            else:
                self.user_U_divide_W[user] = self.user_utilized[user] / self.user_weighted[user]

            if self.user_deserved[user] == 0:
                self.user_fairness[user] = 1
            else:
                self.user_fairness[user] = self.user_utilized[user] / self.user_deserved[user]

    def collect_cluster_gpu_utilization(self) -> dict:
        total, occupied, free = self.CLUSTER.check_total_gpus(), \
                                self.CLUSTER.check_total_gpus() - self.CLUSTER.check_free_gpus(), \
                                self.CLUSTER.check_free_gpus()
        result = {"total_gpu": total, "occupied_gpu": occupied, "free_gpu": free, "utilization": 1. * occupied / total}
        return result

    def collect_user_fairness(self) -> dict:
        keys = [i for i in self.USERS.user_list]
        result = {}
        for key in keys:
            result[key.name + '-fairness'] = self.user_fairness[key] if key in self.user_fairness else ''
        return result

    def collect_user_used_quota(self) -> dict:
        keys = [i for i in self.USERS.user_list]
        result = {}
        for key in keys:
            result[key.name + '-used_quota'] = self.user_used_quota[key] if key in self.user_used_quota else ''
        return result

    def collect_user_total_quota(self) -> dict:
        keys = [i for i in self.USERS.user_list]
        result = {}
        for key in keys:
            result[key.name + '-total_quota'] = self.user_quota[key] if key in self.user_quota else ''
        return result

    def collect_pending_metrics(self) -> dict:
        result = {"pending_job": len(self.pending_jobs), "pending_gpu": sum(job['num_gpu'] for job in self.pending_jobs)}
        return result

    def collect_user_pending_metrics(self) -> dict:
        keys = [i for i in self.USERS.user_list]
        result = {}
        for key in keys:
            result[key.name + '-pending_job'] = len(key.pending_jobs) if key.pending_jobs else 0
            result[key.name + '-pending_gpu'] = sum(job['num_gpu'] for job in key.pending_jobs) if key.pending_jobs else 0
        return result

    def collect_user_request_gpu(self) -> dict:
        keys = [i for i in self.USERS.user_list]
        result = {}
        for key in keys:
            result[key.name + '-request_gpu'] = sum(job['num_gpu'] for job in key.pending_jobs) if key.pending_jobs else 0 + sum(job['num_gpu'] for job in key.running_jobs) if key.running_jobs else 0
        return result

    def collect_runtime_metrics(self, cur_time, header, output='std'):
        if not self.metrics:
            return
        result = {**self.collect_cluster_gpu_utilization(), **self.collect_user_fairness(), **self.collect_user_used_quota(), **self.collect_user_total_quota(), **self.collect_pending_metrics()}
        if output == 'std':
            f = None
        else:
            f = open(output, 'w+' if header else 'a')
        # first met, print header
        if header:
            print("time", end='', file=f)
            for key in result.keys():
                print(',' + key, end='', file=f)
            print(file=f)
        # print value
        print(cur_time, end='', file=f)
        for value in result.values():
            print(',' + str(value), end='', file=f)
        print(file=f)

    def preempt_job(self, job):
        # reclaim resource
        assert self.release_job_resource(job) == True
        #job would lose its progress of current lease due to no checkpoint
        job['total_executed_time'] -= job['executed_time_in_last_lease']
        job['total_pending_time'] += job['executed_time_in_last_lease']
        job['executed_time_in_last_lease'] = 0
        job['preempt'] += 1
        self.job_has_resume_overhead[job['job_id']] = False
        self.user_used_quota[job['user']] -= job['num_gpu']
        # change job status to pending
        self.switch_job_status(job, prev="RUNNING", cur="PENDING")
        self.logger.info('job {} is preempted'.format(job['job_id']))
        return job['num_gpu']

    def preempt_action(self, target_gpu):
        if (not self.user_quota_policy in ['dynamic']):
            return 0

        if (target_gpu <= 0):
            return 0

        preempt_jobs = list()
        running_jobs = self.running_jobs.copy()
        running_jobs.sort(key=lambda e: e.__getitem__('submit_time'))
        used_quota = {}
        for user in self.USERS:
            used_quota[user] = 0
        for j in running_jobs:
            used_quota[j['user']] += j['num_gpu']
            if used_quota[j['user']] > self.user_quota[j['user']]:
                preempt_jobs.append(j)
        if (len(preempt_jobs) == 0):
            return 0

        n = 0
        while (n < target_gpu):
            if (len(preempt_jobs) == 0):
                break
            p_job = preempt_jobs.pop(0)
            n += self.preempt_job(p_job)
        return n

    def backfill_action(self, backfill_job, cur_time, contribute_gpu):
        # backfill is only used for dynamic quota
        if (not self.user_quota_policy in ['dynamic']):
            return
        #shuffle the job list used for backfill
        backfill_job = random.sample(backfill_job, k=len(backfill_job))
        backfill_gpu = 0
        for job in backfill_job:
            user = job['user']
            if backfill_gpu + job['num_gpu'] > contribute_gpu:
                continue
            if (self.dynamic_quota_ratio * self.user_quota[user] > self.user_used_quota[user] + job['num_gpu']):
                if self.try_allocate_resource(job):
                    self.user_used_quota[user] += job['num_gpu']
                    self.launch_job(job, cur_time)
                    backfill_gpu += job['num_gpu']
    
    def quota_check(self, job, quota, used_quota):
        # user static quota is not enough
        if (quota < used_quota + job['num_gpu']):
            return False
        return True

    def allocate_job_resources_with_user(self, cur_time):
        free_gpu = self.CLUSTER.check_free_gpus()
        # store temp utilized/weighted
        user_utilized = self.user_utilized.copy()
        user_weighted = self.user_weighted.copy()
        user_U_divide_W = self.user_U_divide_W.copy()
        user_job_dict  = dict()
        tmp_user_job_dict  = dict()
        user_delay_job = dict()
        user_starve_job_num = dict()
        backfill_job= list()
        user_contribute_gpu = dict()
        user_new_quota = dict()
        user_quota_ratio = dict()

        # update user_weighted in next lease term
        for user in self.USERS:
            # the selected job list is already is copy
            pending_jobs, delay_jobs = self.job_selection(user)
            user_delay_job[user] = delay_jobs
            user_job_dict[user] = self.user_stave_jobs[user] + pending_jobs
            tmp_user_job_dict[user] = user_job_dict[user].copy()
            user_weighted[user] += self.user_quota[user] * self.lease_term_interval
            user_starve_job_num[user] = len(self.user_stave_jobs[user])
            user_contribute_gpu[user] = self.user_quota[user] - self.user_used_quota[user]
            user_new_quota[user] = self.user_used_quota[user]
            user_quota_ratio[user] = user_new_quota[user] / self.user_quota[user]

        # update the user quota
        if self.user_quota_policy in ['fair']:
            while free_gpu > 0 and len(self.pending_jobs) > 0 and len(user_U_divide_W) > 0:
                user_with_least_quota_ratio = min(user_quota_ratio, key=lambda x: user_quota_ratio[x])
                if user_quota_ratio[user_with_least_quota_ratio] < 0.2:
                    # select user with least user quota ratio
                    user = user_with_least_quota_ratio
                else:
                    # select user with least U_divide_W to do max-min
                    user = min(user_U_divide_W, key=lambda x: user_U_divide_W[x])
                user_job_list = tmp_user_job_dict[user]
                # if user has pending job
                while len(user_job_list) > 0:
                    job = user_job_list.pop(0)
                    if free_gpu < job['num_gpu']:
                        continue
                    user_new_quota[user] += job['num_gpu']
                    free_gpu -= job['num_gpu']
                    # update share: use whole lease term (will be normalized anyway)
                    user_utilized[user] += job['num_gpu'] * self.lease_term_interval
                    user_U_divide_W[user] = user_utilized[user] / user_weighted[user]
                    user_quota_ratio[user] = user_new_quota[user] / self.user_quota[user]
                    break
                if len(user_job_list) == 0: 
                    del user_U_divide_W[user]
                    del user_quota_ratio[user]
        else:
            user_new_quota = self.user_quota.copy()
        
        # reset the number of free gpu
        free_gpu = self.CLUSTER.check_free_gpus()

        for user in self.USERS:
            user_job_list = user_job_dict[user]
            while len(user_job_list) > 0:
                # try to allocate one job
                job = user_job_list.pop(0)
                
                if self.user_quota_policy in ['dynamic']:
                    if self.quota_check(job, user_new_quota[user], self.user_used_quota[user]) and free_gpu < job['num_gpu']:
                        # in case user has enough quota but the free gpu num is not enough
                        preempt_gpu = self.preempt_action(job['num_gpu'] - free_gpu)
                        free_gpu += preempt_gpu

                # process newly submitted job
                if self.user_quota_policy in ['fair']:
                    if job['total_executed_time'] == 0 and job['preempt'] == 0:
                        while not self.quota_check(job, user_new_quota[user], self.user_used_quota[user]):
                            if len(user_delay_job[user]) == 0:
                                break
                            p_job = user_delay_job[user].pop(0)
                            free_gpu += self.preempt_job(p_job)
                
                # check the quota 
                if not self.quota_check(job, user_new_quota[user], self.user_used_quota[user]):
                    backfill_job.append(job)
                    if user_starve_job_num[user] > 0 or self.block_scheduling:
                        user_contribute_gpu[user] = 0
                        break
                    else:
                        continue

                # sucessfully place the job
                if not self.try_allocate_resource(job):
                    backfill_job.append(job)
                    # fail to place the job, using block scheduling policy or processing starve jobs
                    if user_starve_job_num[user] > 0 or self.block_scheduling:
                        user_contribute_gpu[user] = 0
                        break
                else:
                    user_starve_job_num[user] = max(0, user_starve_job_num[user] - 1)
                    free_gpu -= job['num_gpu']
                    user_contribute_gpu[user] -= job['num_gpu']
                    self.user_used_quota[user] += job['num_gpu']
                    # launch job, delete job in job_list cache
                    self.launch_job(job, cur_time)

        # avoid backfill is there are any starve jobs
        contribute_gpu = sum(user_contribute_gpu.values())
        self.backfill_action(backfill_job, cur_time, contribute_gpu)

    def run(self, **kwargs):
        cur_time = 0
        # self.prepare_job_distribution()
        self.prepare_user_share()
        metrics_header_flag = True

        while not self.finish_all_jobs():
            if (cur_time % (3600) == 0):
                for user in self.user_share:
                    if user not in self.user_utilized_history:
                        self.user_utilized_history[user] = list()
                        self.user_deserved_history[user] = list()
                        self.user_weighted_history[user] = list()
                    self.user_utilized_history[user].append(self.user_utilized_temp[user] if user in self.user_utilized_temp else 0)
                    self.user_deserved_history[user].append(self.user_deserved_temp[user] if user in self.user_deserved_temp else 0)
                    self.user_weighted_history[user].append(self.user_weighted_temp[user] if user in self.user_weighted_temp else 0)
                    self.user_deserved_temp[user] = 0
                    self.user_utilized_temp[user] = 0
                    self.user_weighted_temp[user] = 0

            self.collect_runtime_metrics(cur_time=cur_time, header=metrics_header_flag, output=self.metrics_path)
            metrics_header_flag = False

            prev_time = max(0, cur_time - self.check_time_interval)

            # update job status
            self.update_pending_jobs(cur_time)
            self.update_running_jobs(cur_time)

            # update job and user fairness
            self.update_job_fairness()
            self.update_user_fairness(cur_time)

            # check terminated jobs (lease expire or finish) and delay jobs
            self.check_terminated_jobs(cur_time)
            self.check_delay_jobs()
            self.update_user_used_quota()

            # dispatch new seen job
            self.dispatch_new_job(prev_time, cur_time)

            # try to allocate resources to pending jobs
            self.allocate_job_resources_with_user(cur_time)

            cur_time = cur_time + self.check_time_interval

        self.lease_summary()
