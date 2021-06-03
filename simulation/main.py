
import os, sys
import csv
import options
#from alg import PlaceMentFactory, FifoScheduler, DlasScheduler, GittinsScheduler, \
#    GandivaScheduler, TimeAwareScheduler, TimeAwareWithLeaseScheduler, LeaseScheduler, FairnessScheduler
from alg import PlaceMentFactory, LeaseScheduler
from server import cluster
from client import jobs, users
from client.jobs import Job
from client.users import VanillaUser
from utils.logger import getLogger

opt = options.Singleton.init()
print(opt)
USERS = users.USERS
JOBS = jobs.JOBS
CLUSTER = cluster.CLUSTER
logger = getLogger(level=opt.log_level)


def parse_job_file(trace_file):
    #check trace_file is *.csv
    fd = open(trace_file, 'r')
    deli = ','
    if ((trace_file.find('.csv') == (len(trace_file) - 4))):
        deli = ','
    elif ((trace_file.find('.txt') == (len(trace_file) - 4))):
        deli = ' '

    reader = csv.DictReader(fd, delimiter = deli)
    for info_dict in reader:
        if 'num_gpu' not in info_dict or info_dict['num_gpu'] == "0":
            continue
        exist_user = USERS.index_user(info_dict['user'])
        info_dict['user'] = exist_user
        new_job = Job(info_dict)
        exist_user.submit_job(new_job)
    
    assert JOBS.num_job == len(JOBS.job_list) 

    JOBS.sort_all_jobs(key='submit_time')
    logger.info('---------------------------------- Get %d TF jobs in total ----------------------------------' % JOBS.num_job)
    fd.close()


def prepare_cluster():
    CLUSTER.init_infra(num_switch=opt.num_switch, 
                        num_node_p_switch=opt.num_node_p_switch, 
                        num_gpu_p_node=opt.num_gpu_p_node, 
                        num_cpu_p_node=opt.num_cpu_p_node, 
                        mem_p_node=opt.mem_p_node)


def prepare_user(namelist):
    with open(namelist, 'r') as f:
        names = f.readlines()
        for name in names:
            new_user = VanillaUser(JOBS=JOBS, CLUSTER=CLUSTER, name=name.strip(), logger=logger)
            USERS.add_user(new_user)
            


def summary_all_jobs():
    assert all([job['status'] == 'END' for job in JOBS.job_list])
    num_job = 1.0 * len(JOBS.job_list)
    jct = 0
    
    for job in JOBS.job_list:
        jct += (job['end_time'] - job['submit_time']) / num_job
        # logger.info("%d %d" %( job['end_time'] - job['submit_time'], job['pending_time']))
    logger.info('average of scheduler %s is %d'%(opt.schedule,  jct))


def main():
    prepare_cluster()
    prepare_user(opt.name_list)
    parse_job_file(opt.trace)

    JOBS.prepare_job_start_events()

    global PM
    PM = PlaceMentFactory(cluster=CLUSTER, name=opt.placement) # construct placement after init cluster
    
    if opt.schedule == 'lease':
        scheduler = LeaseScheduler(JOBS=JOBS, CLUSTER=CLUSTER, placement=PM, name=opt.schedule,
                                   logger=logger, check_time_interval=opt.check_time_interval,
                                   lease_term_interval=opt.lease_term_interval, replacement=False,
                                   job_selection=opt.job_selection, share=opt.share, USERS=USERS,
                                   fairness_output=opt.fairness_output,
                                   user_quota=opt.user_quota,
                                   dynamic_quota_ratio=opt.dynamic_quota_ratio,
                                   stavation_threshold=opt.stavation_threshold,
                                   delay_threshold=opt.delay_threshold,
                                   block_scheduling=opt.block_scheduling,
                                   numgpu_fallback_threshold=opt.numgpu_fallback_threshold,
                                   dist_trace_path=opt.dist_trace_path, metrics=opt.metrics,
                                   disc_priority_k=opt.disc_priority_k,
                                   metrics_path=opt.metrics_path, skip_big_job_lease=opt.skip_big_job_lease)
    else: 
        raise NotImplementedError

    scheduler.run()
    summary_all_jobs()



if __name__ == '__main__':
    main()
