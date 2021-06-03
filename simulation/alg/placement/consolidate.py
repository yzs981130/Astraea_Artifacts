import os, sys
import math
import random
sys.path.insert(0, os.path.basename(__file__) + os.sep + '..' + os.sep + '..')
from server.switch import _Switch
from server.node import _Node
from utils import util
from alg.utils.topology import Topology
from .base import BasePlaceMent


class ConsolidatePlaceMent(BasePlaceMent):
    __alias__ = 'consolidate'
    def __init__(self, cluster, name):
        super(ConsolidatePlaceMent, self).__init__(cluster=cluster, name=name)
    

    def place_jobs(self, job):
        '''
        consolidate first, but randomly pick machines;
        if cross machines, still try to consolidate.
        if can't consolidate, consider spreed the jobs;
        also PS is randomly placed on the selected machines
        '''

        num_ps, num_w, demand_node = len(job['ps_network']), job['num_gpu'], 0
        assert num_ps == num_w or (num_ps == 0 and num_w == 1)
        
        # place as few nodes as possible
        num_gpu_list = [self.num_gpu_p_node for _ in range(self.num_gpu_p_node-1, num_w, self.num_gpu_p_node)]
        demand_node = len(num_gpu_list)
        if  sum(num_gpu_list) < num_w:
            demand_node += 1
            num_gpu_list.append(num_w % self.num_gpu_p_node)

        # go through workers
        w_node_list, w_switch_list, p_done = list(), list(), True
        
        for i in range(demand_node):
            allocated = False
            need_gpu = num_gpu_list[i]
            need_cpu = need_gpu * 2 if num_w == 1 else need_gpu * 6

            for switch_idx in range(self.num_switch):
                switch = self.cluster.switch_list[switch_idx]
                for node_idx in range(self.num_node_p_switch):
                    node = switch.node_list[node_idx]
                    allocated = allocated or self.allocate_resource(job=job, resource={'node':node, 'switch':switch}, node_list=w_node_list, \
                                                                switch_list=w_switch_list, gpu_num=1, cpu_num=need_cpu // need_gpu, job_num=need_gpu)
                    if allocated: break
                if allocated == True: break

            # go through all the machines, can't consolidate the jobs
            if allocated == False:
                p_done = False
                break

        # can't conslidate
        if p_done == False:  
            remain_gpu = 0
            for j in range(i, demand_node):            
                remain_gpu += num_gpu_list[j]
            if remain_gpu <= 1:
                # release allocated resource
                for node in w_node_list:
                    assert node.release_job_gpu_cpu(num_gpu=1, num_cpu=6, job=job) == True
                return False
            
            for switch_idx in range(self.num_switch):
                switch = self.cluster.switch_list[switch_idx]
                for node_idx in range(self.num_node_p_switch):
                    node = switch.node_list[node_idx]
                    free_cpu, free_gpu = node.check_free_cpus(), node.check_free_gpus()
                    max_capacity = min(min(free_cpu // 6, free_gpu), remain_gpu)
                    if max_capacity == 0:
                        continue
                    assert self.allocate_resource(job=job, resource={'node':node, 'switch':switch}, node_list=w_node_list, \
                                            switch_list=w_switch_list, gpu_num=1, cpu_num=6, job_num=max_capacity) == True
                
                    remain_gpu = int(remain_gpu - max_capacity)
                    
                    if remain_gpu == 0: break
                if remain_gpu == 0: break
            
            if remain_gpu != 0:
                for node in w_node_list:
                    assert node.release_job_gpu_cpu(num_gpu=1, num_cpu=6, job=job) == True
                return False
        
        assert len(w_node_list) == num_w
        # randomly place PS to node_list
        ps_node_list, ps_switch_list = list(), list()
        for i in range(num_ps):
            ps_node_list.append(w_node_list[i])
            ps_switch_list.append(w_switch_list[i])

        #go through all the related nodes
        # node_list = list() # useful for network load, but now no use
        # for i in range(len(w_node_list)):
        #     self.update_node_list_info(w_node_list[i], node_list, worker=1, ps=0)

        # for i in range(len(ps_node_list)):
        #     self.update_node_list_info(ps_node_list[i], node_list, worker=0, ps=i)

        # rocess job placement information
        
        for i, (s_id, node) in enumerate(zip(w_switch_list, w_node_list)):
            node_dict = {
                'id' : node.id, 
                'node_instance' : node, 
                'num_gpu' : 1,
                'num_cpu' : 2,
                'mem' : job['model']['mem_util'],
                'tasks': list(), 
            }
            job['placements'].append({
                'switch' : s_id, 
                'nodes' : [node_dict],
            })
        
        for i, (s_id, node) in enumerate(zip(ps_switch_list, ps_node_list)):
            node_dict = {
                'id' : node.id, 
                'node_instance' : node, 
                'num_gpu' : 0, 
                'num_cpu' : 4, 
                'mem' : 0, # job['model']['mem_util'], fix a bug
                'tasks' : list(), 
            }
            job['placements'].append({
                'switch': s_id, 
                'nodes' : [node_dict]
            })

        job['topology'] = Topology(job=job, placements=job['placements'])
        return True

