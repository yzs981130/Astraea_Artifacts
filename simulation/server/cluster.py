import os, sys
import math
import random
from .switch import _Switch
from .node import _Node
sys.path.insert(0, os.path.basename(__file__) + os.sep + '..')
from utils import util


class _Cluster(object):

    def __init__(self, num_switch=0, num_node_p_switch=0, num_gpu_p_node=0, num_cpu_p_node=0, mem_p_node=0):
        ''' Init GPU cluster with basic switch, node, gpu information'''
        self.set_spec(num_switch=num_switch, num_node_p_switch=num_node_p_switch, \
                    num_gpu_p_node=num_gpu_p_node, num_cpu_p_node=num_cpu_p_node, mem_p_node=mem_p_node)

        #for non-placement
        self.switch_list = list()
        #for gandiva
        self.set_node_group()
 

    def set_node_group(self, ):
        self.free_nodes = list()
        self.node_g = dict()
        for i in [1, 2, 4, 8, 12, 16, 24, 32, 64]:
            setattr(self, 'node_g{}'.format(i), list())
            self.node_g[i] = getattr(self, 'node_g{}'.format(i))
        
    
    def set_spec(self, num_switch=0, num_node_p_switch=0, num_gpu_p_node=0, num_cpu_p_node=0, mem_p_node=0):
        self.num_switch =  num_switch
        self.num_node_p_switch = num_node_p_switch
        self.num_gpu_p_node = num_gpu_p_node
        self.num_cpu_p_node = num_cpu_p_node
        self.mem_p_node = mem_p_node
        self.num_node = num_switch * num_node_p_switch
        self.num_gpu = self.num_node * num_gpu_p_node
        self.num_cpu = self.num_node * num_cpu_p_node
        self.free_gpu = self.num_gpu
        self.mem = self.num_node * mem_p_node


    def print_cluster_spec(self):
        print('Custer Spec')
        print('#ofswitch: %d, #ofnode: %d, #ofgpu: %d, #ofcpu: %d, #ofmem: %d'%(self.num_switch, self.num_node, self.num_gpu, self.num_cpu, self.mem))
        print('#ofnode/switch: %d, #ofgpu/node: %d, #ofcpu/node: %d, #ofmem/node: %d' % (self.num_node_p_switch, self.num_gpu_p_node, self.num_cpu_p_node, self.mem_p_node))


    def init_infra(self, num_switch=0, num_node_p_switch=0, num_gpu_p_node=0, num_cpu_p_node=0, mem_p_node=0):
        # Init and create cluster infration entities (switches, nodes) by using class _Switch, _Node
        self.set_spec(num_switch, num_node_p_switch, num_gpu_p_node, num_cpu_p_node, mem_p_node)

        # create/init switch and node objects    
        for switch_id in range(self.num_switch):
            switch_instance = _Switch(switch_id, self.num_node_p_switch, self.num_gpu_p_node, self.num_cpu_p_node, self.mem_p_node) 
            switch_instance.add_nodes(self.num_node_p_switch, self.num_gpu_p_node, self.num_cpu_p_node, self.mem_p_node, self)
            self.switch_list.append(switch_instance)

        #print('Cluster is ready to use')
        #self.print_cluster_spec()

    
    def init_gandiva_nodes(self, ):
        # init node class
        for switch in self.switch_list:
            for node in switch.node_list:
                self.free_nodes.append(node)
        assert len(self.free_nodes) == self.num_node, '# of free nodes {}  is incorrect'.format(len(self.free_nodes))



    def release_gpus(self, job, status='END'):
        for placement in job['placements']:
            assert 'switch' in placement and 'nodes' in placement
            switch = self.switch_list[placement['switch']]
            assert switch.release_gpus(placement['nodes'], job) == True
        
        if status == 'END':
            job['status'] = 'END'
            print('**** job[%d] completed' % job['job_idx'])
        return True


    def release_job_resource(self, job, status='END'):

        for placement in job['placements']:
            assert 'switch' in placement and 'nodes' in placement
            switch = self.switch_list[placement['switch']]
            assert switch.release_job_resource(placement['nodes'], job=job) == True
        
        if status == 'END': job['status'] = 'END'
        
        job['gpus'] = list()
        job['placements'] = list() # prepare an empty job_placement 
        job['topology'] = None
        return True
    
    
    def check_free_gpus(self, ):
        return sum([switch.check_free_gpus() for switch in self.switch_list])

    
    def check_free_cpus(self, ):
        return sum([switch.check_free_cpus() for switch in self.switch_list])

    def check_total_gpus(self, ):
        return sum([switch.check_total_gpus() for switch in self.switch_list])

        
    def gandiva_node_set_adjust(self, cur_time, jobs):
        """
        when there are free nodes in cluster, reduce burden of heavy nodes
        """
        total_gpu_demands = 0
        nl_gpu_demands = dict()
        nl_gpu_occupied = dict()
        
        for num_gpu, node_list in self.node_g.items():
            total_jobs = 0
            occupied_gpus = 0

            for node_set in node_list:
                total_jobs += len(node_set['jobs'])
                occupied_gpus += len(node_set['nodes']) * self.num_gpu_p_node
            
            total_gpu_demands += total_jobs * num_gpu
            nl_gpu_demands[num_gpu] = total_jobs * num_gpu
            nl_gpu_occupied[num_gpu] = occupied_gpus
        
        if total_gpu_demands == 0:
            return 
        
        for num_gpu, node_list in self.node_g.items():
            if nl_gpu_demands[num_gpu] == 0:
                continue

            nl_gpu_plan = int(math.floor(1.0 * nl_gpu_demands[num_gpu] / total_gpu_demands * self.num_gpu))
            nl_gpu_target = min(nl_gpu_plan, nl_gpu_demands[num_gpu])
            nl_gpu_diff = nl_gpu_target - nl_gpu_occupied[num_gpu]

            if nl_gpu_diff > 0:
                # growth: 
                num_ns = int(math.ceil(1. * nl_gpu_diff / num_gpu))
                expand_ns = self.gandiva_node_set_expand(num_gpu, node_list, num_ns, cur_time, jobs)
            elif nl_gpu_diff < 0:
                # shrink
                num_ns = int(math.ceil(-1. * nl_gpu_diff / num_gpu))
                shrink_ns = self.gandiva_node_set_shrink(num_gpu, node_list, num_ns, cur_time, jobs)
        
       


    def gandiva_node_set_shrink(self, node_group, occupied_node_list, release_node_num, cur_time, jobs):
        '''
        ns_num_gpu: num_gpu of job in this node_set
        '''
        # can't shrink too many node_set ?? why
        # decrease ns nodes
        if len(occupied_node_list) <= release_node_num:
            release_node_num = len(occupied_node_list) - 1 # at least keep single node
        
        job_list = list()
        i = 0
        for i in range(1, release_node_num + 1):
            node_set = occupied_node_list.pop(0)

            if len(node_set['jobs']) > 0:
                job_list.extend(node_set['jobs'])
                update_info = {
                    'jobs': list(), 
                    'concurrency' : 0, 
                    'util' : 0, 
                    'num_jobs' : 0,
                }
                node_set.update(update_info)
                
            for node in node_set['nodes']:
                self.free_nodes.append(node)

        for job in job_list:
            node_set = occupied_node_list[0]
            job_util = round(job['model']['mem_util'], 2)
            node_set['util'] = round(node_set['util'] + job_util, 2)
            assert job not in node_set['jobs'], 'cannot repeat  too many times'
            node_set['jobs'].append(job)
            node_set['num_jobs'] += 1
            occupied_node_list.sort(key=lambda x: x.__getitem__('util'))
        if i > 0:
            print("node_g{} shrink {} node_sets" .format(node_group, i))
        return i
    

    def gandiva_node_set_expand(self, node_group, occupied_node_list, required_node_num, cur_time, jobs):
        num_node = int(math.ceil(node_group * 1. / self.num_gpu_p_node)) # add num_node more nodes
        i = 0
        for i in range(1, required_node_num + 1):
            if num_node <= len(self.free_nodes):
                node_set = {
                    'nodes' : list(), 
                    'jobs' : list(), 
                    'concurrency' : 0, 
                    'capacity' : int(num_node * self.num_gpu_p_node * 1.0 / node_group), 
                    'util' : 0, 
                    'num_gpus': node_group, 
                    'num_jobs' : 0, 
                }
                for _ in range(num_node):
                    node_set['nodes'].append(self.free_nodes.pop(0))
                occupied_node_list.append(node_set)

        if i > 0:
            job_list = list()
            for node_set in occupied_node_list:
                if len(node_set['jobs']) > 0:
                    job_list.extend(node_set['jobs'])
                    update_info = {
                        'jobs': list(), 
                        'concurrency' : 0, 
                        'util' : 0, 
                        'num_jobs' : 0,
                    }
                    node_set.update(update_info)
            
            for job in job_list:
                node_set = occupied_node_list[0]
                job_util = round(job['model']['mem_util'], 2)
                node_set['util'] = round(node_set['util'] + job_util, 2)
                assert job not in node_set['jobs'], 'cannot repeat too many times'
                node_set['jobs'].append(job)
                node_set['num_jobs'] += 1
                occupied_node_list.sort(key=lambda x: x.__getitem__('util'))
        
        print("node_g{} expand {} node_sets".format(node_group, i))

    

    def time_slicing_execute(self, cur_time, jobs, time_diff):
        node_release = False
        switch_job = int(cur_time % 60) == 0 # specify time, switch job
        used_gpus = 0
        
        for num_gpu, node_list in self.node_g.items():
            release_nodes = list() # release nodes
            for node_set in node_list:
                concurrency = 0
                total_util = 0
                for r_job in node_set['jobs']:
                    total_util = total_util + r_job['model']['mem_util']
                    if total_util > node_set['capacity']:
                        break
                    concurrency += 1


                tmp_used_gpus = \
                    num_gpu if (len(node_set['jobs']) * num_gpu  > self.num_gpu_p_node) else (len(node_set['jobs'] * num_gpu)) # TODO: figure out
                
                used_gpus += tmp_used_gpus

                i = 0
                end_job_list = list()
                for r_job in node_set['jobs']:
                    r_job['executed_time'] = r_job['executed_time'] + time_diff
                    if r_job['executed_time'] >= r_job['duration']:
                        r_job['end_time'] = cur_time + r_job['duration'] - r_job['executed_time']
                        r_job['status'] = 'END'
                        end_job_list.append(r_job)
                        print("job[%d] ends at time[%d]" %(r_job['job_id'], r_job['end_time']))
                    i += 1
                    if i >= concurrency:
                        break
                
                if switch_job and len(node_set['jobs']) > concurrency:
                    # node_set['jobs'].reverse()
                    random.shuffle(node_set['jobs'])

                for end_job in end_job_list:
                    jobs.running_jobs.remove(end_job)
                    node_set['jobs'].remove(end_job)
                    node_set['num_jobs'] = node_set['num_jobs'] - 1

                
                if len(node_set['jobs']) == 0:
                    assert node_set['num_jobs'] == 0
                    for node in node_set['nodes']:
                        self.free_nodes.append(node)
                    release_nodes.append(node_set)
                    used_gpus = used_gpus - tmp_used_gpus
                    node_release = True
                
            for release_node in release_nodes:
                node_list.remove(release_node)
            
        return node_release



CLUSTER = _Cluster()


_allowed_symbols = [
    'CLUSTER'
]