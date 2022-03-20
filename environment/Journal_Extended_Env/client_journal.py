import math
import random
import numpy as np


def get_round_time(selection_list,clients):
    idx = 0
    max_time = 0
    for i in selection_list:
        if i == 32:
            tmp_time = clients[idx].computing + clients[idx].downlink + clients[idx].uplink
            if tmp_time > max_time:
                max_time = tmp_time
        if i == 16:
            tmp_time = clients[idx].computing/1.46 + clients[idx].downlink + clients[idx].uplink/2
            if tmp_time > max_time:
                max_time = tmp_time
        if i == 8:
            tmp_time = clients[idx].computing/1.81 + clients[idx].downlink + clients[idx].uplink/4
            if tmp_time > max_time:
                max_time = tmp_time
        idx += 1
    return max_time

def get_round_time_cluster(selection_list,client_cluster_list,clients):
    idx = 0
    cluster_time_list = [0 for i in range(len(selection_list))] #total number of cluster
    for i in client_cluster_list:
        tmp_time = 0
        if selection_list[i] == 32:
            tmp_time = clients[idx].computing + clients[idx].downlink + clients[idx].uplink
        if selection_list[i] == 16:
            tmp_time = clients[idx].computing/1.46 + clients[idx].downlink/2 + clients[idx].uplink/2
        if selection_list[i] == 8:
            tmp_time = clients[idx].computing/1.81 + clients[idx].downlink/4 + clients[idx].uplink/4
        cluster_time_list[i] += tmp_time
        idx += 1
    max_time = max(cluster_time_list)
    return max_time

def get_round_time_comm_only(selection_list,clients):
    idx = 0
    max_time = 0
    for i in selection_list:
        if i == 32:
            tmp_time = clients[idx].computing + clients[idx].downlink + clients[idx].uplink
            if tmp_time > max_time:
                max_time = tmp_time
        if i == 16:
            tmp_time = clients[idx].computing + clients[idx].downlink + clients[idx].uplink/2
            if tmp_time > max_time:
                max_time = tmp_time
        if i == 8:
            tmp_time = clients[idx].computing + clients[idx].downlink + clients[idx].uplink/4
            if tmp_time > max_time:
                max_time = tmp_time
        idx += 1
    return max_time

def kl_divergence(p,q):
    return np.sum(np.where(p!=0, p*np.log(p/q),0))


def psi_d(clients, cluster_clients, cluster_theta, theta_d,len_of_cluster):
    idx = 0
    clients_distribution = []
    uniform_dist = [0.1 for i in range(10)]
    for i in range(len(clients)):
        clients_distribution.append(clients[i].distribution)

    cluster_dist = [[float(0) for i in range(10)] for i in range(len_of_cluster)]
    cluster_dist = np.array(cluster_dist)
    for i in cluster_clients:
        cluster_dist[i] += clients[idx].distribution
        idx +=1
    for i in range(len_of_cluster):
        if np.sum(cluster_dist[i]) != 0:
            cluster_dist[i] = cluster_dist[i]/np.sum(cluster_dist[i])
    tmp_ret = True
    for i in range(len_of_cluster):
        if cluster_theta[i] != 0 and kl_divergence(cluster_dist[i], uniform_dist) > theta_d: # kl divergence!
            tmp_ret = False
    return tmp_ret

def psi_b_for_randql(selection_list):
    bit_8=0
    bit_16=0
    bit_32=0
    for i in selection_list:
        if i == 8:
            bit_8+=1
        if i == 16:
            bit_16+=1
        if i == 32:
            bit_32+=1
    bit_8 = bit_8
    bit_16 = bit_16
    bit_32 = bit_32

    return bit_32, bit_16, bit_8


def psi_b_for_proposed(selection_list,selected_cluster):
    bit_8=0
    bit_16=0
    bit_32=0
    idx = 0
    for i in range(len(selection_list)):
        if selection_list[i] == 8:
            bit_8+=len(selected_cluster[i])
        if selection_list[i] == 16:
            bit_16+=len(selected_cluster[i])
        if selection_list[i] == 32:
            bit_32+=len(selected_cluster[i])
    bit_8 = bit_8
    bit_16 = bit_16
    bit_32 = bit_32

    return bit_32, bit_16, bit_8

def psi_b(selection_list,theta_b):
    bit_8=0
    bit_16=0
    bit_32=0
    for i in selection_list:
        if i == 8:
            bit_8+=1
        if i == 16:
            bit_16+=1
        if i == 32:
            bit_32+=1
    bit_8 = bit_8
    bit_16 = bit_16
    bit_32 = bit_32
    N= 0
    for i in selection_list:
        if i !=0:
            N+=1
    if N == 0:
        return False
    if theta_b[0] <= bit_32/N and (theta_b[0]+theta_b[1]) <= (bit_32 + bit_16)/N:
        return True
    else: return False

def psi_b_cluster_ver(cluster, selection_list,theta_b):
    checker = [0 for i in range(len(cluster))]
    for i in range(len(cluster)):
        if selection_list[i] != 0:
            checker[cluster[i]] = 1
    for i in range(len(cluster)):
        if checker[i] == 0 and selection_list[i] != 0:
            return False
    bit_8=0
    bit_16=0
    bit_32=0
    idx = 0
    for i in selection_list:
        if checker[idx] == 1:
            if i == 8:
                bit_8+=1
            if i == 16:
                bit_16+=1
            if i == 32:
                bit_32+=1
        idx +=1
    N = 0
    for i in range(len(checker)):
        if checker[i] ==1:
            N+=1
    bit_8 = bit_8
    bit_16 = bit_16
    bit_32 = bit_32
    if N == 0:
        return False
    if theta_b[0] <= bit_32/N and (theta_b[0]+theta_b[1]) <= (bit_32 + bit_16)/N:
        return True
    else: return False



def psi_n(selection_list):
    num_of_selected = 0
    for i in selection_list:
        if i != 0:
            num_of_selected+=1

    return num_of_selected

class Env:
    def __init__(self,num_clients, class_tier_list, distribution_list,model_size=20):
        self.num_clients = num_clients
        self.round_count = 0
        self.class_tier_list = class_tier_list
        self.clients = [Client(i,distribution_list[i],model_size) for i in range(self.num_clients)]

    def new_round(self):
        comp_tier_list = []
        comu_tier_list = []
        _tier=0
        for i in self.class_tier_list:
            for _ in range(i):
                comp_tier_list.append(_tier)
                comu_tier_list.append(_tier)
            _tier += 1
        random.shuffle(comu_tier_list)
        random.shuffle(comp_tier_list)
        for i in range(self.num_clients):
            self.clients[i].new_round(comu_tier_list[i],comp_tier_list[i])


class Client:
    def __init__(self,id,distribution,model_size=20,bandwidth=5):
        self.id = id
        self.computing=0
        self.downlink=0
        self.uplink=0
        self.F = model_size*20
        self.power=bandwidth
        self.model_size=model_size
        self.distribution = distribution

    def new_round(self,comm_tier,comp_tier):
        self.computing = self.computing_time(comp_tier)
        self.downlink  = self.downlink_time(comm_tier)
        self.uplink    = self.uplink_time(comm_tier)

    def computing_time(self,class_tier):
        computing = random.normalvariate((class_tier+1)*4, 0.5)*self.F/400
        while computing < 0:
            computing = random.normalvariate((class_tier+1)*4, 0.5)*self.F/400
        return computing



    def uplink_time(self,class_tier):
        snr= 0
        if snr == 0:
            snr = random.normalvariate(math.pow(10,(4-class_tier)),math.pow(10,(3-class_tier))*2)
        while snr < 1:
            snr = random.normalvariate(math.pow(10,(4-class_tier)),math.pow(10,(3-class_tier))*2)
        Mbps = self.power*math.log2(1+snr) #uplinkn
        self.MB_per_sec = Mbps/8
        uplink = self.model_size/self.MB_per_sec #uplink time
        return uplink


    def downlink_time(self,class_tier):
        snr= 0
        if snr == 0:
            snr = random.normalvariate(math.pow(10,(4-class_tier)),math.pow(10,(3-class_tier))*2)
        while snr < 1:
            snr = random.normalvariate(math.pow(10,(4-class_tier)),math.pow(10,(3-class_tier))*2)
        Mbps = self.power*math.log2(1+snr) #downlinnk
        MB_per_sec = Mbps/8
        downlink = self.model_size/MB_per_sec #downlink time
        return downlink


class ClientSelectionAlgorithm:
    pass


