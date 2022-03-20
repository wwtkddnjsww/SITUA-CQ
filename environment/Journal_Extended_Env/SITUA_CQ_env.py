from environment.Journal_Extended_Env.client_journal import *
import numpy as np
import math

class SITUA_CQ:
    def __init__(self, env):
        self.name = "SITUA-CQ"
        self.env = env



    def cluster_and_quantization_level_selection(self, theta_b, theta_k, theta_d):
        selected_cluster = []
        check_list = [0 for i in range(len(self.env.clients))]
        time_list = []
        max_time = 0
        uniform_dist = [0.1 for i in range(10)]
        cluster_list = []
        temp_cluster = []

        clients_distribution = []
        min_kl_div = 999999
        for i in range(len(self.env.clients)):
            temp_idx = -1
            for j in range(len(self.env.clients)):
                if check_list[j] == 0 and self.kl_div_cluster(temp_cluster, j, uniform_dist)<min_kl_div: #todo: find min if check_list is not 0
                    temp_idx = j
                    min_kl_div = self.kl_div_cluster(temp_cluster, j, uniform_dist)
            if temp_idx != -1:
                temp_cluster.append(temp_idx)
                check_list[temp_idx] = 1
            if self.kl_div_cluster(temp_cluster, -1, uniform_dist) <= theta_d or temp_idx == -1:
                cluster_list.append(temp_cluster)
                temp_cluster = []
                min_kl_div = 999999
        l = 0
        min_time  = 9999999999
        selection_list = [0 for i in range(len(cluster_list))]
        while l < theta_k:
            temp_n = -1
            for i in range(len(cluster_list)):
                if selection_list[i] == 0 and self.get_cluster_time_for_algorithm(cluster_list[i]) <= min_time:
                    temp_n = i
                    min_time = self.get_cluster_time_for_algorithm(cluster_list[i])
            if temp_n != -1:
                selected_cluster.append(cluster_list[temp_n])
                selection_list[temp_n] = 1
                l += len(cluster_list[temp_n])
                min_time = 9999999999
        theta_list = [0 for i in range(len(selected_cluster))]
        total_client=0
        for i in selected_cluster:
            total_client+=len(i)
        idx = 0
        bit_list = [32, 16, 8]
        for i in range(len(selected_cluster)):
            theta_list[i] = bit_list[idx]
            if sum(psi_b_for_proposed(theta_list, selected_cluster)[0:idx+1]) >= sum(theta_b[0:idx+1])*total_client:
                idx = idx+1
        round_time = self.get_total_round_time(selected_cluster, theta_list)
        return round_time, selected_cluster, theta_list

    def get_total_round_time(self, selected_cluster, theta_list):
        round_time_list = [0 for i in range(len(selected_cluster))]
        for i in range(len(selected_cluster)):
            for j in selected_cluster[i]:
                if theta_list[i] == 32:
                    tmp_time = self.env.clients[j].computing + self.env.clients[j].downlink + self.env.clients[j].uplink
                elif theta_list[i] == 16:
                    tmp_time = self.env.clients[j].computing/1.46 + self.env.clients[j].downlink/2 + self.env.clients[j].uplink/2
                elif theta_list[i] == 8:
                    tmp_time = self.env.clients[j].computing/1.81 + self.env.clients[j].downlink/4 + self.env.clients[j].uplink/4
                else:
                    print('ERROR')
                round_time_list[i] += tmp_time
        max_time = max(round_time_list)
        return max_time

    def get_cluster_time_for_algorithm(self, client_cluster_list):
        idx = 0
        cluster_time = 0 # total number of cluster
        for i in client_cluster_list:
            tmp_time = self.env.clients[i].computing + self.env.clients[i].downlink + self.env.clients[i].uplink

            cluster_time += tmp_time
            idx += 1
        return cluster_time

    def kl_div_cluster(self, temp_cluster, client_idx, uniform_dist):
        cluster_dist = [float(0) for i in range(10)]
        cluster_dist = np.array(cluster_dist)
        for i in temp_cluster:
            cluster_dist += self.env.clients[i].distribution
        if client_idx != -1:
            cluster_dist += self.env.clients[client_idx].distribution
        if np.sum(cluster_dist) != 0:
            cluster_dist = cluster_dist / np.sum(cluster_dist)
        return kl_divergence(cluster_dist, uniform_dist)


    