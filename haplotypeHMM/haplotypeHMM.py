from scipy.special import logsumexp
import time
import math
import numpy as np
from joblib import Parallel, delayed
# from multiprocessing import Pool
from .genoSegmentGraph import genoSegmentGraph
class haplotypeHMM(object):
    '''
    hidden_states -> mosaic state (X,Y), indicating states of diploids from haplotypes X and Y
    transition -> transition prob for mosaic state
    emission -> prob of genotype condition on mosaic state
    inital -> prob initial mosaic state
    
    '''
    def __init__(self, hap_graph,pseudocount=1e-100,seed=42):
        self.hap_graph = hap_graph
        self.B = self.hap_graph.B
        self.total_num_haplotypes = self.hap_graph.total_num_haplotypes
        self.seed = seed
        self.pseudocount = pseudocount
        K = self.total_num_haplotypes
        self.theta = 1/(math.log(K) + 0.5772)
        self.population_size = 15000

    def emit_prob(self, this_node, geno_graph_node):
        K = self.total_num_haplotypes
        theta = self.theta
        if geno_graph_node.allele == this_node.allele:
            delta = 1
        else:
            delta = 0
        prob = K/(K+theta) * delta + theta/(K+theta)/2
        assert prob >= 0 and prob <= 1
        if prob == 0:
            prob = self.pseudocount
        if prob == 1:
            prob -= self.pseudocount
        return prob
    
    def transition_prob(self, this_node, next_node):
        K = self.total_num_haplotypes
        rho = 1 - np.exp(-4*self.population_size*this_node.dist(next_node)/K)
        if next_node.id in this_node.outer_weights:
            edge_weight = this_node.outer_weights[next_node.id]
        else:
            edge_weight = 0
        prob = (1-rho) * edge_weight / this_node.weight + rho * next_node.weight / self.total_num_haplotypes
        assert prob >= 0 and prob <= 1
        if prob == 0:
            prob = self.pseudocount
        if prob == 1:
            prob -= self.pseudocount
        return prob
    
    def init_prob(self, node):
        prob = node.weight / self.total_num_haplotypes
        assert prob >= 0 and prob <= 1
        if prob == 0:
            prob = self.pseudocount
        if prob == 1:
            prob -= self.pseudocount
        return prob
    
    def logsumexp(self,lst):
        arr = np.array(lst)
        return logsumexp(arr)
    def forward(self, geno_graph):
        log_left_list = []
        log_left = np.empty(shape=(len(geno_graph.nodes[0]),len(self.hap_graph.nodes[0])))
        # for first marker
        for i,geno_graph_node in enumerate(geno_graph.nodes[0]):
            for j,hap_graph_node in enumerate(self.hap_graph.nodes[0]):
                log_left[i,j] = np.log(self.init_prob(hap_graph_node)) + np.log(self.emit_prob(hap_graph_node, geno_graph_node))
        log_left_list.append(log_left)
        all_marker_id = list(sorted(self.hap_graph.nodes.keys()))
        for marker_id in all_marker_id[1:]:
            log_left = np.empty(shape=(len(geno_graph.nodes[marker_id]),len(self.hap_graph.nodes[marker_id])))
            for i,geno_graph_node in enumerate(geno_graph.nodes[marker_id]):
                for j,hap_graph_node in enumerate(self.hap_graph.nodes[marker_id]):
                    all_temp_log_left = []
                    for u,prev_geno_graph_node in enumerate(geno_graph.nodes[marker_id-1]):
                        if prev_geno_graph_node in geno_graph_node.inner_nodes:
                            for v,prev_hap_graph_node in enumerate(self.hap_graph.nodes[marker_id-1]):
                                all_temp_log_left.append(log_left_list[marker_id-1][u,v] +  np.log(self.transition_prob(prev_hap_graph_node,hap_graph_node)))
                    
                    log_left[i,j] = np.log(self.emit_prob(hap_graph_node, geno_graph_node)) + self.logsumexp(all_temp_log_left)
            log_left_list.append(log_left)
        return log_left_list
    def backward(self, geno_graph):
        log_right_list = []
        # for last marker
        all_marker_id = list(sorted(self.hap_graph.nodes.keys()))
        marker_id = all_marker_id[-1]
        log_right = np.empty(shape=(len(geno_graph.nodes[marker_id]),len(self.hap_graph.nodes[marker_id])))
        for i,geno_graph_node in enumerate(geno_graph.nodes[marker_id]):
            for j,hap_graph_node in enumerate(self.hap_graph.nodes[marker_id]):
                log_right[i,j] = 0
        log_right_list.insert(0,log_right)
        for marker_id in all_marker_id[-2::-1]:
            log_right = np.empty(shape=(len(geno_graph.nodes[marker_id]),len(self.hap_graph.nodes[marker_id])))
            for i,geno_graph_node in enumerate(geno_graph.nodes[marker_id]):
                for j,hap_graph_node in enumerate(self.hap_graph.nodes[marker_id]):
                    all_temp_log_right = []
                    for u,next_geno_graph_node in enumerate(geno_graph.nodes[marker_id+1]):
                        if next_geno_graph_node in geno_graph_node.outer_nodes:
                            for v,next_hap_graph_node in enumerate(self.hap_graph.nodes[marker_id+1]):
                                all_temp_log_right.append(log_right_list[0][u,v] +  np.log(self.transition_prob(hap_graph_node,next_hap_graph_node)))
                    
                    log_right[i,j] = np.log(self.emit_prob(hap_graph_node, geno_graph_node)) + self.logsumexp(all_temp_log_right)
            log_right_list.insert(0,log_right)
        return log_right_list
    def expectation(self,geno_graph):
        log_alpha_list = self.forward(geno_graph)
        log_beta_list = self.backward(geno_graph)
        
        all_marker_id = list(sorted(self.hap_graph.nodes.keys()))
        marker_id = all_marker_id[0]
        init_log_marginal = np.empty(shape=(len(geno_graph.nodes[marker_id])))
        for i,geno_graph_node in enumerate(geno_graph.nodes[marker_id]):
            init_log_marginal[i] = self.logsumexp(log_beta_list[0][i,:]) - self.logsumexp(log_beta_list[0])
        
        log_marginal_list = [init_log_marginal]
        for marker_id in all_marker_id[0:-1]:
            log_marginal = np.empty(shape=(len(geno_graph.nodes[marker_id]),len(geno_graph.nodes[marker_id+1])))
            temp_log_marginal = []
            for i1,geno_graph_node_1 in enumerate(geno_graph.nodes[marker_id]):
                for i2,geno_graph_node_2 in enumerate(geno_graph.nodes[marker_id+1]):
                    for j1,hap_graph_node_1 in enumerate(self.hap_graph.nodes[marker_id]):
                        for j2,hap_graph_node_2 in enumerate(self.hap_graph.nodes[marker_id+1]):
                            temp_log_marginal.append(log_alpha_list[marker_id][i1,j1] + np.log(self.transition_prob(hap_graph_node_1,hap_graph_node_2))+\
                            np.log(self.emit_prob(hap_graph_node_2,geno_graph_node_2)) + log_beta_list[marker_id+1][i2,j2])
            index = 0
            for i1,geno_graph_node_1 in enumerate(geno_graph.nodes[marker_id]):
                for i2,geno_graph_node_2 in enumerate(geno_graph.nodes[marker_id+1]):
                    log_marginal[i1,i2] = temp_log_marginal[index] - self.logsumexp(temp_log_marginal)
                    index += 1
            log_marginal_list.append(log_marginal)
        return log_marginal_list
                            
        
    def predict(self,genos,choice='Viterbi',threads=1):
        if choice == 'Viterbi':
            genos_list = [geno for geno in genos]
            
            predictions = Parallel(n_jobs=threads)(delayed(self.predict_Viterbi)(geno) for geno in genos_list)
        return np.array(predictions)

    def predict_Viterbi(self, geno):
        geno_graph = genoSegmentGraph(geno,self.B)
        log_marginal_list = self.expectation(geno_graph)
        
        # Predict by Viterbi
        previous_col_probs = {}
        for i in range(log_marginal_list[0].shape[0]):
            for j in range(log_marginal_list[0].shape[0]):
                previous_col_probs[(i,j)] = log_marginal_list[0][i] + log_marginal_list[0][j]
        traceback = []
        incompatible_penalty = 0.1
        all_marker_id = list(sorted(geno_graph.nodes.keys()))
        for t in range(1, len(log_marginal_list)): 
            marker_id = all_marker_id[t-1]
            traceback_next = {}
            previous_col_probs_next = {}
            # next state
            for hap1_i2,hap1_geno_graph_node_i2 in enumerate(geno_graph.nodes[marker_id+1]):
                for hap2_i2,hap2_geno_graph_node_i2 in enumerate(geno_graph.nodes[marker_id+1]):
                        
                    best_prob = -np.inf
                    best_haps = None
                    # prev state
                    for hap1_i1,hap1_geno_graph_node_i1 in enumerate(geno_graph.nodes[marker_id]):
                        for hap2_i1,hap2_geno_graph_node_i1 in enumerate(geno_graph.nodes[marker_id]):

                            num_mismatches = np.abs(geno[marker_id] - (hap1_geno_graph_node_i1.allele + hap2_geno_graph_node_i1.allele)) + np.abs(geno[marker_id+1] - (hap1_geno_graph_node_i2.allele + hap2_geno_graph_node_i2.allele))
                            prob = previous_col_probs[(hap1_i1,hap2_i1)] + log_marginal_list[t][hap1_i1,hap1_i2] + \
                            log_marginal_list[t][hap2_i1,hap2_i2]
                            prob *= 1+incompatible_penalty * num_mismatches
                            if prob > best_prob:
                                best_prob = prob
                                best_haps = (hap1_i1,hap2_i1)
                    if best_haps != None:
                        traceback_next[(hap1_i2,hap2_i2)] = best_haps
                        previous_col_probs_next[(hap1_i2,hap2_i2)] = best_prob
            previous_col_probs = previous_col_probs_next
            traceback.append(traceback_next)
            

        max_final_state = None
        max_final_prob = -np.inf
        for state,prob in previous_col_probs.items():
            if prob > max_final_prob:
                max_final_prob = prob
                max_final_state = state
        
        nodes = geno_graph.nodes[all_marker_id[-1]]
#         if max_final_state[0] >= len(nodes) or max_final_state[1] >= len(nodes):
#             return geno,geno_graph,log_marginal_list
        result = [(nodes[max_final_state[0]],nodes[max_final_state[1]])]
        for t in range(len(all_marker_id)-2,-1,-1):
            marker_id = all_marker_id[t]
            nodes = geno_graph.nodes[marker_id]
            max_final_state = traceback[t][max_final_state]
            result.append((nodes[max_final_state[0]],nodes[max_final_state[1]]))
        results = result[::-1]
        hap1 = []
        hap2 = []
        for (hap1_node,hap2_node) in results:
            if hap1_node.type == 'inter':
                hap1 += hap1_node.haplotype
            if hap2_node.type == 'inter':
                hap2 += hap2_node.haplotype
        return [hap1,hap2]
