from .util import construct_possible_haps
import numpy as np
import math
class haplotypeSegmentGraph(object):
    '''
    H_g
    '''
    def __init__(self,genos,genetic_pos,B):
        '''
        B: number of hetero markers in each segment
        '''
        self.nodes = {}
        self.genetic_pos = genetic_pos
        self.B = B
        self.total_num_haplotypes = None
        self.build_haplotype_graph(genos)
    def __str__(self):
        nodes_count = 0
        for marker,nodes in self.nodes.items():
            nodes_count += len(nodes)
        log_num_haps = math.log(self.total_num_haplotypes)/math.log(2)
        output = f'''===========================
Number of haplotypes: {self.total_num_haplotypes} (~2^{log_num_haps})
Number of markers: {len(self.nodes)}
Number of nodes (# segment haplotypes(~=B) x # markers): {nodes_count}
==========================='''
        return output
    def __repr__(self):
        return self.__str__()
    def build_haplotype_graph(self,genos):
        B = self.B
        masking = genos.copy()
        masking[masking!=1] =0
        snp_hetero = masking.max(axis=0)
        cumsum_snp_hetero = np.cumsum(snp_hetero)
        bins = np.array([i*B for i in range(int(cumsum_snp_hetero[-1]//B)+1)])
        inds = np.digitize(cumsum_snp_hetero, bins,right=True)
        
        # for each segment
        marker_id = genos.shape[1]
        after_marker_nodes = []
        node_id = 0
        for unique_ind in np.unique(inds)[::-1]:
            segment_genos = genos[:,inds==unique_ind]
            _,haps = construct_possible_haps(segment_genos)
            last_marker = True
            for i in list(range(segment_genos.shape[1]))[::-1]:
                marker_id -= 1
                marker_nodes = []
                for hap in haps:
                    assert len(hap) == segment_genos.shape[1]
                    pos = self.genetic_pos.iloc[marker_id]
                    node = haplotypeSegmentNode(node_id,marker_id,hap,hap[i],pos)
                    node_id += 1
                    marker_nodes.append(node)
#                     if len(after_marker_nodes) == 0:
#                         print(marker_id+1)
                    if last_marker:
                        node.type = 'inter'
                        node.outer_nodes = after_marker_nodes
                        for outer_node in node.outer_nodes:
                            outer_node.inner_nodes.append(node)
                    else:
                        node.type = 'intra'
                        node.outer_nodes = [n for n in after_marker_nodes if n.haplotype==node.haplotype]
                        for outer_node in node.outer_nodes:
                            outer_node.inner_nodes.append(node)
                after_marker_nodes = marker_nodes
                last_marker = False
                self.nodes[marker_id] = after_marker_nodes
        total_num_haplotypes = 0
        for node in self.nodes[0]:
            total_num_haplotypes += self.forward(node)
        self.total_num_haplotypes = total_num_haplotypes
        for node in self.nodes[genos.shape[1]-1]:
            self.backward(node)
        for node in self.nodes[0]:
            self.update_weight(node)
        
    def forward(self,node):
        if len(node.outer_nodes) == 0:
            node.outer_weight = 1
            return 1
        else:
            all_num_outer_haplotypes  = 0
            for outer_node in node.outer_nodes:
                if outer_node.outer_weight == None:
                    all_num_outer_haplotypes += self.forward(outer_node)
                else:
                    all_num_outer_haplotypes += outer_node.outer_weight
            node.outer_weight = all_num_outer_haplotypes
            return all_num_outer_haplotypes
    def backward(self,node):
        if len(node.inner_nodes) == 0:
            node.inner_weight = 1
            return 1
        else:
            all_num_inner_haplotypes  = 0
            for inner_node in node.inner_nodes:
                if inner_node.inner_weight == None:
                    all_num_inner_haplotypes += self.backward(inner_node)
                else:
                    all_num_inner_haplotypes += inner_node.inner_weight
            node.inner_weight = all_num_inner_haplotypes
            return all_num_inner_haplotypes
    def update_weight(self,node):
        if node.weight == None:
            node.weight = node.inner_weight * node.outer_weight
            for outer_node in node.outer_nodes:
                node.outer_weights[outer_node.id] = node.inner_weight*outer_node.outer_weight
                self.update_weight(outer_node)
        
class haplotypeSegmentNode(object):
    def __init__(self,node_id,marker,haplotype,allele,pos):
        self.id = node_id
        self.marker =  marker
        self.haplotype = haplotype
        self.allele = allele
        self.weight = None
        self.inner_weight = None
        self.outer_weight = None
        self.type = None
        self.inner_nodes = []
        self.outer_nodes = []
        self.outer_weights = {}
        self.pos = pos
    def __str__(self):
        log_weight = math.log(self.weight)/math.log(2)
        log_inner_weight = math.log(self.inner_weight)/math.log(2)
        log_outer_weight = math.log(self.outer_weight)/math.log(2)
        output = f'''===========================
Haplotype segment Node: represents a possible haplotype state for this marker in the whole dataset
--------------------------
Node id: {self.id}
Marker id: {self.marker}
Haplotype: {self.haplotype}
Allele: {self.allele}
Type(it connects to another segment[inter] or connects to the node in the same segment[intra]): {self.type}
Weight (# haplotypes going through this node): {self.weight}(~2^{log_weight})
Inner weight(# haplotypes ending at this node): {self.inner_weight} (~2^{log_inner_weight})
Outer weight weight(# haplotypes starting from this node): {self.outer_weight} (~2^{log_outer_weight})
# inner nodes (# nodes connect to it): {len(self.inner_nodes)}
# outer nodes (# nodes it connects to): {len(self.outer_nodes)}
Genetic position: {self.pos}
==========================='''
        return output
    def __repr__(self):
        return self.__str__()
    def dist(self,another_node):
        return np.abs(self.pos - another_node.pos)
    