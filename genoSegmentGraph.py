from util import construct_possible_haps
import numpy as np
class genoSegmentGraph(object):
    '''
    S_g graph
    '''
    def __init__(self,geno,B):
        self.nodes = {}
        self.B = B
        self.build_geno_graph(geno)
    def __str__(self):
        nodes_count = 0
        for marker,nodes in self.nodes.items():
            nodes_count += len(nodes)
        
        output = f'''===========================
Number of markers: {len(self.nodes)}
Number of nodes (# segment haplotypes(~=B) x # markers): {nodes_count}
==========================='''
        return output
    def __repr__(self):
        return self.__str__()
    def build_geno_graph(self,geno):
        '''
        B: number of hetero markers in each segment
        geno: genotype (num_of_markers)
        '''
        B = self.B
        # split the genotype by segments
        splitted_geno = np.split(geno, np.where(geno == 1)[0][:-1]+1)
        segments = [np.concatenate(splitted_geno[i:i+B]) for i in range(0,len(splitted_geno),B)]
        marker_id = geno.shape[0]
        after_marker_nodes = []
        node_id = 0
        for segment in segments[::-1]:
            segment_geno = np.array([segment])
            _,haps = construct_possible_haps(segment_geno)
            last_marker = True
            for i in list(range(segment_geno.shape[1]))[::-1]:
                marker_id -= 1
                marker_nodes = []
                for hap in haps:
                    node = genoSegmentNode(node_id,marker_id,hap,hap[i],segment)
                    node_id += 1
                    marker_nodes.append(node)
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
                if last_marker and after_marker_nodes == []:
                    leaves_nodes = marker_nodes
                after_marker_nodes = marker_nodes
                last_marker = False
                self.nodes[marker_id] = after_marker_nodes
class genoSegmentNode(object):
    def __init__(self,node_id,marker,haplotype,allele,segment):
        self.id = node_id
        self.marker =  marker
        self.haplotype = haplotype
        self.allele = allele
        self.segment = segment
        self.type = None
        self.inner_nodes = []
        self.outer_nodes = []
    def __str__(self):
        output = f'''===========================
Geno segment Node: represents a possible haplotype state for this marker in this sample genotype
--------------------------
Node id: {self.id}
Marker id: {self.marker}
Haplotype: {self.haplotype}
Segment genotype: {self.segment}
Allele: {self.allele}
Type(it connects to another segment[inter] or connects to the node in the same segment[intra]): {self.type}
# inner nodes (# nodes connect to it): {len(self.inner_nodes)}
# outer nodes (# nodes it connects to): {len(self.outer_nodes)}
==========================='''
        return output
    def __repr__(self):
        return self.__str__()