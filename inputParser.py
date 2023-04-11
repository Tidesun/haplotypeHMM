import pandas as pd
import numpy as np
def parse_input(hap_f_path):
    df = pd.read_csv(hap_f_path,header=None,sep=' ')
    # make sure sort by position
    df = df.sort_values(by=2)
    genetic_pos = df[2]
    haps = df.loc[:,5:].values
    labels = []
    for i in range(haps.shape[1]//2):
        labels.append((haps[:,i],haps[:,haps.shape[1]//2+i]))
    labels = np.array(labels)
    genos = np.full(shape=(haps.shape[0],haps.shape[1]//2),fill_value=-1,dtype=int)
    ## create genotypes by combining each pair of haplotypes
    for i in range(genos.shape[1]):
        genos[:,i] = haps[:,2*i] + haps[:,2*i+1]
    genos = genos.T
    return genos,genetic_pos,labels