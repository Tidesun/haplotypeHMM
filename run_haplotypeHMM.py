#!/usr/bin/env python
# coding: utf-8

# In[14]:


import pandas as pd
import numpy as np


# In[15]:


hap_f_path = 'examples/CCDG_14151_B01_GRM_WGS_2020-08-05_chr19.filtered.shapeit2-duohmm-phased.2504.43-47Mb.ALL.maf01.haps.gz'


# In[16]:


# reading the haplotype data
df = pd.read_csv(hap_f_path,header=None,sep=' ')


# In[17]:


# make sure sort by position
df = df.sort_values(by=2)
genetic_pos = df[2]
haps = df.loc[:,5:].values


# In[18]:


genos = np.full(shape=(haps.shape[0],haps.shape[1]//2),fill_value=-1,dtype=int)
## create genotypes by combining each pair of haplotypes
for i in range(genos.shape[1]):
    genos[:,i] = haps[:,2*i] + haps[:,2*i+1]
genos = genos.T


# In[19]:


# 2504 samples x 26246 SNPs
# use first 5 SNPs and first 10 samples
print(f'Uses the first 5 SNP and first 10 samples from the {hap_f_path}')
genos.shape
eval_genos = genos[:10,:5]
eval_genetic_pos = genetic_pos.iloc[:5]


# In[20]:


from haplotypeHMM.haplotypeSegmentGraph import haplotypeSegmentGraph
from haplotypeHMM.haplotypeHMM import haplotypeHMM


# In[21]:


# construct the haplotype graph from the genotypes data. 
#Here, I enumerate all the combinations of genotypes data as the haplotypes candidates.
# B is the number of heterozygous markers in each segment
B = 3
hap_graph = haplotypeSegmentGraph(eval_genos,eval_genetic_pos,B)


# In[22]:


# shows the basic information of haplotype graph
print('Basic info of the graph')
print(hap_graph)


# In[37]:


# shows the basic information of node
print('Basic info of a node in the graph')
print(hap_graph.nodes[0][0])


# In[23]:


# construct HMM
hmm = haplotypeHMM(hap_graph)


# In[24]:


# set threads to 1
results = hmm.predict(eval_genos,threads=1)


# In[29]:


# shows the ground truth
labels = []
for i in range(haps.shape[1]//2):
    labels.append((haps[:,2*i],haps[:,2*i+1]))
labels = np.array(labels)
print('Ground truth haplotypes:')
print(labels[:10,:,:5])


# In[30]:


# shows the haplotype pairs prediction for each sample
print('Predication haplotypes:')
print(results)


# In[ ]:




