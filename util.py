def construct_possible_haps(genotypes):
    all_possible_haps = set()
    all_possible_dips = set()
    num_samples,num_SNPs = genotypes.shape
    for i in range(num_samples):
        this_sample_possible_haps = [()]
        for j in range(num_SNPs):
            if genotypes[i,j] == 2:
                for k in range(len(this_sample_possible_haps)):
                    this_sample_possible_haps[k]+=(1,)
            elif genotypes[i,j] == 0:
                for k in range(len(this_sample_possible_haps)):
                    this_sample_possible_haps[k]+=(0,)
            else:
                new_this_sample_possible_haps = []
                previous_all_haps = this_sample_possible_haps.copy()
                for k in range(len(previous_all_haps)):
                    previous_all_haps[k]+=(1,)
                new_this_sample_possible_haps += previous_all_haps
                previous_all_haps = this_sample_possible_haps.copy()
                for k in range(len(previous_all_haps)):
                    previous_all_haps[k]+=(0,)
                new_this_sample_possible_haps += previous_all_haps
                this_sample_possible_haps = new_this_sample_possible_haps
        if len(this_sample_possible_haps) == 1:
            hap = this_sample_possible_haps[0]
            all_possible_haps.add(hap)
            all_possible_dips.add((hap,hap))
        else:
            for l in range(len(this_sample_possible_haps)//2):
                hap_1 = this_sample_possible_haps[l] 
                hap_2 = this_sample_possible_haps[len(this_sample_possible_haps)-l-1] 
                all_possible_haps.add(hap_1)
                all_possible_haps.add(hap_2)
                if hap_1<= hap_2:
                    all_possible_dips.add((hap_1,hap_2))
                else:
                    all_possible_dips.add((hap_2,hap_1))
#     all_possible_dips = set(all_possible_dips)
#     reverse_all_possible_haps_index = {}
#     for hap,index in all_possible_haps_index.items():
#         reverse_all_possible_haps_index[index] = hap
    return list(all_possible_dips),list(all_possible_haps)