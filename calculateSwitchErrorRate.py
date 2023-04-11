import numpy as np
def get_switch_error(prediction,truth):
    SERs = []
    hetero_masking = truth[0] + truth[1] == 1
    hetero_prediction = prediction[0][hetero_masking]
    hetero_truth = truth[0][hetero_masking]
    if hetero_truth.shape[0] == 0:
        return 0
    hap_type = hetero_prediction == hetero_truth
    num_switch = 0
    for i in range(len(hap_type)-1):
        if hap_type[i] != hap_type[i+1]:
            num_switch += 1
    SERs.append(num_switch/hap_type.shape[0])
    
    hetero_prediction = prediction[1][hetero_masking]
    hetero_truth = truth[0][hetero_masking]
    if hetero_truth.shape[0] == 0:
        return 0
    hap_type = hetero_prediction == hetero_truth
    num_switch = 0
    for i in range(len(hap_type)-1):
        if hap_type[i] != hap_type[i+1]:
            num_switch += 1
    SERs.append(num_switch/hap_type.shape[0])
    
    return np.min(SERs)
def calculate_switch_error_rate(results,labels):
    all_SER = []
    for i in range(len(results)):
        SER = get_switch_error(results[i],labels[i])
        all_SER.append(SER)
    return all_SER