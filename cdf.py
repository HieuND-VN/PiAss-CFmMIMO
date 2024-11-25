import numpy as np
import matplotlib.pyplot as plt

# Load the saved arrays
sum_rate_lists = np.load('sr-list.npz')

valid_rpas =sum_rate_lists['valid_rpas']
valid_gpas =sum_rate_lists['valid_gpas']
valid_mapas=sum_rate_lists['valid_mapas']
valid_hqcnn=sum_rate_lists['valid_hqcnn']
valid_mlp  =sum_rate_lists['valid_mlp']
valid_cnn  =sum_rate_lists['valid_cnn']

print(f'valid_rpas : \n{valid_rpas}\n')
print(f'valid_gpas : \n{valid_gpas}\n')
print(f'valid_mapas: \n{valid_mapas}\n')
print(f'valid_hqcnn: \n{valid_hqcnn}\n')
print(f'valid_mlp  : \n{valid_mlp}\n')
print(f'valid_cnn  : \n{valid_cnn}\n')
