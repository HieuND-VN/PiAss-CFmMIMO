import numpy as np

cdf_EPAS = np.load('sr-list_5P_9UE.npz')

valid_rpas                      = cdf_EPAS['valid_rpas']
valid_gpas                      = cdf_EPAS['valid_gpas']
valid_mapas                     = cdf_EPAS['valid_mapas']
valid_hqcnn                     = cdf_EPAS['valid_hqcnn']
valid_mlp                       = cdf_EPAS['valid_mlp']
valid_cnn                       = cdf_EPAS['valid_cnn']
exhaustive_search_list_value    = cdf_EPAS['exhaustive_search_list_value']
exhaustive_search_list_scheme   = cdf_EPAS['exhaustive_search_list_scheme']

print(exhaustive_search_list_value)