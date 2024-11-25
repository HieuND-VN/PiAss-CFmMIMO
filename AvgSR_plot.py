import matplotlib.pyplot as plt
import numpy as np

# Data
M_values = np.array([30, 35, 40, 45])
HQCNN_PAS =     np.array([97.5239,  115.4418,   124.8665,   132.7494])
CNN_PAS =       np.array([100.1613, 115.7567,   125.5104,   133.1001])
MasterAP_PAS =  np.array([78.931,   90.9789,    98.1983,    111.4534])
MLP_PAS =       np.array([77.5024,  89.6885,    96.5098,    109.6014])
Greedy_PAS =    np.array([72.8708,  85.1444,    91.0401,    103.6669])
Random_PAS =    np.array([60.1269,  64.987,     67.8984,    77.6367])

# Plot
# plt.figure(figsize=(10, 4))
plt.plot(M_values, HQCNN_PAS, marker='o', label='HQCNN')
plt.plot(M_values, CNN_PAS, marker='^', label='CNN')
plt.plot(M_values, MasterAP_PAS, marker='*', label='Master AP')
plt.plot(M_values, MLP_PAS, marker='d', label='MLP')
plt.plot(M_values, Greedy_PAS, marker='p', label='Greedy')
plt.plot(M_values, Random_PAS, marker='s', label='Random')
# Customization

plt.ylim(29, 46)
plt.ylim(59, 135)
plt.xlabel("Number of APs", fontsize=16)
plt.ylabel("Average Sum-rate (Mbps)", fontsize=16)
plt.xticks(M_values, fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True)

# Move legend outside
# plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)

plt.legend()

# Adjust layout to fit legend
# plt.tight_layout(rect=[0, 0, 0.85, 1])

# Show plot
plt.savefig('unsupervised_sumrate.png', bbox_inches='tight', dpi=600)
plt.show()

fig = plt.gcf()  # Get current figure
print(fig.get_size_inches())