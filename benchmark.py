import numpy as np
import torch.nn.functional as F
import torch.nn as nn

rho_d = 1
rho_p = 1 / 2


class MLPModel(nn.Module):
    def __init__(self, num_ap, num_ue, tau_p,n_qubits):
        super(MLPModel, self).__init__()
        self.fc_1 = nn.Linear(num_ap * num_ue, n_qubits)
        self.fc_2 = nn.Linear(n_qubits, n_qubits)
        self.fc_3 = nn.Linear(n_qubits, num_ue * tau_p)

    def forward(self, x):
        x = self.fc_1(x)
        x = self.fc_2(x)
        x = self.fc_3(x)
        return x


class CNNModel(nn.Module):
    def __init__(self, num_ap, num_ue, tau_p):
        super(CNNModel, self).__init__()
        # Define convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # Placeholder for fully connected layers
        self.fc1 = None  # To be initialized dynamically based on input size
        self.fc_hidden1 = nn.Linear(128, 64)  # First hidden layer after fc1
        self.fc_hidden2 = nn.Linear(64, 32)  # Second hidden layer
        self.fc2 = nn.Linear(32, num_ue * tau_p)  # Final output layer
        self.num_ue = num_ue
        self.tau_p = tau_p

    def forward(self, x):
        # Reshape input to 2D for convolution
        batch_size = x.size(0)
        x = x.view(batch_size, 1, -1, self.num_ue)
        # Convolutional layers
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        # Flatten the output
        x = x.view(batch_size, -1)  # Automatically calculate the flatten size
        # Initialize the first fully connected layer dynamically
        if self.fc1 is None:
            self.fc1 = nn.Linear(x.size(1), 128)
        # Fully connected layers with two hidden layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc_hidden1(x))
        x = F.relu(self.fc_hidden2(x))
        x = self.fc2(x)
        return x


def calculate_dl_rate(beta, pilot_index, num_ap, num_ue, tau_p):
    M = num_ap
    K = num_ue
    rows = np.arange(K)
    pilot_index = pilot_index.astype(int)
    phi = np.zeros((K, tau_p), dtype=int)
    phi[rows, pilot_index] = 1
    c = np.zeros((M, K))
    numerator_c = np.zeros((M, K))
    denominator_c = np.zeros((M, K))
    for m in range(M):
        for k in range(K):
            numerator_c[m, k] = np.sqrt(tau_p * rho_p) * beta[m, k]
            inner_sum = 0
            for j in range(K):
                inner_sum += beta[m, j] * (np.dot(phi[k].T, phi[j]))
            denominator_c[m, k] = tau_p * rho_p * inner_sum + 1
            c[m, k] = numerator_c[m, k] / denominator_c[m, k]

    gamma = np.sqrt(tau_p * rho_p) * beta * c

    etaa = 1/np.sum(gamma, axis = 1)
    eta = np.tile(etaa[:, None], (1, num_ue))
    sinr = np.zeros(K)
    for k in range(K):
        numerator_sum = 0
        for m in range(M):
            numerator_sum += np.sqrt(eta[m, k]) * gamma[m, k]
        numerator = rho_d * numerator_sum ** 2

        sum_term_1 = 0
        for j in range(K):
            sum_term_2 = 0
            if j != k:
                for m in range(M):
                    sum_term_2 += np.sqrt(eta[m, j]) * gamma[m, j] * beta[m, k] / beta[m, j]
            sum_term_1 = sum_term_2 ** 2 * (np.dot(phi[k].T, phi[j]))
        UI = rho_d * sum_term_1

        sum_term_3 = 0
        for j in range(K):
            for m in range(M):
                sum_term_3 = eta[m, j] * gamma[m, j] * beta[m, k]
        BU = rho_d * sum_term_3
        denominator = UI + BU + 1
        sinr[k] = numerator / denominator

    rate = np.log2(1 + sinr)
    return rate


def greedy_assignment(beta, num_ap, num_ue, tau_p, pilot_init):
    N = num_ue
    pilot_index = pilot_init
    pilot_index = pilot_index.astype(int)
    for n in range(N):
        dl_rate = calculate_dl_rate(beta, pilot_index, num_ap, num_ue, tau_p)
        k_star = np.argmin(dl_rate)
        sum_beta = np.zeros(tau_p)
        for tau in range(tau_p):
            for m in range(num_ap):
                for k in range(num_ue):
                    if (k != k_star) and (pilot_index[k] == tau):
                        sum_beta[tau] += beta[m, k]
        pilot_index[k_star] = np.argmin(sum_beta)

    rate_list = calculate_dl_rate(beta, pilot_index, num_ap, num_ue, tau_p)
    sum_rate = np.sum(rate_list)
    return sum_rate , pilot_index


def master_AP_assignment(beta, num_ap, num_ue, tau_p):
    pilot_index = -1 * np.ones(num_ue)
    pilot_index[0:tau_p] = np.random.permutation(tau_p)
    beta_transpose = np.transpose(beta)
    for k in range(tau_p, num_ue):
        m_star = np.argmax(beta_transpose[k])  # master AP
        interference = np.zeros(tau_p)
        for tau in range(tau_p):
            interference[tau] = np.sum(beta_transpose[pilot_index == tau, m_star])
        pilot_index[k] = np.argmin(interference)

    rate_list = calculate_dl_rate(beta, pilot_index, num_ap, num_ue, tau_p)
    sum_rate = np.sum(rate_list)
    pilot_index = pilot_index.astype(int)
    return sum_rate , pilot_index


def random_assignment(beta, pilot_index, num_ap, num_ue, tau_p):
    rate_list = calculate_dl_rate(beta, pilot_index, num_ap, num_ue, tau_p)
    sum_rate = np.sum(rate_list)
    return sum_rate, pilot_index


def greedy_assignment_1(beta, num_ap, num_ue, tau_p, pilot_init):
    N = 3
    pilot_index = pilot_init
    pilot_index = pilot_index.astype(int)
    for n in range(N):
        dl_rate = calculate_dl_rate(beta, pilot_index, num_ap, num_ue, tau_p)
        k_star = np.argmin(dl_rate)
        sum_beta = np.zeros(tau_p)
        for tau in range(tau_p):
            for m in range(num_ap):
                for k in range(num_ue):
                    if (k != k_star) and (pilot_index[k] == tau):
                        sum_beta[tau] += beta[m, k]
        pilot_index[k_star] = np.argmin(sum_beta)

    rate_list = calculate_dl_rate(beta, pilot_index, num_ap, num_ue, tau_p)
    sum_rate = np.sum(rate_list)
    return sum_rate #, pilot_index


def master_AP_assignment_1(beta, num_ap, num_ue, tau_p):
    pilot_index = -1 * np.ones(num_ue)
    pilot_index[0:tau_p] = np.random.permutation(tau_p)
    beta_transpose = np.transpose(beta)
    for k in range(tau_p, num_ue):
        m_star = np.argmax(beta_transpose[k])  # master AP
        interference = np.zeros(tau_p)
        for tau in range(tau_p):
            interference[tau] = np.sum(beta_transpose[pilot_index == tau, m_star])
        pilot_index[k] = np.argmin(interference)

    rate_list = calculate_dl_rate(beta, pilot_index, num_ap, num_ue, tau_p)
    sum_rate = np.sum(rate_list)
    pilot_index = pilot_index.astype(int)
    return sum_rate #, pilot_index


def random_assignment_1(beta, pilot_index, num_ap, num_ue, tau_p):
    rate_list = calculate_dl_rate(beta, pilot_index, num_ap, num_ue, tau_p)
    sum_rate = np.sum(rate_list)
    return sum_rate#, pilot_index

def EPAS_assignment_1(beta,pilot_list,num_ap,num_ue,tau_p):
    rate_each_scheme = np.zeros(len(pilot_list))
    # print(f'BENCHMARK: length of pilot_list {len(pilot_list)}')
    for i in range(len(pilot_list)):

        rate_each_scheme[i] = np.sum(calculate_dl_rate(beta, pilot_list[i], num_ap, num_ue, tau_p))

    optimal_value = np.max(rate_each_scheme)
    optimal_scheme = pilot_list[np.argmax(rate_each_scheme)]
    return optimal_value, optimal_scheme
def balance_array(arr, K, tau_p):
    # Create a copy of the array to avoid modifying the original one
    arr_copy = arr.copy()

    # Count the frequency of each value
    counts = np.bincount(arr_copy, minlength=tau_p)

    # Calculate the target frequency for each value
    target_freq = K // tau_p

    # Indices of elements to replace
    to_replace = []

    # Indices of elements that are under-represented
    under_represented = []

    # Identify over-represented and under-represented values
    for i in range(tau_p):
        if counts[i] > target_freq:
            # Too many of this value
            excess = counts[i] - target_freq
            to_replace.extend([i] * excess)
        elif counts[i] < target_freq:
            # Not enough of this value
            deficit = target_freq - counts[i]
            under_represented.extend([i] * deficit)

    # Replace excess elements with under-represented elements
    np.random.shuffle(to_replace)  # Shuffle to avoid bias
    np.random.shuffle(under_represented)

    # Replace elements
    for i in range(len(to_replace)):
        index_to_replace = np.where(arr_copy == to_replace[i])[0][0]  # Find an occurrence
        arr_copy[index_to_replace] = under_represented[i]  # Replace with an under-represented value

    return arr_copy
