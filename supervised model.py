import numpy as np
import random
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import pennylane as qml
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time

D = 1  # in kilometer
d1 = 0.05  # in kilometer
d0 = 0.01  # in kilometer
h_ap = 15  # in meter
h_ue = 1.7  # in meter
B = 20  # in MHz
f = 1900  # in MHz
L = 46.3 + 33.9 * np.log10(f) - 13.82 * np.log10(h_ap) - (1.1 * np.log10(f) - 0.7) * h_ue + (
        1.56 * np.log10(f) - 0.8)
P_d = 0.2  # downlink power: 200 mW
p_u = 0.1  # uplink power: 100 mW
p_p = 0.1  # pilot power: 100mW
noise_figure = 9  # dB
T = 290  # noise temperature in Kelvin
noise_power = (B * 10 ** 6) * (1.381 * 10 ** (-23)) * T * (10 ** (noise_figure / 10))  # Thermal noise in W
rho = 1 / noise_power
rho_d = P_d / noise_power
rho_u = rho_p = p_u / noise_power
sigma_shd = 8  # in dB

def calculate_dl_rate(beta, pilot_index, num_ap, num_ue, tau_p, rho_d, rho_p):
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

    etaa = 1 / np.sum(gamma, axis=1)
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


def master_AP_assignment(beta, num_ap, num_ue, tau_p, rho_d, rho_p):
    pilot_index = -1 * np.ones(num_ue)
    pilot_index[0:tau_p] = np.random.permutation(tau_p)
    beta_transpose = np.transpose(beta)
    for k in range(tau_p, num_ue):
        m_star = np.argmax(beta_transpose[k])  # master AP
        interference = np.zeros(tau_p)
        for tau in range(tau_p):
            interference[tau] = np.sum(beta_transpose[pilot_index == tau, m_star])
        pilot_index[k] = np.argmin(interference)

    rate_list = calculate_dl_rate(beta, pilot_index, num_ap, num_ue, tau_p, rho_d, rho_p)
    sum_rate = np.sum(rate_list)
    pilot_index = pilot_index.astype(int)
    return sum_rate, pilot_index


def supervised_data(number_AP, number_UE, pilot, batch_sz, num_train, num_test):

    num_ap = number_AP
    num_ue = number_UE
    tau_p = pilot

    total_sample = num_train + num_test
    data = []
    rate = []
    data_CNN = []
    for i in range(total_sample):
        AP_position = np.random.uniform(-D, D, size=(num_ap, 2))
        UE_position = np.random.uniform(-D, D, size=(num_ue, 2))

        AP_expanded = AP_position[:, np.newaxis, :]  # Shape: (num_ap, 1, 2)
        UE_expanded = UE_position[np.newaxis, :, :]

        distanceUE2AP = np.sqrt(np.sum((AP_expanded - UE_expanded) ** 2, axis=2))
        pathloss = np.zeros_like(distanceUE2AP)
        pathloss[(distanceUE2AP < d0)] = -L - 15 * np.log10(d1) - 20 * np.log10(d0)
        pathloss[(distanceUE2AP >= d0) & (distanceUE2AP <= d1)] = -L - 15 * np.log10(d1) - 20 * np.log10(
            distanceUE2AP[(distanceUE2AP >= d0) & (distanceUE2AP <= d1)])
        pathloss[(distanceUE2AP > d1)] = -L - 35 * np.log10(distanceUE2AP[(distanceUE2AP > d1)]) + np.random.normal(0,
                                                                                                                    1) * sigma_shd
        beta = 10 ** (pathloss / 10)

        beta_norm = beta * rho
        sum_rate, pilot_index = master_AP_assignment(beta, num_ap, num_ue, tau_p, rho_d, rho_p)
        # pilot_index_onehot = np.eye(tau_p)[pilot_index]

        # data.append(np.concatenate((beta_norm.flatten(), pilot_index_onehot.flatten())))
        data.append(np.concatenate((beta_norm.flatten(), pilot_index)))
        # data_CNN.append(np.concatenate((beta_norm, pilot_index)))
        rate.append(sum_rate)

    rate = np.array(rate)
    train_avg_rate = np.mean(rate[:num_train])
    test_avg_rate = np.mean(rate[num_train:])
    data = np.array(data)
    # data_CNN = np.array(data_CNN)
    train_data = data[:num_train, :]
    x_train = train_data[:, :num_ap * num_ue]
    y_train = train_data[:, num_ap * num_ue:]

    test_data = data[num_train:, :]
    x_test = test_data[:, :num_ap * num_ue]
    y_test = test_data[:, num_ap * num_ue:]

    X_train = torch.tensor(x_train).float()
    Y_train = torch.tensor(y_train).float()
    Y_train = Y_train.long()

    X_test = torch.tensor(x_test).float()
    Y_test = torch.tensor(y_test).float()
    Y_test = Y_test.long()

    data_train = list(zip(X_train, Y_train))
    data_test = list(zip(X_test, Y_test))
    train_dataloader = DataLoader(data_train, batch_size=batch_sz, shuffle=False, drop_last=True)
    test_dataloader = DataLoader(data_test, batch_size=batch_sz, shuffle=True, drop_last=True)
    # return train_data, test_data, train_avg_rate, test_avg_rate, x_train, y_train
    return train_dataloader, test_dataloader, train_avg_rate, test_avg_rate


n_qubits = 8
dev = qml.device("default.qubit", wires=n_qubits)
weight_shapes_QC = {
    "weights_0": 3,  # shape for weights_0
    "weights_1": 3,  # shape for weights_1
    "weights_2": 1,  # shape for weights_2
    "weights_3": 1,  # shape for weights_3
    "weights_4": 1,  # shape for weights_4
    "weights_5": 3,  # shape for weights_5
    "weights_6": 3,  # shape for weights_6
}


def U_SU4(weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6,
          wires):  # 15 params, Convolutional Circuit 10
    qml.U3(*weights_0, wires=wires[0])
    qml.U3(*weights_1, wires=wires[1])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.RY(weights_2, wires=wires[0])
    qml.RZ(weights_3, wires=wires[1])
    qml.CNOT(wires=[wires[1], wires[0]])
    qml.RY(weights_4, wires=wires[0])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.U3(*weights_5, wires=wires[0])
    qml.U3(*weights_6, wires=wires[1])


@qml.qnode(dev)
def qnode_Ang_QC(inputs, weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6):
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    # qml.AmplitudeEmbedding(inputs, wires=range(n_qubits), pad_with=0.0, normalize=True)
    # --------------------------------------------------------- Convolutional Layer1 ---------------------------------------------------------#
    U_SU4(weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, wires=[0, 1])
    U_SU4(weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, wires=[2, 3])
    U_SU4(weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, wires=[4, 5])
    U_SU4(weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, wires=[6, 7])
    U_SU4(weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, wires=[1, 2])
    U_SU4(weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, wires=[3, 4])
    U_SU4(weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, wires=[5, 6])
    U_SU4(weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, wires=[7, 0])
    result = [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]
    return result


class HQCNN_Ang_noQP(nn.Module):
    def __init__(self, num_ap, num_ue, tau_p, qnode, weight_shapes):
        super(HQCNN_Ang_noQP, self).__init__()
        self.clayer_1 = nn.Linear(num_ap * num_ue, 8)
        self.qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)
        self.clayer_2 = nn.Linear(8, num_ue * tau_p)

    def forward(self, x):
        x = self.clayer_1(x)
        x = self.qlayer(x)
        x = self.clayer_2(x)
        return x
class MLPModel(nn.Module):
    def __init__(self, num_ap, num_ue, tau_p):
        super(MLPModel, self).__init__()
        self.fc_1 = nn.Linear(num_ap * num_ue, 8)
        self.fc_2 = nn.Linear(8,8)
        self.fc_3 = nn.Linear(8, num_ue * tau_p)

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
        self.fc2 = nn.Linear(32, num_ue * tau_p)
        self.num_ue = num_ue
        self.tau_p = tau_p
    def forward(self, x):
        # Reshape input to 2D for convolution
        batch_size = x.size(0)
        x = x.view(batch_size, 1, -1, self.num_ue)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(batch_size, -1)
        if self.fc1 is None:
            self.fc1 = nn.Linear(x.size(1), 32)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
def calculate_dl_rate_model(xs, predict_index, num_ap, num_ue, tau_p, rho_d, rho_p):
    xs = xs.numpy()
    predict_index = predict_index.numpy()
    rate_list = np.zeros(len(xs))
    for i, beta_norm in enumerate(xs):
        beta = beta_norm.reshape(num_ap, num_ue) / rho
        rate_list[i] = np.sum(calculate_dl_rate(beta, predict_index[i], num_ap, num_ue, tau_p, rho_d, rho_p))

    return np.mean(rate_list)

def train_model(model, train_dataloader, test_dataloader, num_ap, num_ue, tau_p, batch_size, num_epoch, is_CNN = False, lr=0.01):
    opt = optim.Adam(model.parameters(), lr=lr)
    loss = torch.nn.CrossEntropyLoss()
    epochs = num_epoch
    train_loss = []
    test_loss = []
    train_rate = []
    test_rate = []
    for epoch in range(epochs):
        running_loss_train = 0
        rate_mean_tr = 0
        running_loss_test = 0
        rate_mean_ts = 0
        model.train()
        for i, (xs, ys) in enumerate(train_dataloader):
            if is_CNN:
                xs = xs.reshape(batch_size, num_ap, num_ue)
            opt.zero_grad()
            output_train = model(xs)
            prediction_tr = output_train.reshape(batch_size, tau_p, num_ue)
            predict_prob = F.softmax(prediction_tr, dim = 1)
            predict_index = torch.argmax(predict_prob, dim = 1)
            rate_mean_tr += calculate_dl_rate_model(xs, predict_index, num_ap, num_ue, tau_p, rho_d, rho_p)
            # if i == 0:
            #     print(xs.shape)
            #     print(predict_prob.shape)
                # print(predict_index)
            # prediction_tr = F.softmax(output_train, dim=-1)
            # pilot_index = torch.argmax(prediction_tr, dim=-1)
            # prediction_tr = prediction_tr.reshape(batch_size,num_ue*tau_p)
            # prediction_tr = output_train.permute(0, 2, 1)
            # if i == 0:
            #     print(prediction_tr.shape)
            #     print(ys.shape)
            #     print(prediction_tr[0])
            #     print(ys[0])
            #     print(type(prediction_tr))
            #     print(type(ys))
            loss_train = loss(prediction_tr, ys)
            loss_train.backward()
            opt.step()
            running_loss_train += loss_train.item()
        avg_loss_train = running_loss_train / len(train_dataloader)
        avg_rate_train = rate_mean_tr / len(train_dataloader)
        model.eval()
        train_loss.append(avg_loss_train)
        train_rate.append(avg_rate_train)
        with torch.no_grad():
            for i, (xt, yt) in enumerate(test_dataloader):
                output_test = model(xt)
                prediction_ts = output_test.reshape(batch_size, tau_p, num_ue)
                predict_prob = F.softmax(prediction_ts, dim=1)
                predict_index = torch.argmax(predict_prob, dim=1)
                rate_mean_ts += calculate_dl_rate_model(xt, predict_index, num_ap, num_ue, tau_p, rho_d, rho_p)
                # prediction_ts = F.softmax(output_test, dim=-1)
                # prediction_ts = prediction_ts.reshape(batch_size, num_ue*tau_p)
                # loss_evaluated_test = loss_function_PA(xt, output_QCNN_test, batch_sz, num_ap, num_ue, tau_p)
                loss_test = loss(prediction_ts, yt)
                running_loss_test += loss_test.item()
            avg_loss_test = running_loss_test / len(test_dataloader)
            avg_rate_test = rate_mean_ts / len(test_dataloader)
            test_loss.append(avg_loss_test)
            test_rate.append(avg_rate_test)
        if epoch%5==0:
            print(
                f'EPOCH: {epoch}/{epochs}: \n\tTrain: {avg_loss_train: .4f}\n\tTest: {avg_loss_test: .4f}\n\tTrain_rate: {avg_rate_train: .4f}\n\tTest_rate: {avg_rate_test: .4f}')
    train_loss = np.array(train_loss)
    test_loss = np.array(test_loss)
    train_rate = np.array(train_rate)
    test_rate = np.array(test_rate)
    return train_loss, test_loss, train_rate, test_rate


M = 40
K = 20
tau_p = 10
batch_sz = 64
num_train = 320*5
num_test = 80*4
train_dataloader, test_dataloader, train_avg_rate, test_avg_rate = supervised_data(M, K, tau_p, batch_sz, num_train,
                                                                                   num_test)
model_HQCNN_Ang_QC = HQCNN_Ang_noQP(M, K, tau_p, qnode_Ang_QC, weight_shapes_QC)
model_MLP = MLPModel(M, K, tau_p)
model_CNN = CNNModel(M, K, tau_p)
print(train_avg_rate)
print(test_avg_rate)

HQCNN_start = time.perf_counter()
print(f'HQCNN')
trloss_HQCNN, teloss_HQCNN, trrate_HQCNN, terate_HQCNN = train_model(model_HQCNN_Ang_QC, train_dataloader, test_dataloader, M, K, tau_p, batch_sz, num_epoch=50,is_CNN = False, lr=0.01)
HQCNN_end = time.perf_counter()
np.savez('hqcnn_training_results.npz',
         trloss_HQCNN=trloss_HQCNN,
         teloss_HQCNN=teloss_HQCNN,
         trrate_HQCNN=trrate_HQCNN,
         terate_HQCNN=terate_HQCNN)

print("HQCNN--Results saved successfully.")
MLP_start = time.perf_counter()
print(f'MLP')
trloss_MLP, teloss_MLP, trrate_MLP, terate_MLP = train_model(model_MLP, train_dataloader, test_dataloader, M, K, tau_p, batch_sz, num_epoch=50, is_CNN = False, lr=0.01)
MLP_end = time.perf_counter()
np.savez('mlp_training_results.npz',
         trloss_MLP=trloss_MLP,
         teloss_MLP=teloss_MLP,
         trrate_MLP=trrate_MLP,
         terate_MLP=terate_MLP)

print("MLP--Results saved successfully.")
CNN_start = time.perf_counter()
print(f'CNN')
trloss_CNN, teloss_CNN, trrate_CNN, terate_CNN = train_model(model_CNN, train_dataloader, test_dataloader, M, K, tau_p, batch_sz, num_epoch=50,is_CNN = True , lr=0.01)
CNN_end = time.perf_counter()
np.savez('cnn_training_results.npz',
         trloss_CNN=trloss_CNN,
         teloss_CNN=teloss_CNN,
         trrate_CNN=trrate_CNN,
         terate_CNN=terate_CNN)

print("CNN--Results saved successfully.")

def plot_loss(trloss_HQCNN, trloss_MLP, trloss_CNN):
    plt.plot(trloss_HQCNN, label='HQCNN')
    plt.plot(trloss_MLP, label='MLP')
    plt.plot(trloss_CNN, label='CNN')
    # plt.plot(teloss, label='Test loss')
    plt.legend()

    # Add x-axis and y-axis labels
    plt.xlabel('Iteration')
    plt.ylabel('Loss value')

    # Display the plot
    plt.show()

def plot_rate(threshold, result_HQCNN, result_MLP, result_CNN, is_train):
    plt.plot(result_HQCNN, label='HQCNN')
    plt.plot(result_MLP, label='MLP')
    plt.plot(result_CNN, label='CNN')
    thr = threshold*np.ones(len(result_HQCNN))
    plt.plot(thr, label='Master-AP')
    plt.legend()
    if is_train:
        plt.title('Training')
    else:
        plt.title('Testing')
    plt.xlabel('Iteration')
    plt.ylabel('Total system rate')
    plt.show()

plot_loss(trloss_HQCNN, trloss_MLP, trloss_CNN)
plot_rate(train_avg_rate, trrate_HQCNN, trrate_MLP, trrate_CNN, is_train=True)
plot_rate(test_avg_rate,  terate_HQCNN, terate_MLP, terate_CNN, is_train=False)

hqcnn_time = HQCNN_end-HQCNN_start
mlp_time   = MLP_end-MLP_start
cnn_time   = CNN_end-CNN_start
print(f'TIME CONSUMPTION:\n\t--HQCNN: {HQCNN_end-HQCNN_start}'
      f'\n\t--MLP: {MLP_end-MLP_start}'
      f'\n\t--CNN: {CNN_end-CNN_start}')


np.savez('system_training_results.npz',
         train_avg_rate = train_avg_rate,
         test_avg_rate = test_avg_rate,
         hqcnn_time= hqcnn_time,
        mlp_time= mlp_time,
        cnn_time= cnn_time)

print("CNN--Results saved successfully.")
