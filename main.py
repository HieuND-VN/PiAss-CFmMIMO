import numpy as np
import torch
import torch.nn.functional as F
import pennylane as qml
from model import HQCNN, train_model
from benchmark import MLPModel, CNNModel, greedy_assignment, random_assignment, master_AP_assignment, calculate_dl_rate
from dataset import one_time_dataset

num_ap = 40
num_ue = 20
tau_p = 10
batch_size = 16
n_qubits = 8
weight_shapes = {
    "weights_0": 3,
    "weights_1": 3,
    "weights_2": 1,
    "weights_3": 1,
    "weights_4": 1,
    "weights_5": 3,
    "weights_6": 3,
    "weights_7": 1,
    "weights_8": 1,
}
Pooling_out = [1, 3, 5, 7]
dev = qml.device("default.qubit", wires=n_qubits)


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


# Unitary Ansatz for Pooling Layer
def Pooling_ansatz1(weights_0, weights_1, wires):  # 2 params
    qml.CRZ(weights_0, wires=[wires[0], wires[1]])
    qml.PauliX(wires=wires[0])
    qml.CRX(weights_1, wires=[wires[0], wires[1]])


@qml.qnode(dev)
def qnode(inputs, weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, weights_7,
          weights_8):  # , weights_9, weights_10, weights_11, weights_12, weights_13, weights_14, weights_15, weights_16, weights_17
    qml.AngleEmbedding(inputs, wires=range(n_qubits))

    # QCNN
    # --------------------------------------------------------- Convolutional Layer1 ---------------------------------------------------------#
    U_SU4(weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, wires=[0, 1])
    U_SU4(weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, wires=[2, 3])
    U_SU4(weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, wires=[4, 5])
    U_SU4(weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, wires=[6, 7])

    U_SU4(weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, wires=[1, 2])
    U_SU4(weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, wires=[3, 4])
    U_SU4(weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, wires=[5, 6])
    U_SU4(weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, wires=[7, 0])

    # --------------------------------------------------------- Pooling Layer1 ---------------------------------------------------------#
    ## Pooling Circuit  Block 2 weights_7, weights_8
    Pooling_ansatz1(weights_7, weights_8, wires=[0, 1])
    Pooling_ansatz1(weights_7, weights_8, wires=[2, 3])
    Pooling_ansatz1(weights_7, weights_8, wires=[4, 5])
    Pooling_ansatz1(weights_7, weights_8, wires=[6, 7])

    result = [qml.expval(qml.PauliZ(wires=i)) for i in Pooling_out]
    return result

test_data_numpy, train_dataloader, test_dataloader, train_dataloader_CNN, test_dataloader_CNN = one_time_dataset(num_ap, num_ue, tau_p, batch_size)

model_HQCNN = HQCNN(num_ap, num_ue, tau_p, qnode, weight_shapes)
model_MLP = MLPModel(num_ap, num_ue, tau_p)
model_CNN = CNNModel(num_ap, num_ue, tau_p)

train_model(model_HQCNN, train_dataloader, test_dataloader, batch_size, num_ap, num_ue, tau_p, lr=0.01)
train_model(model_MLP, train_dataloader, test_dataloader, batch_size, num_ap, num_ue, tau_p, lr=0.01)
train_model(model_CNN, train_dataloader_CNN, test_dataloader_CNN, batch_size, num_ap, num_ue, tau_p, lr=0.01)


beta_test_flatten = test_data_numpy[0] #[num_ap*num_ue, ]
beta_test_reshape = beta_test_flatten.reshape(num_ap,num_ue) #[num_ap, num_ue]
beta_flatten_test = np.expand_dims(beta_test_flatten, axis = 0) #[1, num_ap*num_ue]
beta_reshape_test = np.expand_dims(beta_test_reshape, axis = 0) #[1. num_ap, num_ue]

xs = torch.tensor(beta_flatten_test, dtype = torch.float32)
xs_CNN = torch.tensor(beta_reshape_test, dtype = torch.float32)

output_HQCNN = model_HQCNN(xs)
output_MLP = model_MLP(xs)
output_CNN = model_CNN(xs_CNN)

pilot_probs_HQCNN = F.softmax(output_HQCNN.reshape(num_ue, tau_p), dim = -1)
pilot_index_HQCNN = torch.argmax(pilot_probs_HQCNN, dim = -1)
pilot_index_HQCNN = pilot_index_HQCNN.cpu().numpy()

pilot_probs_MLP = F.softmax(output_MLP.reshape(num_ue, tau_p), dim = -1)
pilot_index_MLP = torch.argmax(pilot_probs_MLP, dim = -1)
pilot_index_MLP = pilot_index_MLP.cpu().numpy()

pilot_probs_CNN = F.softmax(output_CNN.reshape(num_ue, tau_p), dim = -1)
pilot_index_CNN = torch.argmax(pilot_probs_CNN, dim = -1)
pilot_index_CNN = pilot_index_CNN.cpu().numpy()

rate_HQCNN = calculate_dl_rate(beta_test_reshape, pilot_index_HQCNN, num_ap, num_ue, tau_p)
rate_MLP = calculate_dl_rate(beta_test_reshape, pilot_index_MLP)
rate_CNN = calculate_dl_rate(beta_test_reshape, pilot_index_CNN)

pilot_init = np.random.randint(tau_p, size = num_ue)
rate_greedy, pilot_index_greedy = greedy_assignment(beta_test_reshape, num_ap, num_ue, tau_p, pilot_init)
rate_random, pilot_index_random = random_assignment(beta_test_reshape, pilot_init, num_ap, num_ue, tau_p)
rate_masterAP, pilot_index_masterAP = master_AP_assignment(beta_test_reshape, num_ap, num_ue, tau_p)

print(f'HQCNN: {rate_HQCNN}\nPilot_index: {pilot_index_HQCNN}\n---------------------------------')
print(f'MLP: {rate_MLP}\nPilot_index: {pilot_index_MLP}\n---------------------------------')
print(f'CNN: {rate_CNN}\nPilot_index: {pilot_index_CNN}\n---------------------------------')
print(f'Random: {rate_random}\nPilot_index: {pilot_index_random}\n---------------------------------')
print(f'Greedy: {rate_greedy}\nPilot_index: {pilot_index_greedy}\n---------------------------------')
print(f'Master-AP: {rate_masterAP}\nPilot_index: {pilot_index_masterAP}\n---------------------------------')
