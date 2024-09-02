import numpy as np
import torch
import torch.nn.functional as F
import pennylane as qml
from model import *
from benchmark import *
from dataset import one_time_dataset
import itertools
import matplotlib.pyplot as plt
import json



weight_shapes_QCQP = {
    "weights_0": 3,  # shape for weights_0
    "weights_1": 3,  # shape for weights_1
    "weights_2": 1,  # shape for weights_2
    "weights_3": 1,  # shape for weights_3
    "weights_4": 1,  # shape for weights_4
    "weights_5": 3,  # shape for weights_5
    "weights_6": 3,  # shape for weights_6
    "weights_7": 1,  # shape for weights_7
    "weights_8": 1,  # shape for weights_8
}

weight_shapes_QC = {
    "weights_0": 3,  # shape for weights_0
    "weights_1": 3,  # shape for weights_1
    "weights_2": 1,  # shape for weights_2
    "weights_3": 1,  # shape for weights_3
    "weights_4": 1,  # shape for weights_4
    "weights_5": 3,  # shape for weights_5
    "weights_6": 3,  # shape for weights_6
}

weight_shapes_QCQC = {
    "weights_0": 3,  # shape for weights_0
    "weights_1": 3,  # shape for weights_1
    "weights_2": 1,  # shape for weights_2
    "weights_3": 1,  # shape for weights_3
    "weights_4": 1,  # shape for weights_4
    "weights_5": 3,  # shape for weights_5
    "weights_6": 3,  # shape for weights_6
    "weights_7": 1,  # shape for weights_7
    "weights_8": 1,  # shape for weights_8
    "weights_9": 1,  # shape for weights_9
    "weights_10": 1,  # shape for weights_10
}
n_qubits = 8
Pooling_out = [1, 3, 5, 7]
dev = qml.device("default.qubit", wires=n_qubits)


def U_SU5(weights_7, weights_8, weights_9, weights_10, wires):
    qml.RY(weights_7, wires=wires[0])
    qml.RY(weights_8, wires=wires[1])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.RY(weights_9, wires=wires[0])
    qml.RY(weights_10, wires=wires[1])
    qml.CNOT(wires=[wires[1], wires[0]])


def U_SU4(weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, wires):  # 15 params, Convolutional Circuit 10
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


def U_SU4_2(weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, wires):  # 15 params, Convolutional Circuit 10
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
def Pooling_ansatz1(weights_0, weights_1, wires):
    qml.CRZ(weights_0, wires=[wires[0], wires[1]])
    qml.PauliX(wires=wires[0])
    qml.CRX(weights_1, wires=[wires[0], wires[1]])


@qml.qnode(dev)
def qnode_Amp_QCQP(inputs, weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, weights_7, weights_8):
    qml.AmplitudeEmbedding(inputs, wires=range(n_qubits), pad_with=0.0, normalize=True)
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
    # Pooling Circuit  Block 2 weights_7, weights_8
    Pooling_ansatz1(weights_7, weights_8, wires=[0, 1])
    Pooling_ansatz1(weights_7, weights_8, wires=[2, 3])
    Pooling_ansatz1(weights_7, weights_8, wires=[4, 5])
    Pooling_ansatz1(weights_7, weights_8, wires=[6, 7])

    result = [qml.expval(qml.PauliZ(wires=i)) for i in Pooling_out]
    return result


@qml.qnode(dev)
def qnode_Ang_QCQP(inputs, weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, weights_7, weights_8):
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
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
    # Pooling Circuit  Block 2 weights_7, weights_8
    Pooling_ansatz1(weights_7, weights_8, wires=[0, 1])
    Pooling_ansatz1(weights_7, weights_8, wires=[2, 3])
    Pooling_ansatz1(weights_7, weights_8, wires=[4, 5])
    Pooling_ansatz1(weights_7, weights_8, wires=[6, 7])

    result = [qml.expval(qml.PauliZ(wires=i)) for i in Pooling_out]
    return result


@qml.qnode(dev)
def qnode_Amp_QC(inputs, weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6):
    qml.AmplitudeEmbedding(inputs, wires=range(n_qubits), pad_with=0.0, normalize=True)
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


@qml.qnode(dev)
def qnode_Ang_QC(inputs, weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6):
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
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


@qml.qnode(dev)
def qnode_Amp_QCQC(inputs, weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6,
                   weights_7, weights_8, weights_9, weights_10):
    qml.AmplitudeEmbedding(inputs, wires=range(n_qubits), pad_with=0.0, normalize=True)
    # --------------------------------------------------------- Convolutional Layer1 ---------------------------------------------------------#
    U_SU4(weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, wires=[0, 1])
    U_SU4(weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, wires=[2, 3])
    U_SU4(weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, wires=[4, 5])
    U_SU4(weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, wires=[6, 7])

    U_SU4(weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, wires=[1, 2])
    U_SU4(weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, wires=[3, 4])
    U_SU4(weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, wires=[5, 6])
    U_SU4(weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, wires=[7, 0])
    # # --------------------------------------------------------- Convolutional Layer2 ---------------------------------------------------------#
    U_SU5(weights_7, weights_8, weights_9, weights_10, wires=[0, 1])
    U_SU5(weights_7, weights_8, weights_9, weights_10, wires=[2, 3])
    U_SU5(weights_7, weights_8, weights_9, weights_10, wires=[4, 5])
    U_SU5(weights_7, weights_8, weights_9, weights_10, wires=[6, 7])

    U_SU5(weights_7, weights_8, weights_9, weights_10, wires=[1, 2])
    U_SU5(weights_7, weights_8, weights_9, weights_10, wires=[3, 4])
    U_SU5(weights_7, weights_8, weights_9, weights_10, wires=[5, 6])
    U_SU5(weights_7, weights_8, weights_9, weights_10, wires=[7, 0])

    result = [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]
    return result


@qml.qnode(dev)
def qnode_Ang_QCQC(inputs, weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6,
                   weights_7, weights_8, weights_9, weights_10):
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    # --------------------------------------------------------- Convolutional Layer1 ---------------------------------------------------------#
    U_SU4(weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, wires=[0, 1])
    U_SU4(weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, wires=[2, 3])
    U_SU4(weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, wires=[4, 5])
    U_SU4(weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, wires=[6, 7])

    U_SU4(weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, wires=[1, 2])
    U_SU4(weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, wires=[3, 4])
    U_SU4(weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, wires=[5, 6])
    U_SU4(weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, wires=[7, 0])
    # # --------------------------------------------------------- Convolutional Layer2 ---------------------------------------------------------#
    U_SU5(weights_7, weights_8, weights_9, weights_10, wires=[0, 1])
    U_SU5(weights_7, weights_8, weights_9, weights_10, wires=[2, 3])
    U_SU5(weights_7, weights_8, weights_9, weights_10, wires=[4, 5])
    U_SU5(weights_7, weights_8, weights_9, weights_10, wires=[6, 7])

    U_SU5(weights_7, weights_8, weights_9, weights_10, wires=[1, 2])
    U_SU5(weights_7, weights_8, weights_9, weights_10, wires=[3, 4])
    U_SU5(weights_7, weights_8, weights_9, weights_10, wires=[5, 6])
    U_SU5(weights_7, weights_8, weights_9, weights_10, wires=[7, 0])

    result = [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]
    return result

num_ap_list = [20,25,30]
num_ue_list = [10,15]
tau_p_list = [5,10]
batch_size = 16
results = {}

for num_ap, num_ue, tau_p in itertools.product(num_ap_list, num_ue_list, tau_p_list):
    print(f"Running model with num_ap = {num_ap}, num_ue = {num_ue}, tau_p = {tau_p}")
    test_data_numpy, train_dataloader, test_dataloader, train_dataloader_CNN, test_dataloader_CNN = one_time_dataset(num_ap, num_ue, tau_p, batch_size)
    beta_test = test_data_numpy.reshape(batch_size * 12, num_ap, num_ue)
    greedy_list = []
    random_list = []
    masterAP_list = []
    for i in range(batch_size * 3):
        pilot_init = np.random.randint(tau_p, size=num_ue)
        rate_greedy = greedy_assignment_1(beta_test[i], num_ap, num_ue, tau_p, pilot_init)
        rate_random = random_assignment_1(beta_test[i], pilot_init, num_ap, num_ue, tau_p)
        rate_masterAP = master_AP_assignment_1(beta_test[i], num_ap, num_ue, tau_p)
        greedy_list.append(rate_greedy)
        random_list.append(rate_random)
        masterAP_list.append(rate_masterAP)

    greedy_list = np.array(greedy_list)
    random_list = np.array(random_list)
    masterAP_list = np.array(masterAP_list)

    mean_greedy_list = np.mean(greedy_list)
    mean_random_list = np.mean(random_list)
    mean_masterAP_list = np.mean(masterAP_list)

    model_HQCNN_Amp_QCQP = HQCNN_QP(num_ap, num_ue, tau_p, qnode_Amp_QCQP, weight_shapes_QCQP)
    model_HQCNN_Ang_QCQP = HQCNN_QP(num_ap, num_ue, tau_p, qnode_Ang_QCQP, weight_shapes_QCQP)
    model_HQCNN_Amp_QC = HQCNN_noQP(num_ap, num_ue, tau_p, qnode_Amp_QC, weight_shapes_QC)
    model_HQCNN_Ang_QC = HQCNN_noQP(num_ap, num_ue, tau_p, qnode_Ang_QC, weight_shapes_QC)
    model_HQCNN_Amp_QCQC = HQCNN_noQP(num_ap, num_ue, tau_p, qnode_Amp_QCQC, weight_shapes_QCQC)
    model_HQCNN_Ang_QCQC = HQCNN_noQP(num_ap, num_ue, tau_p, qnode_Ang_QCQC, weight_shapes_QCQC)
    model_MLP = MLPModel(num_ap, num_ue, tau_p)
    model_CNN = CNNModel(num_ap, num_ue, tau_p)
    print(f'HQCNN-TRAINING PROCESS ...')
    print(f'\tHQCNN-AMPLITUDE-QCQP ...')
    trloss_Amp_QCQP, teloss_Amp_QCQP = train_model(model_HQCNN_Amp_QCQP, train_dataloader, test_dataloader, batch_size, num_ap, num_ue, tau_p, num_epoch=15, lr=0.01)
    print(f'\tHQCNN-ANGLE-QCQP ...')
    trloss_Ang_QCQP, teloss_Ang_QCQP = train_model(model_HQCNN_Ang_QCQP, train_dataloader, test_dataloader, batch_size, num_ap, num_ue, tau_p, num_epoch=15, lr=0.01)
    print(f'\tHQCNN-AMPLITUDE-QC ...')
    trloss_Amp_QC  , teloss_Amp_QC= train_model(model_HQCNN_Amp_QC, train_dataloader, test_dataloader, batch_size, num_ap, num_ue, tau_p, num_epoch=15, lr=0.01)
    print(f'\tHQCNN-ANGLE-QC ...')
    trloss_Ang_QC  , teloss_Ang_QC= train_model(model_HQCNN_Ang_QC, train_dataloader, test_dataloader, batch_size, num_ap, num_ue, tau_p, num_epoch=15, lr=0.01)
    print(f'\tHQCNN-AMPLITUDE-QCQC ...')
    trloss_Amp_QCQC, teloss_Amp_QCQC = train_model(model_HQCNN_Amp_QCQC, train_dataloader, test_dataloader, batch_size, num_ap, num_ue, tau_p, num_epoch=15, lr=0.01)
    print(f'\tHQCNN-ANGLE-QCQC ...')
    trloss_Ang_QCQC, teloss_Ang_QCQC = train_model(model_HQCNN_Ang_QCQC, train_dataloader, test_dataloader, batch_size, num_ap, num_ue, tau_p, num_epoch=15, lr=0.01)
    print(f'MLP-TRAINING PROCESS ...')
    train_model(model_MLP, train_dataloader, test_dataloader, batch_size, num_ap, num_ue, tau_p, num_epoch=10, lr=0.01)
    print(f'CNN-TRAINING PROCESS ...')
    train_model(model_CNN, train_dataloader_CNN, test_dataloader_CNN, batch_size, num_ap, num_ue, tau_p, num_epoch=10,
                lr=0.01)


    key = f'num_ap={num_ap}_num_ue={num_ue}_tau_p={tau_p}'
    results[key] = {
        'HQCNN_Amp_QCQP': {'train_loss': np.max(trloss_Amp_QCQP), 'test_loss': np.max(teloss_Amp_QCQP)},
        'HQCNN_Ang_QCQP': {'train_loss': np.max(trloss_Ang_QCQP), 'test_loss': np.max(teloss_Ang_QCQP)},
        'HQCNN_Amp_QC': {'train_loss': np.max(trloss_Amp_QC), 'test_loss': np.max(teloss_Amp_QC)},
        'HQCNN_Ang_QC': {'train_loss': np.max(trloss_Ang_QC), 'test_loss': np.max(teloss_Ang_QC)},
        'HQCNN_Amp_QCQC': {'train_loss': np.max(trloss_Amp_QCQC), 'test_loss': np.max(teloss_Amp_QCQC)},
        'HQCNN_Ang_QCQC': {'train_loss': np.max(trloss_Ang_QCQC), 'test_loss': np.max(teloss_Ang_QCQC)},
        'Random_Algo': {'value': mean_random_list},
        'Greedy_Algo': {'value': mean_greedy_list},
        'MasterAP_Algo': {'value': mean_masterAP_list},
    }

models = ['HQCNN_Amp_QCQP', 'HQCNN_Ang_QCQP', 'HQCNN_Amp_QC', 'HQCNN_Ang_QC', 'HQCNN_Amp_QCQC', 'HQCNN_Ang_QCQC']
algorithms = ['Random_Algo', 'Greedy_Algo', 'MasterAP_Algo']
plot_data = {model: {'train_loss': [], 'test_loss': []} for model in models}
plot_data.update({algo: {'value': []} for algo in algorithms})

for num_ap in num_ap_list:
    for model in models:
        key = f'num_ap={num_ap}_num_ue=10_tau_p=5'  # Assuming you want the case with num_ue=10 and tau_p=5
        plot_data[model]['train_loss'].append(results[key][model]['train_loss'])
        plot_data[model]['test_loss'].append(results[key][model]['test_loss'])
    for algo in algorithms:
        plot_data[algo]['value'].append(results[key][algo]['value'])

# Plot the results
plt.figure(figsize=(10, 8))

# Plot training and testing losses for models
for model in models:
    plt.plot(num_ap_list, plot_data[model]['train_loss'], label=f'{model} Train Loss', marker='o')
    plt.plot(num_ap_list, plot_data[model]['test_loss'], label=f'{model} Test Loss', marker='o', linestyle='--')

# Plot algorithm values
for algo in algorithms:
    plt.plot(num_ap_list, plot_data[algo]['value'], label=algo, marker='x')

plt.xlabel('M (Number of APs)')
plt.ylabel('Loss / Value')
plt.title('Model Training/Testing Losses and Algorithm Values')
plt.legend()
plt.grid(True)
plt.show()

with open('results.json', 'w') as json_file:
    json.dump(results, json_file, indent=4)