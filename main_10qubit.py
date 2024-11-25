import numpy as np
import torch
import torch.nn.functional as F
import pennylane as qml
from model import *
from benchmark import *
from dataset import one_time_dataset
from bruteforce_calculator import generate_feasible_assignments
import time


num_ap = 30
num_ue = 20
tau_p = 10
batch_size = 16
n_qubits = 10
weight_shapes_QCQP = {"weights_0": 3,
                      "weights_1": 3,
                      "weights_2": 1,
                      "weights_3": 1,
                      "weights_4": 1,
                      "weights_5": 3,
                      "weights_6": 3,
                      "weights_7": 1,
                      "weights_8": 1,}

weight_shapes_QC = {"weights_0": 3,
                    "weights_1": 3,
                    "weights_2": 1,
                    "weights_3": 1,
                    "weights_4": 1,
                    "weights_5": 3,
                    "weights_6": 3,}


Pooling_out = [1, 3, 5, 7, 9]
dev = qml.device("default.qubit", wires=n_qubits)


def U_SU4(weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, wires):
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

def Pooling_ansatz1(weights_0, weights_1, wires):  # 2 params
    qml.CRZ(weights_0, wires=[wires[0], wires[1]])
    qml.PauliX(wires=wires[0])
    qml.CRX(weights_1, wires=[wires[0], wires[1]])
@qml.qnode(dev)
def qnode_Ang_QCQP(inputs, weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, weights_7, weights_8):
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    # --------------------------------------------------------- Convolutional Layer1 ---------------------------------------------------------#
    U_SU4(weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, wires=[0, 1])
    U_SU4(weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, wires=[2, 3])
    U_SU4(weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, wires=[4, 5])
    U_SU4(weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, wires=[6, 7])
    U_SU4(weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, wires=[8, 9])

    U_SU4(weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, wires=[1, 2])
    U_SU4(weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, wires=[3, 4])
    U_SU4(weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, wires=[5, 6])
    U_SU4(weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, wires=[7, 8])
    U_SU4(weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, wires=[9, 0])

    # --------------------------------------------------------- Pooling Layer1 ---------------------------------------------------------#
    Pooling_ansatz1(weights_7, weights_8, wires=[0, 1])
    Pooling_ansatz1(weights_7, weights_8, wires=[2, 3])
    Pooling_ansatz1(weights_7, weights_8, wires=[4, 5])
    Pooling_ansatz1(weights_7, weights_8, wires=[6, 7])
    Pooling_ansatz1(weights_7, weights_8, wires=[8, 9])

    result = [qml.expval(qml.PauliZ(wires=i)) for i in Pooling_out]
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


test_data_numpy, train_dataloader, test_dataloader, train_dataloader_CNN, \
    test_dataloader_CNN, test_cdf, test_cdf_cnn = one_time_dataset(num_ap, num_ue, tau_p, batch_size)


pilot_list=generate_feasible_assignments(num_ue, tau_p)

beta_test = test_data_numpy.reshape(batch_size * 3, num_ap, num_ue)  # [num_test, num_ap, num_ue]
greedy_list = []
random_list = []
masterAP_list = []

for i in range(len(beta_test)):
    print(f'Working on sample [{i}]/[{len(beta_test)-1}]')
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







model_HQCNN_Ang_QCQP = HQCNN_Ang_QP(num_ap, num_ue, tau_p, qnode_Ang_QCQP, weight_shapes_QCQP, n_qubits)
model_HQCNN_Ang_QC = HQCNN_Ang_noQP(num_ap, num_ue, tau_p, qnode_Ang_QC, weight_shapes_QC, n_qubits)

model_MLP = MLPModel(num_ap, num_ue, tau_p, n_qubits)
model_CNN = CNNModel(num_ap, num_ue, tau_p)
print(f'HQCNN-TRAINING PROCESS ...')

print(f'\tHQCNN-ANGLE-QC ...')
HQCNN_QC_start = time.perf_counter()
trloss_Ang_QC, teloss_Ang_QC= train_model(model_HQCNN_Ang_QC, train_dataloader, test_dataloader, batch_size, num_ap, num_ue, tau_p, num_epoch=10, lr=0.001, is_MLP_CNN = False)
HQCNN_QC_end = time.perf_counter()

print(f'\tHQCNN-ANGLE-QCQP ...')
HQCNN_QCQP_start = time.perf_counter()
trloss_Ang_QCQP, teloss_Ang_QCQP = train_model(model_HQCNN_Ang_QCQP, train_dataloader, test_dataloader, batch_size, num_ap, num_ue, tau_p, num_epoch=10, lr=0.001, is_MLP_CNN = False)
HQCNN_QCQP_end = time.perf_counter()

print(f'MLP-TRAINING PROCESS ...')
MLP_start = time.perf_counter()
trloss_MLP, teloss_MLP = train_model(model_MLP, train_dataloader, test_dataloader, batch_size, num_ap, num_ue, tau_p, num_epoch=10, lr=0.01, is_MLP_CNN = True)
MLP_end = time.perf_counter()

print(f'CNN-TRAINING PROCESS ...')
CNN_start = time.perf_counter()
trloss_CNN, teloss_CNN = train_model(model_CNN, train_dataloader_CNN, test_dataloader_CNN, batch_size, num_ap, num_ue, tau_p, num_epoch=10, lr=0.001, is_MLP_CNN = True)
CNN_end = time.perf_counter()


print(f'Whole Dataset')

print(f'\tHQCNN_Ang_QC: {np.max(trloss_Ang_QC):.4f}\ {np.max(teloss_Ang_QC):.4f}')
print(f'\tHQCNN_Ang_QCQP: {np.max(trloss_Ang_QCQP):.4f}\ {np.max(teloss_Ang_QCQP):.4f}')
print(f'\tMLP: {np.max(trloss_MLP):.4f}\ {np.max(teloss_MLP):.4f}')
print(f'\tCNN: {np.max(trloss_CNN):.4f}\ {np.max(teloss_CNN):.4f}')
print(f'\tRandom Algo: {mean_random_list:.4f}')
print(f'\tGreedy Algo: {mean_greedy_list:.4f}')
print(f'\tMaster-AP Algo: {mean_masterAP_list:.4f}')

print(f'\nTime consuming')
print(f'HQCNN_QC: {(HQCNN_QC_end-HQCNN_QC_start): .4f}')
print(f'HQCNN_QCQP: {(HQCNN_QCQP_end-HQCNN_QCQP_start): .4f}')
print(f'MLP: {(MLP_end-MLP_start): .4f}')
print(f'CNN: {(CNN_end-CNN_start): .4f}')






print(f'Size of greedy list: {np.shape(greedy_list)}')
print(f'Size of random list: {np.shape(random_list)}')
print(f'Size of masterAP list: {np.shape(masterAP_list)}')

valid_hqcnn = valid_model(model_HQCNN_Ang_QC, test_cdf, num_ap, num_ue, tau_p)
valid_mlp = valid_model(model_MLP, test_cdf, num_ap, num_ue, tau_p)
valid_cnn = valid_model(model_CNN, test_cdf, num_ap, num_ue, tau_p)

print(f'valid_hqcnn:  {valid_hqcnn.shape}')
print(f'valid_mlp:  {valid_mlp.shape}')
print(f'valid_cnn:  {valid_cnn.shape}')
no_param_MLP = sum(p.numel() for p in model_MLP.parameters())
no_param_CNN = sum(p.numel() for p in model_CNN.parameters())
no_param_HQCNN_QC = sum(p.numel() for p in model_HQCNN_Ang_QC.parameters())
no_param_HQCNN_QCQP = sum(p.numel() for p in model_HQCNN_Ang_QCQP.parameters())

print(f'No. MLP: {no_param_MLP}')
print(f'No. CNN: {no_param_CNN}')
print(f'No. HQCNN-QC: {no_param_HQCNN_QC}')
print(f'No. HQCNN-QCQP: {no_param_HQCNN_QCQP}')



np.savez('sr-list_10qubit.npz',
         valid_rpas =random_list,
         valid_gpas =greedy_list,
         valid_mapas=masterAP_list,
         valid_hqcnn=valid_hqcnn,
         valid_mlp  =valid_mlp,
         valid_cnn  =valid_cnn,
         param_MLP = no_param_MLP,
         param_CNN = no_param_CNN,
         param_HQCNN_QC = no_param_HQCNN_QC,
         param_HQCNN_QCQP =no_param_HQCNN_QCQP)
         # exhaustive_search_list_value = exhaustive_search_list_value,
         # exhaustive_search_list_scheme = exhaustive_search_list_scheme)

