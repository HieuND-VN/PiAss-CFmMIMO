import numpy as np
import torch
import torch.nn.functional as F
import pennylane as qml
from model import *
from benchmark import *
from dataset import one_time_dataset
import itertools


def balance_array(arr, K, tau_p):
    arr_copy = arr.copy()
    counts = np.bincount(arr_copy, minlength=tau_p)

    # Calculate the dynamic target frequency and handle remainders
    base_target_freq = K // tau_p
    remainder = K % tau_p

    # Distribute the remainder as evenly as possible
    target_freqs = np.full(tau_p, base_target_freq)
    target_freqs[:remainder] += 1  # Distribute the remainder

    to_replace = []
    under_represented = []

    for i in range(tau_p):
        if counts[i] > target_freqs[i]:
            excess = counts[i] - target_freqs[i]
            to_replace.extend([i] * excess)
        elif counts[i] < target_freqs[i]:
            deficit = target_freqs[i] - counts[i]
            under_represented.extend([i] * deficit)

    np.random.shuffle(to_replace)
    np.random.shuffle(under_represented)

    # Ensure that we do not go out of bounds
    min_len = min(len(to_replace), len(under_represented))

    for i in range(min_len):
        index_to_replace = np.where(arr_copy == to_replace[i])[0][0]
        arr_copy[index_to_replace] = under_represented[i]

    return arr_copy

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
    # "weights_9": 1,  # shape for weights_9
    # "weights_10": 1, # shape for weights_10
    # "weights_11": 1, # shape for weights_11
    # "weights_12": 1, # shape for weights_12
    # "weights_13": 1, # shape for weights_13
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


def U_SU4_2(weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6,
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
def qnode_Amp_QCQP(inputs, weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6,
                   weights_7, weights_8):
    # qml.AngleEmbedding(inputs, wires=range(n_qubits))
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
def qnode_Ang_QCQP(inputs, weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6,
                   weights_7,
                   weights_8):
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
    # # --------------------------------------------------------- Convolutional Layer2 ---------------------------------------------------------#
    # U_SU5(weights_7, weights_8, weights_9, weights_10, wires=[0, 1])
    # U_SU5(weights_7, weights_8, weights_9, weights_10, wires=[2, 3])
    # U_SU5(weights_7, weights_8, weights_9, weights_10, wires=[4, 5])
    # U_SU5(weights_7, weights_8, weights_9, weights_10, wires=[6, 7])
    #
    # U_SU5(weights_7, weights_8, weights_9, weights_10, wires=[1, 2])
    # U_SU5(weights_7, weights_8, weights_9, weights_10, wires=[3, 4])
    # U_SU5(weights_7, weights_8, weights_9, weights_10, wires=[5, 6])
    # U_SU5(weights_7, weights_8, weights_9, weights_10, wires=[7, 0])

    # U_SU4_2(weights_7, weights_8, weights_9, weights_10, weights_11, weights_12, weights_13, wires=[0, 1])
    # U_SU4_2(weights_7, weights_8, weights_9, weights_10, weights_11, weights_12, weights_13, wires=[2, 3])
    # U_SU4_2(weights_7, weights_8, weights_9, weights_10, weights_11, weights_12, weights_13, wires=[4, 5])
    # U_SU4_2(weights_7, weights_8, weights_9, weights_10, weights_11, weights_12, weights_13, wires=[6, 7])
    # U_SU4_2(weights_7, weights_8, weights_9, weights_10, weights_11, weights_12, weights_13, wires=[1, 2])
    # U_SU4_2(weights_7, weights_8, weights_9, weights_10, weights_11, weights_12, weights_13, wires=[3, 4])
    # U_SU4_2(weights_7, weights_8, weights_9, weights_10, weights_11, weights_12, weights_13, wires=[5, 6])
    # U_SU4_2(weights_7, weights_8, weights_9, weights_10, weights_11, weights_12, weights_13, wires=[7, 0])

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
    # qml.AngleEmbedding(inputs, wires=range(n_qubits))
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
    # U_SU5(weights_7, weights_8, weights_9, weights_10, wires=[0, 1])
    # U_SU5(weights_7, weights_8, weights_9, weights_10, wires=[2, 3])
    # U_SU5(weights_7, weights_8, weights_9, weights_10, wires=[4, 5])
    # U_SU5(weights_7, weights_8, weights_9, weights_10, wires=[6, 7])
    #
    # U_SU5(weights_7, weights_8, weights_9, weights_10, wires=[1, 2])
    # U_SU5(weights_7, weights_8, weights_9, weights_10, wires=[3, 4])
    # U_SU5(weights_7, weights_8, weights_9, weights_10, wires=[5, 6])
    # U_SU5(weights_7, weights_8, weights_9, weights_10, wires=[7, 0])

    # U_SU4_2(weights_7, weights_8, weights_9, weights_10, weights_11, weights_12, weights_13, wires=[0, 1])
    # U_SU4_2(weights_7, weights_8, weights_9, weights_10, weights_11, weights_12, weights_13, wires=[2, 3])
    # U_SU4_2(weights_7, weights_8, weights_9, weights_10, weights_11, weights_12, weights_13, wires=[4, 5])
    # U_SU4_2(weights_7, weights_8, weights_9, weights_10, weights_11, weights_12, weights_13, wires=[6, 7])
    # U_SU4_2(weights_7, weights_8, weights_9, weights_10, weights_11, weights_12, weights_13, wires=[1, 2])
    # U_SU4_2(weights_7, weights_8, weights_9, weights_10, weights_11, weights_12, weights_13, wires=[3, 4])
    # U_SU4_2(weights_7, weights_8, weights_9, weights_10, weights_11, weights_12, weights_13, wires=[5, 6])
    # U_SU4_2(weights_7, weights_8, weights_9, weights_10, weights_11, weights_12, weights_13, wires=[7, 0])

    # --------------------------------------------------------- Pooling Layer1 ---------------------------------------------------------#
    # Pooling Circuit  Block 2 weights_7, weights_8
    # Pooling_ansatz1(weights_7, weights_8, wires=[0, 1])
    # Pooling_ansatz1(weights_7, weights_8, wires=[2, 3])
    # Pooling_ansatz1(weights_7, weights_8, wires=[4, 5])
    # Pooling_ansatz1(weights_7, weights_8, wires=[6, 7])

    result = [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]
    return result


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


@qml.qnode(dev)
def qnode_Amp_QCQC(inputs, weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6,
                   weights_7, weights_8, weights_9, weights_10):
    # qml.AngleEmbedding(inputs, wires=range(n_qubits))
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

    # U_SU4_2(weights_7, weights_8, weights_9, weights_10, weights_11, weights_12, weights_13, wires=[0, 1])
    # U_SU4_2(weights_7, weights_8, weights_9, weights_10, weights_11, weights_12, weights_13, wires=[2, 3])
    # U_SU4_2(weights_7, weights_8, weights_9, weights_10, weights_11, weights_12, weights_13, wires=[4, 5])
    # U_SU4_2(weights_7, weights_8, weights_9, weights_10, weights_11, weights_12, weights_13, wires=[6, 7])
    # U_SU4_2(weights_7, weights_8, weights_9, weights_10, weights_11, weights_12, weights_13, wires=[1, 2])
    # U_SU4_2(weights_7, weights_8, weights_9, weights_10, weights_11, weights_12, weights_13, wires=[3, 4])
    # U_SU4_2(weights_7, weights_8, weights_9, weights_10, weights_11, weights_12, weights_13, wires=[5, 6])
    # U_SU4_2(weights_7, weights_8, weights_9, weights_10, weights_11, weights_12, weights_13, wires=[7, 0])
    result = [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]
    return result


@qml.qnode(dev)
def qnode_Ang_QCQC(inputs, weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6,
                   weights_7, weights_8, weights_9, weights_10):
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
    # # --------------------------------------------------------- Convolutional Layer2 ---------------------------------------------------------#
    U_SU5(weights_7, weights_8, weights_9, weights_10, wires=[0, 1])
    U_SU5(weights_7, weights_8, weights_9, weights_10, wires=[2, 3])
    U_SU5(weights_7, weights_8, weights_9, weights_10, wires=[4, 5])
    U_SU5(weights_7, weights_8, weights_9, weights_10, wires=[6, 7])

    U_SU5(weights_7, weights_8, weights_9, weights_10, wires=[1, 2])
    U_SU5(weights_7, weights_8, weights_9, weights_10, wires=[3, 4])
    U_SU5(weights_7, weights_8, weights_9, weights_10, wires=[5, 6])
    U_SU5(weights_7, weights_8, weights_9, weights_10, wires=[7, 0])

    # U_SU4_2(weights_7, weights_8, weights_9, weights_10, weights_11, weights_12, weights_13, wires=[0, 1])
    # U_SU4_2(weights_7, weights_8, weights_9, weights_10, weights_11, weights_12, weights_13, wires=[2, 3])
    # U_SU4_2(weights_7, weights_8, weights_9, weights_10, weights_11, weights_12, weights_13, wires=[4, 5])
    # U_SU4_2(weights_7, weights_8, weights_9, weights_10, weights_11, weights_12, weights_13, wires=[6, 7])
    # U_SU4_2(weights_7, weights_8, weights_9, weights_10, weights_11, weights_12, weights_13, wires=[1, 2])
    # U_SU4_2(weights_7, weights_8, weights_9, weights_10, weights_11, weights_12, weights_13, wires=[3, 4])
    # U_SU4_2(weights_7, weights_8, weights_9, weights_10, weights_11, weights_12, weights_13, wires=[5, 6])
    # U_SU4_2(weights_7, weights_8, weights_9, weights_10, weights_11, weights_12, weights_13, wires=[7, 0])
    result = [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]
    return result

num_ap_list = [20,25,30]
num_ue_list = [10,15]
tau_p_list = [5,10]
batch_size = 16
# n_qubits = 8

for num_ap, num_ue, tau_p in itertools.product(num_ap_list, num_ue_list, tau_p_list):
    print(f"Running model with num_ap = {num_ap}, num_ue = {num_ue}, tau_p = {tau_p}")
    test_data_numpy, train_dataloader, test_dataloader, train_dataloader_CNN, test_dataloader_CNN = one_time_dataset(num_ap,
                                                                                                                 num_ue,
                                                                                                                 tau_p,
                                                                                                                 batch_size)
    beta_test = test_data_numpy.reshape(batch_size * 12, num_ap, num_ue)  # [num_test, num_ap, num_ue]
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
    #
    beta_test_flatten = test_data_numpy[0]  # [num_ap*num_ue]
    beta_test_reshape = beta_test_flatten.reshape(num_ap, num_ue)  # [num_ap, num_ue]
    beta_flatten_test = np.expand_dims(beta_test_flatten, axis=0)  # [1, num_ap*num_ue]
    beta_reshape_test = np.expand_dims(beta_test_reshape, axis=0)  # [1, num_ap, num_ue]

    xs = torch.tensor(beta_flatten_test, dtype=torch.float32)
    xs_CNN = torch.tensor(beta_reshape_test, dtype=torch.float32)

    output_HQCNN_Amp_QCQP = model_HQCNN_Amp_QCQP(xs)
    output_HQCNN_Ang_QCQP = model_HQCNN_Ang_QCQP(xs)
    output_HQCNN_Amp_QC = model_HQCNN_Amp_QC(xs)
    output_HQCNN_Ang_QC = model_HQCNN_Ang_QC(xs)
    output_HQCNN_Amp_QCQC = model_HQCNN_Amp_QCQC(xs)
    output_HQCNN_Ang_QCQC = model_HQCNN_Ang_QCQC(xs)
    output_MLP = model_MLP(xs)
    output_CNN = model_CNN(xs_CNN)

    pilot_probs_HQCNN_Amp_QCQP = F.softmax(output_HQCNN_Amp_QCQP.reshape(num_ue, tau_p), dim=-1)
    pilot_probs_HQCNN_Ang_QCQP = F.softmax(output_HQCNN_Ang_QCQP.reshape(num_ue, tau_p), dim=-1)
    pilot_probs_HQCNN_Amp_QC = F.softmax(output_HQCNN_Amp_QC  .reshape(num_ue, tau_p), dim=-1)
    pilot_probs_HQCNN_Ang_QC = F.softmax(output_HQCNN_Ang_QC  .reshape(num_ue, tau_p), dim=-1)
    pilot_probs_HQCNN_Amp_QCQC = F.softmax(output_HQCNN_Amp_QCQC.reshape(num_ue, tau_p), dim=-1)
    pilot_probs_HQCNN_Ang_QCQC = F.softmax(output_HQCNN_Ang_QCQC.reshape(num_ue, tau_p), dim=-1)

    pilot_index_HQCNN_Amp_QCQP = torch.argmax(pilot_probs_HQCNN_Amp_QCQP, dim=-1)
    pilot_index_HQCNN_Ang_QCQP = torch.argmax(pilot_probs_HQCNN_Ang_QCQP, dim=-1)
    pilot_index_HQCNN_Amp_QC = torch.argmax(pilot_probs_HQCNN_Amp_QC  , dim=-1)
    pilot_index_HQCNN_Ang_QC = torch.argmax(pilot_probs_HQCNN_Ang_QC  , dim=-1)
    pilot_index_HQCNN_Amp_QCQC = torch.argmax(pilot_probs_HQCNN_Amp_QCQC, dim=-1)
    pilot_index_HQCNN_Ang_QCQC = torch.argmax(pilot_probs_HQCNN_Ang_QCQC, dim=-1)

    pilot_index_HQCNN_Amp_QCQP = pilot_index_HQCNN_Amp_QCQP.cpu().numpy()
    pilot_index_HQCNN_Ang_QCQP = pilot_index_HQCNN_Ang_QCQP.cpu().numpy()
    pilot_index_HQCNN_Amp_QC = pilot_index_HQCNN_Amp_QC.cpu().numpy()
    pilot_index_HQCNN_Ang_QC = pilot_index_HQCNN_Ang_QC.cpu().numpy()
    pilot_index_HQCNN_Amp_QCQC = pilot_index_HQCNN_Amp_QCQC.cpu().numpy()
    pilot_index_HQCNN_Ang_QCQC = pilot_index_HQCNN_Ang_QCQC.cpu().numpy()

    balance_HQCNN_pilot_index_Amp_QCQP = balance_array(pilot_index_HQCNN_Amp_QCQP, num_ue, tau_p)
    balance_HQCNN_pilot_index_Ang_QCQP = balance_array(pilot_index_HQCNN_Ang_QCQP, num_ue, tau_p)
    balance_HQCNN_pilot_index_Amp_QC = balance_array(pilot_index_HQCNN_Amp_QC, num_ue, tau_p)
    balance_HQCNN_pilot_index_Ang_QC = balance_array(pilot_index_HQCNN_Ang_QC, num_ue, tau_p)
    balance_HQCNN_pilot_index_Amp_QCQC = balance_array(pilot_index_HQCNN_Amp_QCQC, num_ue, tau_p)
    balance_HQCNN_pilot_index_Ang_QCQC = balance_array(pilot_index_HQCNN_Ang_QCQC, num_ue, tau_p)

    pilot_probs_MLP = F.softmax(output_MLP.reshape(num_ue, tau_p), dim=-1)
    pilot_index_MLP = torch.argmax(pilot_probs_MLP, dim=-1)
    pilot_index_MLP = pilot_index_MLP.cpu().numpy()
    balance_MLP_pilot_index = balance_array(pilot_index_MLP, num_ue, tau_p)

    pilot_probs_CNN = F.softmax(output_CNN.reshape(num_ue, tau_p), dim=-1)
    pilot_index_CNN = torch.argmax(pilot_probs_CNN, dim=-1)
    pilot_index_CNN = pilot_index_CNN.cpu().numpy()
    balance_CNN_pilot_index = balance_array(pilot_index_CNN, num_ue, tau_p)

    rate_HQCNN_Amp_QCQP= calculate_dl_rate(beta_test_reshape, pilot_index_HQCNN_Amp_QCQP, num_ap, num_ue, tau_p)
    rate_HQCNN_Ang_QCQP= calculate_dl_rate(beta_test_reshape, pilot_index_HQCNN_Ang_QCQP, num_ap, num_ue, tau_p)
    rate_HQCNN_Amp_QC  = calculate_dl_rate(beta_test_reshape, pilot_index_HQCNN_Amp_QC, num_ap, num_ue, tau_p)
    rate_HQCNN_Ang_QC  = calculate_dl_rate(beta_test_reshape, pilot_index_HQCNN_Ang_QC, num_ap, num_ue, tau_p)
    rate_HQCNN_Amp_QCQC= calculate_dl_rate(beta_test_reshape, pilot_index_HQCNN_Amp_QCQC, num_ap, num_ue, tau_p)
    rate_HQCNN_Ang_QCQC= calculate_dl_rate(beta_test_reshape, pilot_index_HQCNN_Ang_QCQC, num_ap, num_ue, tau_p)

    rate_balance_HQCNN_Amp_QCQP = calculate_dl_rate(beta_test_reshape, balance_HQCNN_pilot_index_Amp_QCQP, num_ap, num_ue, tau_p)
    rate_balance_HQCNN_Ang_QCQP = calculate_dl_rate(beta_test_reshape, balance_HQCNN_pilot_index_Ang_QCQP, num_ap, num_ue, tau_p)
    rate_balance_HQCNN_Amp_QC   = calculate_dl_rate(beta_test_reshape, balance_HQCNN_pilot_index_Amp_QC, num_ap, num_ue, tau_p)
    rate_balance_HQCNN_Ang_QC   = calculate_dl_rate(beta_test_reshape, balance_HQCNN_pilot_index_Ang_QC, num_ap, num_ue, tau_p)
    rate_balance_HQCNN_Amp_QCQC = calculate_dl_rate(beta_test_reshape, balance_HQCNN_pilot_index_Amp_QCQC, num_ap, num_ue, tau_p)
    rate_balance_HQCNN_Ang_QCQC = calculate_dl_rate(beta_test_reshape, balance_HQCNN_pilot_index_Ang_QCQC, num_ap, num_ue, tau_p)

    rate_MLP = calculate_dl_rate(beta_test_reshape, pilot_index_MLP, num_ap, num_ue, tau_p)
    rate_balance_MLP = calculate_dl_rate(beta_test_reshape, balance_MLP_pilot_index, num_ap, num_ue, tau_p)
    rate_CNN = calculate_dl_rate(beta_test_reshape, pilot_index_CNN, num_ap, num_ue, tau_p)
    rate_balance_CNN = calculate_dl_rate(beta_test_reshape, balance_CNN_pilot_index, num_ap, num_ue, tau_p)

    pilot_init = np.random.randint(tau_p, size=num_ue)
    rate_greedy = greedy_assignment_1(beta_test_reshape, num_ap, num_ue, tau_p, pilot_init)
    rate_random = random_assignment_1(beta_test_reshape, pilot_init, num_ap, num_ue, tau_p)
    rate_masterAP = master_AP_assignment_1(beta_test_reshape, num_ap, num_ue, tau_p)

    print(f'ONE SAMPLE')
    print(f'\tHQCNN_Amp_QCQP: {np.sum(rate_HQCNN_Amp_QCQP):.4f}\tPilot_index: {pilot_index_HQCNN_Amp_QCQP}\n-------------------------------------------------')
    print(f'\tHQCNN_Ang_QCQP: {np.sum(rate_HQCNN_Ang_QCQP):.4f}\tPilot_index: {pilot_index_HQCNN_Ang_QCQP}\n-------------------------------------------------')
    print(f'\tHQCNN_Amp_QC: {np.sum(rate_HQCNN_Amp_QC):.4f}\tPilot_index: {pilot_index_HQCNN_Amp_QC}\n-------------------------------------------------')
    print(f'\tHQCNN_Ang_QC: {np.sum(rate_HQCNN_Ang_QC):.4f}\tPilot_index: {pilot_index_HQCNN_Ang_QC}\n-------------------------------------------------')
    print(f'\tHQCNN_Amp_QCQC: {np.sum(rate_HQCNN_Amp_QCQC):.4f}\tPilot_index: {pilot_index_HQCNN_Amp_QCQC}\n-------------------------------------------------')
    print(f'\tHQCNN_Ang_QCQC: {np.sum(rate_HQCNN_Ang_QCQC):.4f}\tPilot_index: {pilot_index_HQCNN_Ang_QCQC}\n-------------------------------------------------')

    print(f'\tHQCNN_balance: {np.sum(rate_balance_HQCNN_Amp_QCQP):.4f}\tPilot_index: {balance_HQCNN_pilot_index_Amp_QCQP}\n-------------------------------------------------')
    print(f'\tHQCNN_balance: {np.sum(rate_balance_HQCNN_Ang_QCQP):.4f}\tPilot_index: {balance_HQCNN_pilot_index_Ang_QCQP}\n-------------------------------------------------')
    print(f'\tHQCNN_balance: {np.sum(rate_balance_HQCNN_Amp_QC):.4f}\tPilot_index: {balance_HQCNN_pilot_index_Amp_QC}\n-------------------------------------------------')
    print(f'\tHQCNN_balance: {np.sum(rate_balance_HQCNN_Ang_QC):.4f}\tPilot_index: {balance_HQCNN_pilot_index_Ang_QC}\n-------------------------------------------------')
    print(f'\tHQCNN_balance: {np.sum(rate_balance_HQCNN_Amp_QCQC):.4f}\tPilot_index: {balance_HQCNN_pilot_index_Amp_QCQC}\n-------------------------------------------------')
    print(f'\tHQCNN_balance: {np.sum(rate_balance_HQCNN_Ang_QCQC):.4f}\tPilot_index: {balance_HQCNN_pilot_index_Ang_QCQC}\n-------------------------------------------------')

    print(f'\tMLP: {np.sum(rate_MLP)}\tPilot_index: {pilot_index_MLP}\n-------------------------------------------------')
    # print(f'MLP_balance: {np.sum(rate_balance_MLP)}\nPilot_index: {balance_MLP_pilot_index}\n'
    #       f'-------------------------------------------------')
    print(f'\tCNN: {np.sum(rate_CNN)}\nPilot_index: {pilot_index_CNN}\n-------------------------------------------------')
    # print(f'CNN_balance: {np.sum(rate_balance_CNN)}\nPilot_index: {balance_CNN_pilot_index}\n'
    #       f'-------------------------------------------------')
    print(f'\tRandom Algo: {rate_random}\n-------------------------------------------------')
    print(f'\tGreedy Algo: {rate_greedy}\n-------------------------------------------------')
    print(f'\tMaster-AP Algo: {rate_masterAP}\n-------------------------------------------------')

    # print(f'HQCNN: {np.sum(rate_HQCNN)}\n-------------------------------------------------')
    # print(f'MLP: {np.sum(rate_MLP)}\n-------------------------------------------------')
    # print(f'CNN: {np.sum(rate_CNN)}\n-------------------------------------------------')
    print(f'Whole Dataset')
    print(f'\tHQCNN_Amp_QCQP:{np.max(trloss_Amp_QCQP):.4f}\{np.max(teloss_Amp_QCQP):.4f}')
    print(f'\tHQCNN_Ang_QCQP:{np.max(trloss_Ang_QCQP):.4f}\{np.max(teloss_Ang_QCQP):.4f}')
    print(f'\tHQCNN_Amp_QC: {np.max(trloss_Amp_QC):.4f}\{np.max(teloss_Amp_QC):.4f}')
    print(f'\tHQCNN_Ang_QC: {np.max(trloss_Ang_QC):.4f}\{np.max(teloss_Ang_QC):.4f}')
    print(f'\tHQCNN_Amp_QCQC:{np.max(trloss_Amp_QCQC):.4f}\{np.max(teloss_Amp_QCQC):.4f}')
    print(f'\tHQCNN_Ang_QCQC:{np.max(trloss_Ang_QCQC):.4f}\{np.max(teloss_Ang_QCQC):.4f}')
    print(f'\tRandom Algo: {mean_random_list}\n-------------------------------------------------')
    print(f'\tGreedy Algo: {mean_greedy_list}\n-------------------------------------------------')
    print(f'\tMaster-AP Algo:{mean_masterAP_list}\n-------------------------------------------------')
