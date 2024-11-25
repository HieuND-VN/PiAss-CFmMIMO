import os
import numpy as np
import torch
from benchmark import calculate_dl_rate
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import pennylane as qml

rho_d = 1
rho_p = 1 / 2


class HQCNN_Amp_QP(nn.Module):
    def __init__(self, num_ap, num_ue, tau_p, qnode, weight_shapes):
        super(HQCNN_Amp_QP, self).__init__()
        self.clayer_1 = nn.Linear(num_ap * num_ue, 2 ** 8)
        self.qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)
        self.clayer_2 = nn.Linear(4, num_ue * tau_p)
        # self.clayer_2 = nn.Linear(4, num_ue * num_ue)

    def forward(self, x):
        x = self.clayer_1(x)
        x = self.qlayer(x)
        x = self.clayer_2(x)
        return x


class HQCNN_Ang_QP(nn.Module):
    def __init__(self, num_ap, num_ue, tau_p, qnode, weight_shapes, n_qubits):
        super(HQCNN_Ang_QP, self).__init__()
        self.clayer_1 = nn.Linear(num_ap * num_ue, n_qubits)
        self.qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)
        self.clayer_2 = nn.Linear(n_qubits/2, num_ue * tau_p)
        # self.clayer_2 = nn.Linear(4, num_ue * num_ue)

    def forward(self, x):
        x = self.clayer_1(x)
        x = self.qlayer(x)
        x = self.clayer_2(x)
        return x


class HQCNN_Amp_noQP(nn.Module):
    def __init__(self, num_ap, num_ue, tau_p, qnode, weight_shapes):
        super(HQCNN_Amp_noQP, self).__init__()
        self.clayer_1 = nn.Linear(num_ap * num_ue, 2 ** 8)
        self.qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)
        self.clayer_2 = nn.Linear(8, num_ue * tau_p)
        # self.clayer_2 = nn.Linear(4, num_ue * num_ue)

    def forward(self, x):
        x = self.clayer_1(x)
        x = self.qlayer(x)
        x = self.clayer_2(x)
        return x


class HQCNN_Ang_noQP(nn.Module):
    def __init__(self, num_ap, num_ue, tau_p, qnode, weight_shapes, n_qubits):
        super(HQCNN_Ang_noQP, self).__init__()
        self.clayer_1 = nn.Linear(num_ap * num_ue, n_qubits)
        self.qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)
        self.clayer_2 = nn.Linear(n_qubits, num_ue * tau_p)
        # self.clayer_2 = nn.Linear(4, num_ue * num_ue)

    def forward(self, x):
        x = self.clayer_1(x)
        x = self.qlayer(x)
        x = self.clayer_2(x)
        return x


class MLPModel(nn.Module):
    def __init__(self, num_ap, num_ue, tau_p,n_qubits):
        super(MLPModel, self).__init__()
        self.fc_1 = nn.Linear(num_ap * num_ue, n_qubits)
        self.fc_2 = nn.Linear(n_qubits, n_qubits)
        self.fc_3 = nn.Linear(n_qubits, num_ue * tau_p)
        # self.fc_3 = nn.Linear(4, num_ue * num_ue)

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
        # self.fc_hidden1 = nn.Linear(128, 64)  # First hidden layer after fc1
        # self.fc_hidden2 = nn.Linear(64, 32)  # Second hidden layer
        self.fc2 = nn.Linear(128, num_ue * tau_p)  # Final output layer
        # self.fc2 = nn.Linear(32, num_ue * num_ue)  # Final output layer
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
        x = self.fc1(x)
        # x = F.relu(self.fc_hidden1(x))
        # x = F.relu(self.fc_hidden2(x))
        x = self.fc2(x)
        return x


def dl_rate_calculator_w_pilot_probs(pilot_probs, beta, tau_p):
    num_ap, num_ue = beta.shape
    phi_inner_product = torch.matmul(pilot_probs, pilot_probs.T)
    phi_inner_product_squared = phi_inner_product
    inner_sum = torch.matmul(beta, phi_inner_product_squared)
    numerator_c = torch.sqrt(torch.tensor(rho_p * tau_p, dtype=torch.float32)) * beta  # Shape: (M, K)
    denominator_c = tau_p * rho_p * inner_sum + 1
    c_mk = numerator_c / denominator_c
    gamma = torch.sqrt(torch.tensor(tau_p * rho_p, dtype=torch.float32)) * beta * c_mk
    etaa = 1 / torch.sum(gamma, dim=1)
    eta = etaa[:, None].expand(-1, num_ue)
    DS = rho_d * (torch.sum(torch.sqrt(eta) * gamma, dim=0)) ** 2  # Shape (num_ue,)
    eta_gamma = eta * gamma
    product = torch.sum(eta_gamma[:, :, None] * beta[:, None, :], dim=0)
    BU = torch.sum(product, dim=0)
    UI = torch.zeros(num_ue, device=beta.device)
    flag = torch.matmul(pilot_probs, pilot_probs.T)
    flag = flag.float().to(dtype=torch.float32)
    for k in range(num_ue):
        sum_term_1 = 0
        for j in range(num_ue):
            sum_term_2 = 0
            if j != k:
                for m in range(num_ap):
                    sum_term_2 += (torch.sqrt(eta[m, j]) * gamma[m, j] * beta[m, k]) / (beta[m, j])

            sum_term_1 += (sum_term_2 ** 2) * (flag[k, j])
        UI[k] = rho_d * sum_term_1

    sinr = DS / (UI + BU + 1)
    return torch.log2(1 + sinr)


def loss_function_PA(beta, output, batch_sz, num_ap, num_ue, tau_p):
    # batch_sz = 1
    sum_rate_each_batch = torch.zeros(batch_sz)
    beta = beta.reshape(batch_sz, num_ap, num_ue)
    pilot_probs_1 = output.reshape(batch_sz, num_ue, tau_p)
    # pilot_probs_1 = output.reshape(batch_sz, num_ue, num_ue)
    pilot_probs = F.softmax(pilot_probs_1, dim=-1)

    # pilot_index = torch.argmax(pilot_probs, dim=-1) # loss grad_fn from here
    # torch_one_hot = F.one_hot(pilot_index, num_classes = tau_p)

    # pilot_probs_process = pilot_probs * torch_one_hot
    for i in range(batch_sz):
        sum_rate_each_batch[i] = torch.sum(dl_rate_calculator_w_pilot_probs(pilot_probs[i], beta[i], tau_p))
    return torch.mean(sum_rate_each_batch) * (-1)


def loss_function_PA_test(beta, output, batch_sz, num_ap, num_ue, tau_p):
    # batch_sz = 1
    sum_rate_each_batch = torch.zeros(batch_sz)
    beta = beta.reshape(batch_sz, num_ap, num_ue)
    pilot_probs_1 = output.reshape(batch_sz, num_ue, tau_p)
    # pilot_probs_1 = output.reshape(batch_sz, num_ue, num_ue)
    pilot_probs = F.softmax(pilot_probs_1, dim=-1)  # [batch_size, num_ue, tau_p]

    pilot_index = torch.argmax(pilot_probs, dim=-1)  # [batch_size, num_ue]
    pilot_index_one_hot = F.one_hot(pilot_index, num_classes=tau_p)
    pilot_index_one_hot = pilot_index_one_hot.float()
    # pilot_probs_process = pilot_probs * torch_one_hot
    for i in range(batch_sz):
        sum_rate_each_batch[i] = torch.sum(dl_rate_calculator_w_pilot_probs(pilot_index_one_hot[i], beta[i], tau_p))
    return torch.mean(sum_rate_each_batch) * (-1)


def process_pilot_probs(beta, pilot_probs, tau_p):
    # print(beta.size())
    # print(pilot_probs.size())
    num_ap, num_ue = beta.shape
    eta = torch.rand(num_ap, num_ue)
    pilot_probs = pilot_probs.unsqueeze(1)
    A = torch.zeros((num_ue, num_ue, tau_p, tau_p))
    for k in range(num_ue):
        for k1 in range(num_ue):
            A[k, k1, :, :] = torch.matmul(pilot_probs[k].T, pilot_probs[k1])
    # print(A.size())
    gamma = torch.zeros((num_ap, num_ue, tau_p, tau_p))
    for m in range(num_ap):
        for k in range(num_ue):
            inner_sum = 0
            numerator = tau_p * rho_p * beta[m, k] ** 2
            for k1 in range(num_ue):
                inner_sum += beta[m, k1] * A[k, k1, :, :] ** 2
            denominator = tau_p * rho_p * inner_sum + torch.eye(tau_p)
            gamma[m, k, :, :] = numerator / denominator
    # print(gamma.size())
    sinr = torch.zeros(num_ue)
    for k in range(num_ue):
        inner_sum_ds = 0
        for m in range(num_ap):
            inner_sum_ds = eta[m, k] * gamma[m, k, :, :]
        DS = rho_d * inner_sum_ds ** 2

        inner_sum_BU = 0
        for k1 in range(num_ue):
            for m in range(num_ap):
                inner_sum_BU += eta[m, k1] * gamma[m, k1] * beta[m, k]
        BU = rho_d * inner_sum_BU

        inner_sum_UI1 = 0
        for k1 in range(num_ue):
            if k1 != k:
                inner_sum_UI2 = 0
                for m in range(num_ap):
                    inner_sum_UI2 += np.sqrt(eta[m, k1]) * gamma[m, k1, :, :]
                inner_sum_UI1 += (inner_sum_UI2 ** 2) * (A[k1, k] ** 2)
        UI = rho_d * inner_sum_UI1
        s = (DS / (UI + BU + torch.eye(tau_p)))
        # print(f'DS.size(): {DS.size()}')
        # print(f'UI.size(): {UI.size()}')
        # print(f'BU.size(): {BU.size()}\n')
        sinr[k] = s.sum()
        # print(f'sinr: \t{sinr}')
        # print(f'rate: \t{torch.log2(1 + sinr)}')
    return torch.log2(1 + sinr)


def loss_function_PA_using_probs(beta, output, batch_sz, num_ap, num_ue, tau_p):
    sum_rate_pilot_probs = torch.zeros(batch_sz)
    beta = beta.reshape(batch_sz, num_ap, num_ue)
    output = output.reshape(batch_sz, num_ue, tau_p)
    pilot_probs = F.softmax(output, dim=-1)
    for i in range(batch_sz):
        sum_rate_pilot_probs[i] = torch.sum(process_pilot_probs(beta[i], pilot_probs[i], tau_p))
    return_value = torch.mean(sum_rate_pilot_probs) * (-1)
    # print(f'return_value: {return_value.size()}')
    # print(f'return_value: {return_value}')
    return return_value


no_epochs = 10


def train_model(model, train_dataloader, test_dataloader, batch_sz, num_ap, num_ue, tau_p, num_epoch, lr, is_MLP_CNN):
    opt = optim.Adam(model.parameters(), lr=lr)
    epochs = num_epoch
    train_loss = -1 * np.ones(epochs)
    test_loss = -1 * np.ones(epochs)
    for epoch in range(epochs):
        running_loss_train = 0
        running_loss_test = 0
        model.train()
        for i, xs in enumerate(train_dataloader):
            opt.zero_grad()

            output_QCNN_train = model(xs)
            if is_MLP_CNN:
                loss = loss_function_PA(xs, output_QCNN_train, batch_sz, num_ap, num_ue, tau_p)
            else:
                loss = loss_function_PA_using_probs(xs, output_QCNN_train, batch_sz, num_ap, num_ue, tau_p)
            print(f'[{i}]/[{epoch}] loss value: {loss}')
            loss.backward()
            # print(f"Epoch {epoch + 1} - Gradients:")
            # for name, param in model.named_parameters():
                # if param.grad is not None:
                    # print(f"{name}.grad: {param.grad}")
                    # print(f"{name}.grad: {param.grad.size}")
            opt.step()
            running_loss_train += loss.item()
        # avg_loss = running_loss_train / len(train_dataloader)
        avg_loss = running_loss_train / 4
        print(f"-----> [{epoch}]/[{epochs}] Loss_evaluated_train: {avg_loss: .6f}")
        model.eval()
        train_loss[epoch] = train_loss[epoch] * avg_loss
        with torch.no_grad():
            for i, xt in enumerate(test_dataloader):
                output_QCNN_test = model(xt)
                # loss_evaluated_test = loss_function_PA_test(xt, output_QCNN_test, batch_sz, num_ap, num_ue, tau_p)
                loss_evaluated_test = loss_function_PA(xt, output_QCNN_test, batch_sz, num_ap, num_ue, tau_p)
                running_loss_test += loss_evaluated_test.item()
            # avg_loss_test = running_loss_test / len(test_dataloader)
            # print(f"[{epoch}]/[{epochs}] Loss_evaluated_testing: {avg_loss_test}")
            avg_loss_test = running_loss_test
            # print("--> Test loss over epoch {}: {:.6f}".format(epoch + 1, avg_loss_test))
            test_loss[epoch] = test_loss[epoch] * avg_loss_test
    return train_loss, test_loss


def valid_model(model, test_data, num_ap, num_ue, tau_p):
    sr_list = []
    with torch.no_grad():
        for i,xt in enumerate(test_data):
            output = model(xt)
            pilot_probs_1 = output.reshape(1, num_ue, tau_p)
            pilot_probs = F.softmax(pilot_probs_1, dim=-1)
            pilot_index = torch.argmax(pilot_probs, dim=-1)
            beta = xt[0].numpy().reshape(num_ap, num_ue)
            rate_list = calculate_dl_rate(beta, pilot_index[0].numpy(), num_ap, num_ue, tau_p)
            sr_list.append(np.sum(rate_list))


    # print(f'model/valid_model: information of xt: {xt.shape}-{type(xt)}')
    # beta = xt[0].numpy().reshape(num_ap, num_ue)
    # print(f'model/valid_model: information of xt: {xt.shape}-{type(xt)}')
    # rate_list = calculate_dl_rate(xt, pilot_index[0].numpy(), num_ap, num_ue, tau_p)
    sr_list = np.array(sr_list)
    return sr_list