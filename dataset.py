import numpy as np
import torch
from torch.utils.data import DataLoader
def one_time_dataset(Num_ap, Num_ue, tau_p, batch_size):
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
    # rho_d = P_d / noise_power
    rho_d = 1
    # rho_u = rho_p = p_u / noise_power
    rho_u = rho_p = 1 / 2
    sigma_shd = 8  # in dB
    D_cor = 0.1
    tau_c = 200

    num_ap = Num_ap
    num_ue = Num_ue
    tau_p = 5

    num_train = batch_size * 12
    num_test = batch_size * 3

    total_sample = num_train + num_test
    data = np.zeros((total_sample, num_ue * num_ap))
    data_CNN = np.zeros((total_sample, num_ap, num_ue))
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
        beta = 10 ** (pathloss / 10) * rho
        data[i] = beta.flatten()
        data_CNN[i] = beta

    batch_sz = batch_size
    data_train = data[:num_train]
    data_test = data[num_train:]
    X_train = torch.tensor(data_train).float()
    X_test = torch.tensor(data_test).float()
    train_dataset = list(X_train)
    test_dataset = list(X_test)
    data_train_CNN = data_CNN[:num_train]
    data_test_CNN = data_CNN[num_train:]
    X_train_CNN = torch.tensor(data_train_CNN).float()
    X_test_CNN = torch.tensor(data_test_CNN).float()
    train_dataset_CNN = list(X_train_CNN)
    test_dataset_CNN = list(X_test_CNN)
    train_dataloader_CNN = DataLoader(train_dataset_CNN, batch_size=batch_sz, shuffle=False, drop_last=False)
    test_dataloader_CNN = DataLoader(test_dataset_CNN, batch_size=batch_sz, shuffle=True, drop_last=False)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_sz, shuffle=False, drop_last=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_sz, shuffle=True, drop_last=False)
    test_cdf = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False)
    test_cdf_cnn = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False)
    return data_test, train_dataloader, test_dataloader, train_dataloader_CNN, test_dataloader_CNN, test_cdf, test_cdf_cnn