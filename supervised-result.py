import numpy as np
import matplotlib.pyplot as plt

# Load the saved arrays
data_CNN = np.load('cnn_training_results.npz')

trloss_CNN = data_CNN['trloss_CNN']
teloss_CNN = data_CNN['teloss_CNN']
trrate_CNN = data_CNN['trrate_CNN']
terate_CNN = data_CNN['terate_CNN']

print("Results loaded successfully.")
# print(trloss_CNN)

data_HQCNN = np.load('hqcnn_training_results.npz')

trloss_HQCNN = data_HQCNN['trloss_HQCNN']
teloss_HQCNN = data_HQCNN['teloss_HQCNN']
trrate_HQCNN = data_HQCNN['trrate_HQCNN']
terate_HQCNN = data_HQCNN['terate_HQCNN']

print("Results loaded successfully.")
data_MLP = np.load('mlp_training_results.npz')

trloss_MLP = data_MLP['trloss_MLP']
teloss_MLP = data_MLP['teloss_MLP']
trrate_MLP = data_MLP['trrate_MLP']
terate_MLP = data_MLP['terate_MLP']
# print(trloss_CNN)

data_system = np.load('system_training_results.npz')
train_avg_rate = data_system['train_avg_rate']
test_avg_rate = data_system['test_avg_rate']
hqcnn_time = data_system['hqcnn_time']
mlp_time = data_system['mlp_time']
cnn_time = data_system['cnn_time']


def plot_loss_1(trloss_HQCNN, trloss_MLP, trloss_CNN):
    plt.plot(trloss_HQCNN, label='HQCNN')
    # plt.plot(trloss_MLP, label='MLP')
    # plt.plot(trloss_CNN, label='CNN')
    # plt.plot(teloss, label='Test loss')
    plt.legend()

    # Add x-axis and y-axis labels
    plt.xlabel('Iteration')
    plt.ylabel('Loss value')

    # Display the plot
    plt.show()


def plot_loss_2(trloss_HQCNN, trloss_MLP, trloss_CNN):
    # plt.plot(trloss_HQCNN, label='HQCNN')
    plt.plot(trloss_MLP, label='MLP')
    # plt.plot(trloss_CNN, label='CNN')
    # plt.plot(teloss, label='Test loss')
    plt.legend()

    # Add x-axis and y-axis labels
    plt.xlabel('Iteration')
    plt.ylabel('Loss value')

    # Display the plot
    plt.show()


def plot_loss_3(trloss_HQCNN, trloss_MLP, trloss_CNN):
    # plt.plot(trloss_HQCNN, label='HQCNN')
    # plt.plot(trloss_MLP, label='MLP')
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
    thr = threshold * np.ones(len(result_HQCNN))
    plt.plot(thr, label='Master-AP')
    plt.legend()
    if is_train:
        plt.title('Training')
    else:
        plt.title('Testing')
    plt.xlabel('Iteration')
    plt.ylabel('Total system rate')
    plt.show()


# plot_loss_1(trloss_HQCNN, trloss_MLP, trloss_CNN)
# plot_loss_2(trloss_HQCNN, trloss_MLP, trloss_CNN)
# plot_loss_3(trloss_HQCNN, trloss_MLP, trloss_CNN)
# plot_rate(train_avg_rate, trrate_HQCNN, trrate_MLP, trrate_CNN, is_train=True)
# plot_rate(test_avg_rate, terate_HQCNN, terate_MLP, terate_CNN, is_train=False)


print(trloss_MLP)



