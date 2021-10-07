import torch
import matplotlib.pyplot as plt
import numpy as np

def get_device():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda":
        n_gpu = torch.cuda.device_count()
        torch.cuda.get_device_name(0)
        device_name = tf.test.gpu_device_name()
        if device_name != '/device:GPU:0':
            raise SystemError('GPU device not found')
            print('Found GPU at: {}'.format(device_name))

    else:
        print(f'Currently Running on {device}')
    
    return device


def seed_all(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def plot_model(data):

    plt.plot(loss_qna.tolist())
    plt.title('Bi-Encoders Loss')

    plt.show()
    # loss_qna.tolist()

    loss_bart = torch.load('/content/drive/MyDrive/END2_CAPSTONE/checkpoint-103/training_loss.pt')
    plt.title('BART Loss')

    plt.plot(loss_bart.tolist())
    plt.show()