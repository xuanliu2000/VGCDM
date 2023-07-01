
import matplotlib.pyplot as plt
from matplotlib.pylab import mpl
import librosa
import numpy as np
from scipy.fftpack import fft
mpl.rcParams['font.sans-serif'] = ['simhei']  # 显示中文
mpl.rcParams['axes.unicode_minus'] = False  # 显示负号


def plot_np(y, patch=8, path=None, show_mode='time', sample_rate=12000):
    if show_mode=='mel':
        y = [np.squeeze(librosa.feature.melspectrogram(y=signal ,sr=sample_rate,n_fft=64,
            hop_length=16,n_mels=64)) for signal in y]
        print('melshape',y[0].shape)
        L = len(y)
        H = L // patch if L % patch == 0 else L // patch + 1

        fig, axs = plt.subplots(H, patch, sharex=True, figsize=(patch * 2, H * 2))
    elif show_mode=='fft':
        pass
    else:

        L = y.shape[0]
        H = L // patch if L % patch == 0 else L // patch + 1

        fig, axs = plt.subplots(H, patch, sharex=True, figsize=(patch * 6, H * 2))

    color = ['#0074D9', '#000000', '#0343DF', '#006400', '#E50000']

    for i in range(L):
        row = i // patch
        col = i % patch
        ax = axs[row][col]

        if show_mode=='mel':
            ax.imshow(y[i], cmap='jet', origin='lower', aspect='auto')
            # librosa.display.specshow(mel_spect, sr=fs, cmap='jet', x_axis='s', y_axis='mel', fmax=fs / 2)
            ax.set_yticks([])

        elif show_mode == 'fft':
            pass
        else:
            x = range(y.shape[-1])
            ax.plot(x, y[i].reshape(-1), c=color[0])

            ax.set_yticks([])
        ax.set_xticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')

    if show_mode=='mel':
        fig.supxlabel('时间/秒', y=0.08, fontsize=16)
        fig.supylabel('频率/Hz', x=0.09, fontsize=16)
        fig.subplots_adjust(wspace=0.1, hspace=0.1)
    elif show_mode=='fft':
        pass
    else:
        fig.supxlabel('时间/秒', y=0.08, fontsize=16)
        fig.supylabel('振幅/g', x=0.09, fontsize=16)
        fig.subplots_adjust(wspace=0, hspace=0)


    if path is not None:
        plt.savefig(path, dpi=300)
        print('photo is saved in {}'.format(path))

    plt.show()

def plot_two_np(x,y, patch=8, path=None,show_mode='time', sample_rate=12000):

    if show_mode=='time':
        L = min(y.shape[0],x.shape[0])
        H = L // patch if L % patch == 0 else L // patch + 1

        fig, axs = plt.subplots(H, patch, sharex=True, figsize=(patch * 6, H * 2))

        color = ['#0074D9', '#FF69B4', '#0343DF', '#006400', '#E50000']

        for i in range(L):
            row = i // patch
            col = i % patch
            ax = axs[row][col]


            t = range(y.shape[-1])

            ax.plot(t, y[i].reshape(-1), c=color[0], alpha=0.5)
            ax.plot(t, x[i].reshape(-1), c=color[1], alpha=0.5)

            ax.set_yticks([])
            ax.set_xticks([])
            ax.set_xlabel('')
            ax.set_ylabel('')

            fig.supxlabel('时间/秒', y=0.08, fontsize=16)
            fig.supylabel('振幅/g', x=0.09, fontsize=16)
            fig.subplots_adjust(wspace=0, hspace=0)


    elif show_mode == 'fft':

        L = min(y.shape[0], x.shape[0])

        H = L // patch if L % patch == 0 else L // patch + 1

        fig, axs = plt.subplots(H, patch, sharex=True, figsize=(patch * 6, H * 2))

        color = ['#0074D9', '#FF69B4', '#0343DF', '#006400', '#E50000']

        for i in range(L):
            row = i // patch
            col = i % patch
            ax = axs[row][col]
            x_fft = np.fft.fft(x[i].reshape(-1))
            n = len(x_fft)
            half_X = x_fft[1:n // 2]
            X_fre = np.fft.fftfreq(n)[1:n // 2]

            y_fft = np.fft.fft(y[i].reshape(-1))
            half_Y = y_fft[1:n // 2]
            Y_fre = np.fft.fftfreq(n)[1:n // 2]

            ax.plot(Y_fre, np.abs(half_Y), c=color[0], alpha=0.5)
            ax.plot(X_fre, np.abs(half_X), c=color[1], alpha=0.5)

            # ax.stem(X_fre, np.abs(half_X), linefmt='b-', markerfmt='bo')
            # ax.stem(Y_fre, np.abs(half_Y), linefmt='y-', markerfmt='ro')

            ax.set_yticks([])
            ax.set_xticks([])
            ax.set_xlabel('')
            ax.set_ylabel('')

            fig.supxlabel('频率/Hz', y=0.08, fontsize=16)

            fig.supylabel('幅度', x=0.09, fontsize=16)

            fig.subplots_adjust(wspace=0, hspace=0)



    if path is not None:
        plt.savefig(path, dpi=300)
        print('photo is saved in {}'.format(path))

    plt.show()

