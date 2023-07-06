
import matplotlib.pyplot as plt
from matplotlib.pylab import mpl
import librosa
import numpy as np
from scipy.fftpack import fft
mpl.rcParams['font.sans-serif'] = ['simhei']  # 显示中文
mpl.rcParams['axes.unicode_minus'] = False  # 显示负号

def plot_np(y, z=None,patch=8, path=None, show_mode='time', sample_rate=12000):
    fre=1/sample_rate
    if show_mode=='mel':
        y = [np.squeeze(librosa.feature.melspectrogram(y=signal ,sr=sample_rate,n_fft=64,
            hop_length=16,n_mels=64)) for signal in y]
        print('melshape',y[0].shape)
        L = len(y)
        H = L // patch if L % patch == 0 else L // patch + 1

        fig, axs = plt.subplots(H, patch, sharex=True, figsize=(patch * 2, H * 2))
    elif show_mode=='fft':
        L = y.shape[0]
        H = L // patch if L % patch == 0 else L // patch + 1

        fig, axs = plt.subplots(H, patch, sharex=True, figsize=(patch * 6, H * 2))

    else:

        L = y.shape[0]
        H = L // patch if L % patch == 0 else L // patch + 1

        fig, axs = plt.subplots(H, patch, sharex=True, figsize=(patch * 6, H * 2))

    color = ['#0074D9', '#000000', '#0343DF', '#006400', '#E50000','#FF69B4']

    for i in range(L):
        row = i // patch
        col = i % patch
        ax = axs[row][col]

        if show_mode=='mel':
            ax.imshow(y[i], cmap='jet', origin='lower', aspect='auto')
            # librosa.display.specshow(mel_spect, sr=fs, cmap='jet', x_axis='s', y_axis='mel', fmax=fs / 2)
            ax.set_yticks([])

        elif show_mode == 'fft':
            y_fft = np.fft.fft(y[i].reshape(-1))
            n = len(y_fft)
            half_Y = y_fft[1:n // 2]
            Y_fre = np.fft.fftfreq(n,fre)[1:n // 2]
            ax.plot(Y_fre, np.abs(half_Y), c=color[0], alpha=0.5)

        elif show_mode=='time':
            t = np.linspace(0,fre, y[i].reshape(-1).shape[-1])
            ax.plot(t, y[i].reshape(-1), c=color[0])
            ax.set_yticks([])
            if z is not None:
                ax2=ax.twinx()
                color2 = color[1]
                ax2.set_ylabel('pulse', color=color2)
                ax2.plot(t, z[i].reshape(-1), color=color2, alpha=0.5)
                ax2.tick_params(axis='y', labelcolor=color2)
        else:
            # Not consider time
            t = np.linspace(0, fre,y[i].reshape(-1).shape[-1])
            ax.plot(t, y[i].reshape(-1), c=color[0])
            ax.set_yticks([])
            if z is not None:
                ax2=ax.twinx()
                color2 = color[1]
                ax2.set_ylabel('pulse', color=color2)
                ax2.plot(t, z[i].reshape(-1), color=color2, alpha=0.5)
                ax2.tick_params(axis='y', labelcolor=color2)
        ax.set_xticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')

    if show_mode=='mel':
        fig.supxlabel('时间/秒', y=0.08, fontsize=16)
        fig.supylabel('梅尔幅值', x=0.09, fontsize=16)
        fig.subplots_adjust(wspace=0.1, hspace=0.1)
    elif show_mode=='fft':
        fig.supxlabel('频率/Hz', y=0.08, fontsize=16)
        fig.supylabel('幅度', x=0.09, fontsize=16)
        fig.subplots_adjust(wspace=0, hspace=0)
    else:
        fig.supxlabel('时间/秒', y=0.08, fontsize=16)
        fig.supylabel('振幅/g', x=0.09, fontsize=16)
        fig.subplots_adjust(wspace=0, hspace=0)

    if path is not None:
        plt.savefig(path, dpi=300)
        print('photo is saved in {}'.format(path))

    plt.show()


def plot_two_np(x,y,z1=None,z2=None, patch=8, path=None,show_mode='time', sample_rate=12000):
    fre=1/sample_rate
    if show_mode=='time':
        L = min(y.shape[0],x.shape[0])
        H = L // patch if L % patch == 0 else L // patch + 1

        fig, axs = plt.subplots(H, patch, sharex=True, figsize=(patch * 6, H * 2))

        color = ['#0074D9', '#FF69B4', '#0343DF', '#E50000']

        for i in range(L):
            row = i // patch
            col = i % patch
            ax = axs[row][col]
            # t = range(y.shape[-1])
            t=np.linspace(0, fre, y.shape[-1])

            ax.plot(t, y[i].reshape(-1), c=color[0], alpha=0.5)
            ax.plot(t, x[i].reshape(-1), c=color[1], alpha=0.5)
            if z1 is not None:
                ax2=ax.twinx()
                color2 = color[3]
                ax2.set_ylabel('pulse', color=color2)
                ax2.plot(t, z1[i].reshape(-1), color=color2, alpha=0.5)
                ax2.tick_params(axis='y', labelcolor=color2)
            if z2 is not None:
                ax3=ax.twinx()
                color3 = color[2]
                ax3.set_ylabel('pulse', color=color3)
                ax3.plot(t, z2[i].reshape(-1), color=color3, alpha=0.5)
                ax3.tick_params(axis='y', labelcolor=color3)

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
            X_fre = np.fft.fftfreq(n,fre)[1:n // 2]

            y_fft = np.fft.fft(y[i].reshape(-1))
            half_Y = y_fft[1:n // 2]
            Y_fre = np.fft.fftfreq(n,fre)[1:n // 2]

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


