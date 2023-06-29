
import matplotlib.pyplot as plt
from matplotlib.pylab import mpl
import librosa
import numpy as np
mpl.rcParams['font.sans-serif'] = ['simhei']  # 显示中文
mpl.rcParams['axes.unicode_minus'] = False  # 显示负号


def plot_np(y, patch=8, path=None, show_mel=False, sample_rate=12000):
    if show_mel:
        y = [np.squeeze(librosa.feature.melspectrogram(y=signal ,sr=sample_rate,n_fft=64,
            hop_length=16,n_mels=64)) for signal in y]
        print('melshape',y[0].shape)
        L = len(y)
        H = L // patch if L % patch == 0 else L // patch + 1

        fig, axs = plt.subplots(H, patch, sharex=True, figsize=(patch * 2, H * 2))
    else:

        L = y.shape[0]
        H = L // patch if L % patch == 0 else L // patch + 1

        fig, axs = plt.subplots(H, patch, sharex=True, figsize=(patch * 6, H * 2))

    color = ['#0074D9', '#000000', '#0343DF', '#006400', '#E50000']

    for i in range(L):
        row = i // patch
        col = i % patch
        ax = axs[row][col]

        if show_mel:
            ax.imshow(y[i], cmap='jet', origin='lower', aspect='auto')
            # librosa.display.specshow(mel_spect, sr=fs, cmap='jet', x_axis='s', y_axis='mel', fmax=fs / 2)
            ax.set_yticks([])
        else:
            x = range(y.shape[-1])
            ax.plot(x, y[i].reshape(-1), c=color[0])

            ax.set_yticks([])
        ax.set_xticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')

    if show_mel:
        fig.supxlabel('时间/秒', y=0.08, fontsize=16)
        fig.supylabel('频率/Hz', x=0.09, fontsize=16)
        fig.subplots_adjust(wspace=0.1, hspace=0.1)
    else:
        fig.supxlabel('时间/秒', y=0.08, fontsize=16)
        fig.supylabel('振幅/g', x=0.09, fontsize=16)
        fig.subplots_adjust(wspace=0, hspace=0)


    if path is not None:
        plt.savefig(path, dpi=300)

    plt.show()