
import matplotlib.pyplot as plt
from matplotlib.pylab import mpl
import librosa
import numpy as np
from matplotlib.ticker import FormatStrFormatter
mpl.rcParams['font.sans-serif'] = ['simhei']  # 显示中文
mpl.rcParams['axes.unicode_minus'] = False  # 显示负号

font = {'family': 'Times New Roman',
                'weight': 'normal',
                'size': 14}

font2 = {'family': 'Times New Roman',
                'weight': 'normal',
                'size': 10}

color = ['#0074D9', '#000000', '#0343DF', '#006400', '#E50000', '#FF69B4','#3776ab']

def plot_np(y, z=None,patch=8,  show_mode='time', sample_rate=12000,path=None,):
    fre = 1 / sample_rate
    if show_mode == 'mel':
        y = [np.squeeze(librosa.feature.melspectrogram(y=signal, sr=sample_rate, n_fft=64,
                                                      hop_length=16, n_mels=64)) for signal in y]
        print('melshape', y[0].shape)
        L = len(y)
        H = L // patch if L % patch == 0 else L // patch + 1

        fig, axs = plt.subplots(H, patch, sharex=True, sharey=True, figsize=(patch * 2, H * 2))
    elif show_mode == 'fft':
        L = y.shape[0]
        H = L // patch if L % patch == 0 else L // patch + 1

        fig, axs = plt.subplots(H, patch, sharex=True, sharey=True, figsize=(patch * 6, H * 2))

    else:
        L = y.shape[0]
        H = L // patch if L % patch == 0 else L // patch + 1
        fig, axs = plt.subplots(H, patch, sharex=True, sharey=True, figsize=(patch * 6, H * 2))


    for i in range(L):
        row = i // patch
        col = i % patch
        ax = axs[row][col]

        if show_mode == 'mel':
            ax.imshow(y[i], cmap='jet', origin='lower', aspect='auto')
            # librosa.display.specshow(mel_spect, sr=fs, cmap='jet', x_axis='s', y_axis='mel', fmax=fs / 2)
            ax.set_yticks([])

        elif show_mode == 'fft':
            y_fft = np.fft.fft(y[i].reshape(-1))
            n = len(y_fft)
            half_Y = y_fft[1:n // 2]
            Y_fre = np.fft.fftfreq(n, fre)[1:n // 2]
            ax.plot(Y_fre, np.abs(half_Y), c=color[0], alpha=0.5)

        elif show_mode == 'time':
            t = np.linspace(0, fre, y[i].reshape(-1).shape[-1])
            ax.plot(t, y[i].reshape(-1), c=color[0], alpha=0.5)

            if z is not None:
                ax2 = ax.twinx()
                color2 = color[1]
                ax2.set_ylabel('pulse', color=color2)
                if col == patch - 1:
                    ax2.set_ylim([-6, 6])
                    ax2.set_yticks([-6, 0, 6])  # 设置最右边子图的y轴刻度
                else:
                    ax2.set_yticks([])  # 其他子图不显示y轴刻度

                ax2.plot(t, z[i].reshape(-1), color=color2, alpha=0.5)
                ax2.tick_params(axis='y', labelcolor=color2)
        else:
            # Not consider time
            t = np.linspace(0, fre, y[i].reshape(-1).shape[-1])
            ax.plot(t, y[i].reshape(-1), c=color[0])

            color2 = color[1]
            if col != 1:
              ax.set_yticks([])

            if col == 0:
                ax.yaxis.set_label_position("left")
                ax2 = ax.twinx()
                color2 = color[1]
                ax2.set_ylim([-6, 6])
                ax2.plot(t, z[i].reshape(-1), color=color2, alpha=0.5)
            elif col == patch - 1:
                ax2 = ax.twinx()
                color2 = color[1]
                ax2.set_ylim([-6, 6])
                ax2.set_yticks([-6, 0, 10])  # 右边y轴
                ax2.plot(t, z[i].reshape(-1), color=color2, alpha=0.5)
                ax2.tick_params(axis='y', labelcolor=color2)
            else:
                ax2 = ax.twinx()
                ax2.set_ylim([-6, 6])
                ax2.set_yticks([])
                ax2.plot(t, z[i].reshape(-1), color=color2, alpha=0.5)
                ax2.tick_params(axis='y', labelcolor=color2)

        ax.set_xticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')
    plt.tight_layout()
    fig.subplots_adjust(wspace=0.2, hspace=0.1, left=0.1, right=0.9, top=0.9, bottom=0.1)
    fig.text(0.5, 0.04, '时间/秒', ha='center', va='center', fontsize=16)
    if show_mode == 'mel':
        fig.text(0.09, 0.5, '梅尔幅值', ha='center', va='center', rotation='vertical', fontsize=16)
    elif show_mode == 'fft':
        fig.text(0.09, 0.5, '幅度', ha='center', va='center', rotation='vertical', fontsize=16)
    else:
        fig.text(0.06, 0.5, '振幅/g', ha='center', va='center', rotation='vertical', fontsize=16)

    if path is not None:
        plt.savefig(path, dpi=300)
        print('photo is saved in {}'.format(path))

    # plt.tight_layout()
    plt.show()

def plot_single_np(y, z=None, sample_rate=25600, path=None,use_lable=None):
    fre = 1 / sample_rate
    t = np.linspace(0, fre* y.reshape(-1).shape[-1],y.reshape(-1).shape[-1])

    fig, ax1 = plt.subplots(figsize=(6,3))

    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['font.family'] = 'Times New Roman'#'Arial'

    ax1.plot(t, y.reshape(-1), c=color[-1], alpha=0.8, label='Vibration Signal')
    if use_lable is not None:
        ax1.set_xlabel('Time/s',fontdict=font)
        ax1.set_ylabel('Vibration Amplitude/g', color=color[-1],fontdict=font)
        ax1.tick_params(axis='x', labelcolor=color[-1])
    else:
        for spine in ax1.spines.values():
            spine.set_visible(False)
        ax1.set_xticks([])
        ax1.set_yticks([])

    if z is not None:
        assert z.reshape(-1).shape[-1]==y.reshape(-1).shape[-1]
        ax2 = ax1.twinx()
        ax2.plot(t, z.reshape(-1), c=color[1], alpha=0.6, label='Current Signal')
        if use_lable is not None:
            ax2.set_ylabel('Impulse Current/V', color=color[1],fontdict=font)
            ax2.tick_params(axis='y', labelcolor=color[1])

    fig.tight_layout()

    # Adding the legend
    lines, labels = ax1.get_legend_handles_labels()
    ax1.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    if z is not None:
        ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines += lines2
        labels += labels2
        fig.legend(lines, labels, loc='upper center',prop=font, bbox_to_anchor=(0.5, 1.08), ncol=2)

    if path is not None:
        plt.savefig(path, dpi=300)
        print('photo is saved in {}'.format(path))

    plt.show()

def plot_two_np(x,y,z1=None,z2=None, patch=8, path=None,show_mode='time', sample_rate=12000):
    fre=1/sample_rate
    color = ['#4A90E2', '#FF9999','#0074D9', '#FF69B4', '#0343DF', '#006400', '#E50000', '#FF69B4', '#3776ab','#000000']
    if show_mode=='time':
        L = min(y.shape[0], x.shape[0])
        H = L // patch if L % patch == 0 else L // patch + 1
        fig, axarr = plt.subplots(H, patch, sharex=True, sharey=True, figsize=(patch * 6, H * 2))
        if H == 1 and patch == 1:
            axarr = np.array([[axarr]])
        elif H == 1:
            axarr = axarr.reshape(1, -1)
        elif patch == 1:
            axarr = axarr.reshape(-1, 1)

        for i in range(L):
            row = i // patch
            col = i % patch

            ax = axarr[row][col]
            # t = range(y.shape[-1])
            t=np.linspace(0, fre, y.shape[-1])

            ax.plot(t, y[i].reshape(-1), c=color[0], alpha=0.5)
            ax.plot(t, x[i].reshape(-1), c=color[1], alpha=0.5)

            if z1 is not None:
                ax2=ax.twinx()
                color2 = color[-1]
                ax2.set_ylabel('Current', color=color2)
                ax2.set_ylim([-6,6])
                ax2.plot(t, z1[i].reshape(-1), color=color2, alpha=0.6)
                ax2.tick_params(axis='y', labelcolor=color2)
            if z2 is not None:
                ax3=ax.twinx()
                color3 = color[-1]
                ax3.set_ylabel('pulse', color=color3)
                ax3.set_ylim([-6, 6])
                ax3.plot(t, z2[i].reshape(-1), color=color3, alpha=0.6)
                ax3.tick_params(axis='y', labelcolor=color3)

            # ax.set_yticks([])
            # ax.set_xticks([])
            ax.set_xlabel('')
            ax.set_ylabel('')
            plt.tight_layout()
            # plt.style.use('seaborn-v0_8-darkgrid')
            fig.subplots_adjust(wspace=0.2, hspace=0.1, left=0.1, right=0.9, top=0.9, bottom=0.1)
            # fig.text(0.5, 0.04, 'time/second', ha='center', va='center', fontsize=16)
            # fig.text(0.09, 0.5, 'magrantide/g', ha='center', va='center', rotation='vertical', fontsize=16)

    elif show_mode == 'fft':

        L = min(y.shape[0], x.shape[0])

        H = L // patch if L % patch == 0 else L // patch + 1
        fig, axarr = plt.subplots(H, patch, sharex=True, sharey=True, figsize=(patch * 6, H * 2))
        if H == 1 and patch == 1:
            axarr = np.array([[axarr]])
        elif H == 1:
            axarr = axarr.reshape(1, -1)
        elif patch == 1:
            axarr = axarr.reshape(-1, 1)

        color = ['#4A90E2', '#FF9999','#0074D9', '#FF69B4', '#0343DF', '#006400', '#E50000']

        for i in range(L):
            row = i // patch
            col = i % patch
            ax = axarr[row][col]
            x_fft = np.fft.fft(x[i].reshape(-1))
            n = len(x_fft)
            half_X = x_fft[1:n // 2]
            X_fre = np.fft.fftfreq(n,fre)[1:n // 2]

            y_fft = np.fft.fft(y[i].reshape(-1))
            half_Y = y_fft[1:n // 2]
            Y_fre = np.fft.fftfreq(n,fre)[1:n // 2]

            ax.plot(Y_fre, np.abs(half_Y), c=color[0], alpha=0.7)
            ax.plot(X_fre, np.abs(half_X), c=color[1], alpha=0.7)

            # ax.stem(X_fre, np.abs(half_X), linefmt='b-', markerfmt='bo')
            # ax.stem(Y_fre, np.abs(half_Y), linefmt='y-', markerfmt='ro')

            # ax.set_yticks([])
            # ax.set_xticks([])
            ax.set_xlabel('')
            ax.set_ylabel('')
            plt.tight_layout()
            fig.subplots_adjust(wspace=0.2, hspace=0.1, left=0.1, right=0.9, top=0.9, bottom=0.1)
            # fig.text(0.5, 0.04, '时间/秒', ha='center', va='center', fontsize=16)
            # fig.text(0.09, 0.5, '幅度', ha='center', va='center', rotation='vertical', fontsize=28)


    if path is not None:
        plt.savefig(path, dpi=300)
        print('photo is saved in {}'.format(path))

    plt.show()

def plot_fre_time(x, y, z1=None, z2=None, patch=8, path=None, sample_rate=12000):
    fre = 1 / sample_rate
    color = ['#0074D9', '#FF69B4', '#0343DF', '#006400', '#E50000', '#FF69B4', '#3776ab', '#000000']

    L = min(y.shape[0], x.shape[0])
    H = L // patch if L % patch == 0 else L // patch + 1

    fig, axarr = plt.subplots(2*H, patch, sharex='col', figsize=(patch * 6, 2 * H * 2))
    if H == 1 and patch == 1:
        axarr = np.array([[axarr[0]], [axarr[1]]])
    elif H == 1:
        axarr = axarr.reshape(2, 1, -1)
    elif patch == 1:
        axarr = axarr.reshape(2, -1, 1)

    # Time Domain Plots
    for i in range(L):
        row = i // patch
        col = i % patch

        ax = axarr[row][col]
        t = np.linspace(0, fre, y.shape[-1])

        ax.plot(t, y[i].reshape(-1), c=color[0], alpha=0.5)
        ax.plot(t, x[i].reshape(-1), c=color[1], alpha=0.5)

        if z1 is not None:
            ax2 = ax.twinx()
            color2 = color[-1]
            ax2.set_ylabel('Voltage', color=color2)
            ax2.set_ylim([-6, 6])
            ax2.plot(t, z1[i].reshape(-1), color=color2, alpha=0.7)
            ax2.tick_params(axis='y', labelcolor=color2)
        if z2 is not None:
            ax3 = ax.twinx()
            color3 = color[-1]
            ax3.set_ylabel('Voltage', color=color3)
            ax3.set_ylim([-6, 6])
            ax3.plot(t, z2[i].reshape(-1), color=color3, alpha=0.7)
            ax3.tick_params(axis='y', labelcolor=color3)

        ax.set_xticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')

    # FFT Plots
    for i in range(L):
        row = i // patch
        col = i % patch
        ax = axarr[row*2][col]
        x_fft = np.fft.fft(x[i].reshape(-1))
        n = len(x_fft)
        half_X = x_fft[1:n // 2]
        X_fre = np.fft.fftfreq(n, fre)[1:n // 2]

        y_fft = np.fft.fft(y[i].reshape(-1))
        half_Y = y_fft[1:n // 2]
        Y_fre = np.fft.fftfreq(n, fre)[1:n // 2]

        ax.plot(Y_fre, np.abs(half_Y), c=color[0], alpha=0.7)
        ax.plot(X_fre, np.abs(half_X), c=color[1], alpha=0.7)

        ax.set_xticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')

    plt.tight_layout()
    fig.subplots_adjust(wspace=0.2, hspace=0.1, left=0.1, right=0.9, top=0.9, bottom=0.1)

    if path is not None:
        plt.savefig(path, dpi=300)
        print('photo is saved in {}'.format(path))

    plt.show()


if __name__ == 'main':
    x1= np.random.random((1,1024))
    x = np.random.randn(8, 100)
    y = np.random.randn(8, 100)
    plot_fre_time(x, y)
    print(x1)
