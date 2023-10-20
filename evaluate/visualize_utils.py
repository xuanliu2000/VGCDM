import os
import time
import torch
import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from matplotlib.patches import Patch


class TSNEVisualizer:
    def __init__(self, model, dataloader, device):
        self.model = model
        self.dataloader = dataloader
        self.device = device

    def get_embeddings(self):
        self.model.eval()
        embeddings = []
        labels = []
        with torch.no_grad():
            for inputs, targets,_ in self.dataloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                outputs = self.model(inputs,return_features=True)
                embeddings.append(outputs.cpu().numpy())
                labels.append(targets.cpu().numpy())
        embeddings = np.vstack(embeddings)
        print('emb.shape',embeddings.shape)
        labels = np.concatenate(labels)
        return embeddings, labels

    def visualize(self, n_components=2, perplexity=30.0, random_state=42, fig_out=False):
        embeddings, labels = self.get_embeddings()
        tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state)
        tsne_results = tsne.fit_transform(embeddings)

        fig = plt.figure(figsize=(8, 8))
        plt.rc('font', family='Times New Roman', style='normal', weight='light', size=10)
        font = {'family': 'Times New Roman',
                'weight': 'normal',
                'size': 18}

        cmap = plt.cm.get_cmap("Pastel1", 10)
        plt.scatter(tsne_results[:, 0],
                              tsne_results[:, 1],
                              c=labels,
                              cmap=cmap,
                              marker='o')

        # Create legend handles manually
        unique_labels = np.unique(labels)
        legend_elements = [Patch(facecolor=cmap(i), edgecolor=cmap(i),
                                 label=f'{i}') for i in unique_labels]

        # Create a legend
        legend1 = plt.legend(handles=legend_elements,
                             loc="upper right",
                             prop=font,)

        plt.gca().add_artist(legend1)

        plt.clim(-0.5, 9.5)
        plt.xticks([])
        plt.yticks([])
        if fig_out:
            plt.savefig(fig_out, dpi=300, bbox_inches='tight')
            print('Save confusion tsne.png to \n', fig_out)
            plt.show()
        else:
            plt.show()



