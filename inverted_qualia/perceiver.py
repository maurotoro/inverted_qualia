"""A Perceiver reads the colors, and has a qualia mapping

The subject needs to be able to cluster the colors by some algo and have a way
to tell something about new colors.
"""

from typing import Union, Callable, TextIO
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import sklearn.metrics as metrics
import sklearn.cluster as clst
from sklearn import svm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances, pairwise_distances_argmin

from inverted_qualia.color_spaces import (get_df_qualia, inv_sat, inv_val, inv_hue, rand_hue,
                           rand_all)

# Class that works as a collection of colors
class Perceiver():
    # Available inversions
    _inversions = (inv_sat, inv_val, inv_hue, rand_hue, rand_all)
    # Color space labels
    _labels = labs = [[y for y in x] for x in ['rgb', 'hsv', 'cie']]
    # classifier state
    _learned = False

    def __init__(
        self,
        file: TextIO,
        inversion: Union[Callable[[pd.DataFrame], pd.DataFrame], None] = None,
    ):
        # Set qualia
        self.qualia, self.prototypes = self.set_qualia(file)
        # If needed set an inversion
        if inversion == None:
            pass
        elif inversion in self._inversions:
            self.qualia = inversion(self.qualia)
        else:
            raise NotImplementedError('The requested inversion is not implemented.')

        # Set the dataframes for train and test
        self.set_classification_frames()

    @property
    def rgb(self) -> pd.DataFrame:
        # return the RGB values of the qualia space
        return self.qualia.loc[:, self._labels[0]]

    @property
    def hsv(self) -> pd.DataFrame:
        # return the HSV values of the qualia space
        return self.qualia.loc[:, self._labels[1]]

    @property
    def cie(self) -> pd.DataFrame:
        # return the CIElab values of the qualia space
        return self.qualia.loc[:, self._labels[2]]

    def set_qualia(self, file: TextIO) -> list[pd.DataFrame]:
        df_qual = get_df_qualia(file)
        df_prototype = (
            df_qual.loc[df_qual.loc[:, 'color_family'].dropna().unique(), :]
        )
        return df_qual, df_prototype

    def plot_qualia(self) -> list[plt.figure, np.array]:
        # Plot the qualia space of the perceiver
        fig, axs = plt.subplots(
            nrows=1,
            ncols=3,
            figsize=[15, 7],
            subplot_kw=dict(projection='3d'),
        )
        # Labels for color spaces
        labs = self._labels
        # Plot qualia prototype groups
        qualia, prototypes = self.qualia, self.prototypes
        axlabs = ['xlabel', 'ylabel', 'zlabel']
        for ax, lab in zip(axs, labs):
            # First plot the colors with prototypical labels
            _ = [
                ax.scatter3D(
                    *qualia.loc[qualia.loc[:, 'color_family'] == c,
                                lab].values.T,
                    edgecolor=ser.loc[labs[0]].values,
                    facecolor='w',
                    s=75,
                ) for c, ser in prototypes.iterrows()
            ]
            # Add the prototypes, larger and filled
            _ = ax.scatter3D(
                *prototypes.loc[:, lab].values.T,
                s=150,
                edgecolor='k',
                c=prototypes.loc[:, labs[0]].values,
            )
            # Add labels to plot axes
            _ = ax.set(
                **{
                    x: r'$\bf{' + f'{lab.upper()}' + '}$'
                    for x, lab in zip(axlabs, labs)
                })
        return fig, axs

    def cluster_perception(self, show=False, score_metric="v_measure_score"):
        n_colors = self.prototypes.shape[0]
        acc_metric = getattr(metrics, score_metric)
        ret = []
        for label in self._labels:
            clust = clst.KMeans(n_clusters=n_colors, n_init=n_colors)
            train = self.train.loc[:, label]
            train_l = self.train.loc[:, 'color_family']
            test = self.test.loc[:, label]
            test_l = self.test.loc[:, 'color_family']
            df_eval = self.eval.dropna().loc[:, label]
            df_eval_l = self.eval.dropna().loc[:, 'color_family']
            prototypes = self.prototypes.loc[:, label]
            _ = clust.fit(train.values)
            # Compare the prototypes and the cluster centres, get closest labels
            idx_2_ids = pairwise_distances_argmin(
                clust.cluster_centers_,
                prototypes.values,
            )
            dm = pairwise_distances(
                clust.cluster_centers_,
                prototypes,
            )
            # Map the labels closer to centers ordered by distances
            clust_labels = prototypes.iloc[idx_2_ids].index
            # give prediction of the test labels
            predictions = clust.predict(test.values)
            score = acc_metric(test_l.values,
                                 clust_labels[predictions].values)
            ret.append(score)
            if show:
                fig, axs = plt.subplots(3, 1)
                _ = fig.suptitle("".join(label))
                pca = PCA(n_components=2)
                dim_red = pca.fit_transform(train)
                _ = axs[0].scatter(*dim_red.T, c=train_l)
                _ = axs[0].set_title('train')
                _ = axs[1].scatter(*pca.transform(test).T,
                                c=clust_labels[predictions])
                _ = axs[1].set_title('test')
                _ = axs[2].scatter(*pca.transform(df_eval).T,
                                c=df_eval_l)
                _ = axs[2].set_title('eval')
        return ret

    def set_classification_frames(self):
        # split train and test, use prototypes to train, always
        df_prototype = self.prototypes
        df_qualia = self.qualia
        df_qualia = df_qualia.loc[~df_qualia.index.isin(df_prototype.index), :]
        df_test = df_qualia.groupby('color_family').sample(n=5)
        df_train = (
            df_qualia.loc[~df_qualia.index.isin(df_test.index), :]
            .query('~color_family.isnull()')
            .groupby('color_family')
            .sample(n=20)
        )
        df_train = pd.concat([df_train, df_prototype])
        # Now for eval the rest
        df_eval =  df_qualia.loc[~df_qualia.index.isin(np.concatenate([df_train.index, df_test.index])), :]
        # Set them as properties
        self.train = df_train
        self.test = df_test
        self.eval = df_eval

    def test_classification(self, score_metric="v_measure_score", n_iter=200):
        # use the learned classification scheme to the rest
        ret = [
            self.cluster_perception(score_metric=score_metric)
            for _ in range(n_iter)
        ]
        return ret


if __name__ == "__main__":
    file_color = Path("assets/rgb.txt")
    observer = Perceiver(file_color)
    scores = observer.test_classification()
