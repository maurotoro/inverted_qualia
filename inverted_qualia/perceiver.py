"""A Perceiver reads the colors, and has a qualia mapping

The subject needs to be able to cluster the colors by some algo and have a way
to tell something about new colors.
"""

import numpy as np
import pandas as pd
import sklearn.cluster as clst
import sklearn.metrics as metrics
import sklearn.manifold as skmanif
import matplotlib.pyplot as plt

from typing import Union, Callable, TextIO
from sklearn import svm
from pathlib import Path
from sklearn.metrics import pairwise_distances, pairwise_distances_argmin, accuracy_score
from sklearn.decomposition import PCA
from tqdm.auto import tqdm

from inverted_qualia.color_spaces import (
    get_df_qualia, inv_sat, inv_val, inv_hue, rand_hue, rand_all
)


# Class that works as a collection of colors
class Perceiver():
    # Available inversions
    _inversions = [inv_sat, inv_val, inv_hue, rand_hue, rand_all]
    # Color space labels
    _labels = labs = [[y for y in x] for x in ['rgb', 'hsv', 'cie']]
    # classifier state
    _learned = False

    def __init__(
        self,
        file: TextIO,
        inversion: Callable | None = None,
    ) -> None:
        # Set qualia
        self.qualia, self.prototypes = self.set_qualia(file, inversion)
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

    def set_qualia(
        self,
        file: TextIO,
        inversion: Callable | None = None,
    ) -> list[pd.DataFrame]:
        # Set the qualia dataframe
        df_qual = get_df_qualia(file)
        # If needed do an inversion
        if inversion == None:
            pass
        elif inversion in self._inversions:
            df_qual = inversion(df_qual)
        else:
            raise NotImplementedError('The requested inversion is not implemented.')
        # Set the prototypes
        df_prototype = (
            df_qual.loc[df_qual.loc[:, 'color_family'].dropna().unique(), :]
        )
        return df_qual, df_prototype

    def plot_qualia_3d(self) -> list[plt.figure, np.array]:
        # Labels for color spaces
        labels = self._labels
        # Plot qualia prototype groups
        df_qualia, prototypes = self.qualia.dropna(subset='color_family'), self.prototypes
        df_rgb = df_qualia.loc[:, ['r', 'g', 'b']]
        axlabs = ['xlabel', 'ylabel', 'zlabel']
        # Plot the qualia space of the perceiver
        fig, axs = plt.subplots(
            nrows=1,
            ncols=3,
            figsize=[16, 5],
            subplot_kw=dict(projection='3d'),
        )
        for ax, lab in zip(axs.ravel(), labels):
            df = df_qualia.loc[:, lab]
            c = df_rgb.values
            edgecolors = df_qualia.loc[:, 'color_family'].values
            ax.scatter3D(*df.values.T, c=c, edgecolors=edgecolors)

            _ = ax.set(
                **{
                    x: r'$\bf{' + f'{l.upper()}' + '}$'
                    for x, l in zip(axlabs, lab)
                })
        return fig, axs

    def plot_qualia_2d(self) -> list[plt.figure, np.array]:
        df_qualia = self.qualia.dropna(subset='color_family')
        labels = self._labels
        df_rgb = df_qualia.loc[:, ['r', 'g', 'b']]
        nrows = np.ceil(df_rgb.shape[0] / 10).astype(int)
        fig, axs = plt.subplots(ncols=3, nrows=1, figsize=[16, 5])
        for lab, ax in zip(labels, axs.ravel()):
            df = df_qualia.loc[:, lab]
            pca = PCA(n_components=2)
            c = df_rgb.values
            edgecolors = df_qualia.loc[:, 'color_family'].values
            dim_red = pca.fit_transform(df.values)
            ax.scatter(*dim_red.T, c=c, edgecolors=edgecolors)
        return fig, axs.ravel()


    def manifold_cluster(self, show: bool = False):
        n_colors = self.prototypes.shape[0]
        acc_metric = getattr(metrics, score_metric)
        ret = []
        if show:
            fig, axs = plt.subplots(nrows=2, ncols=3, figsize=[16, 5])
        labels = self._labels
        for x, label in enumerate(labels):
            # Get DFs
            train = obs_.train.loc[:, label]
            df_qualia = obs_.qualia.dropna(
                subset='color_family').loc[:, label]
            df_qualia_l = obs_.qualia.dropna(
                subset='color_family').loc[:, 'color_family']
            # prototypes as
            df_prototypes = (
                pd.concat([df_qualia, df_qualia_l],axis=1)
                .groupby('color_family')
                .median()
            )
            # Learn a map for the data that
            manif_ = skmanif.Isomap(n_neighbors=10, n_components=2)
            learn_s = manif_.fit_transform(df_qualia.values)
            # Cluster on the new space
            clustering = clst.KMeans(
                n_clusters=df_prototypes.shape[0],
                n_init=10,
            )
            preds = clustering.fit_predict(learn_s)
            ixs_ql = df_qualia_l.unique()[preds]
            idx_2_ids = pairwise_distances_argmin(
                clustering.cluster_centers_,
                df_prototypes.values,
            )
            dm = pairwise_distances(
                clustering.cluster_centers_,
                df_prototypes,
            )

            if show:
                ax = axs[0, x]
                c = observer.qualia.loc[df_qualia.index, ['r', 'g', 'b']].values
                edgecolors = observer.qualia.loc[df_qualia.index,
                                                 'color_family'].values
                _ = ax.scatter(*learn_s.T, c=c, edgecolors=edgecolors)

        pass

    def cluster_perception(self, show=False, score_metric="v_measure_score"):
        n_colors = self.prototypes.shape[0]
        acc_metric = getattr(metrics, score_metric)
        ret = []
        for label in self._labels:
            train = self.train.loc[:, label]
            train_l = self.train.loc[:, 'color_family']
            test = self.test.loc[:, label]
            test_l = self.test.loc[:, 'color_family']
            df_eval = self.eval.dropna().loc[:, label]
            df_eval_l = self.eval.dropna().loc[:, 'color_family']
            prototypes = self.prototypes.loc[:, label]
            clust = clst.KMeans(n_clusters=n_colors, n_init=n_colors)
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

    def plot_2d_prototypes(self) -> list[plt.figure, np.array]:
        fig, axs = plt.subplots(ncols=3, nrows=3, )
        df_rgbs = self.prototypes.loc[:, ['r', 'g', 'b']]
        for ax, (c_n, df) in zip(axs.ravel(), df_rgbs.iterrows()):
            ax.set_facecolor(df.values)
            ax.set_title(c_n)
            ax.set(xticks=[], yticks=[])
        return fig, axs.ravel()

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

    def test_classification(self, score_metric="v_measure_score", n_iter=10):
        # use the learned classification scheme to the rest
        ret = [
            self.cluster_perception(score_metric=score_metric)
            for _ in tqdm(range(n_iter))
        ]
        return ret


def compare_qualia_spaces(observers: list[Perceiver]):
    dfs_train = [obs.train for obs in observers]
    common_ixs = observer[0]
    pass


if __name__ == "__main__":
    file_color = Path("assets/rgb.txt")
    observer = Perceiver(file_color)
    observer_ih = Perceiver(file_color, inv_hue)
    observer_is = Perceiver(file_color, inv_sat)
    observer_iv = Perceiver(file_color, inv_val)
    scores = observer.test_classification()
    scores_ih = observer_ih.test_classification()
    scores_is = observer_is.test_classification()
    scores_iv = observer_iv.test_classification()
