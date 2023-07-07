"""A Perceiver reads the colors, and has a qualia mapping

The subject needs to be able to cluster the colors by some algo and have a way
to tell something about new colors.
"""

from typing import Union, Callable, TextIO
from pathlib import Path
import sklearn.cluster as clst
from sklearn import svm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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
                    for x, lab in zip(axlabs, ls)
                })
        return fig, axs

    def cluster_perception(self):
        pass

    def set_classification_frames(self):
        # learn a mapping of prototypes
        model = svm.LinearSVC()
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

    def test_classification(self):
        # use the learned classification scheme to the rest
        pass


if __name__ == "__main__":
    file_color = Path("assets/rgb.txt")
