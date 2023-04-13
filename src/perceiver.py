"""A Perceiver reads the colors, and has a qualia mapping

The subject needs to be able to cluster the colors by some algo and have a way
to tell something about new colors.
"""

from typing import Union, Callable, TextIO
import sklearn.cluster as clst
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from .color_spaces import (get_df_qualia, inv_sat, inv_val, inv_hue, rand_hue,
                           rand_all)

# Class that works as a collection of colors
class Perceiver():
    # Available inversions
    __inversions = (inv_sat, inv_val, inv_hue, rand_hue, rand_all)
    # Color space labels
    __labels = labs = [[y for y in x] for x in ['rgb', 'hsv', 'cie']]

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
        elif inversion in self.__inversions:
            self.qualia = inversion(self.qualia)
        else:
            raise NotImplementedError('The requested inversion is not implemented.')

    @property
    def rgb(self) -> pd.DataFrame:
        # return the RGB values of the qualia space
        return self.qualia.loc[:, self.__labels[0]]

    @property
    def hsv(self) -> pd.DataFrame:
        # return the HSV values of the qualia space
        return self.qualia.loc[:, self.__labels[1]]

    @property
    def cie(self) -> pd.DataFrame:
        # return the CIElab values of the qualia space
        return self.qualia.loc[:, self.__labels[2]]

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
        labs = self.__labels
        # Plot qualia prototype groups
        _ = ([
            ax.scatter3D(
                *self.qualia.loc[self.qualia.loc[:, 'color_family'] == c, lab].values.T,
                edgecolor=ser.loc[lab].values,
                facecolor='w',
                s=75,
            ) for c, ser in self.prototypes.iterrows()
            ]
            for ax, lab in zip(axs, self.__labels)
        )

        # Plot all qualia
        _ = (
            ax.scatter3D(
                *self.qualia.loc[:, lab].values.T,
                s=50,
                c=self.prototypes.loc[:, labs[0]].values,
            ) for ax, lab in zip(axs, labs)
        )
        # Plot the prototype colors, larger and with black edges
        _ = (
            ax.scatter3D(
                *self.prototypes.loc[:, lab].values.T,
                s=150,
                edgecolor='k',
                c=self.prototypes.loc[:, labs[0]].values,
            ) for ax, lab in zip(axs, labs)
        )
        # Add labels to plot axes
        axlabs = ['xlabel', 'ylabel', 'zlabel']
        _ = (ax.set(
            **{
                x: r'$\bf{' + f'{lab.upper()}' + '}$'
                for x, lab in zip(axlabs, ls)
            }) for ax, ls in zip(axs, labs))
        return fig, axs

    def cluster_perception(self):
        pass


    def learn_classification(self):
        # learn a mapping of prototypes
        pass

    def test_classification(self):
        # use the learned classification scheme to the rest
        pass
