'''Color spaces as dataframes

Map colors into DataFrames
'''

from typing import TextIO, Union, Callable
import numpy as np
import pandas as pd
from skimage.color import rgb2hsv, rgb2lab, hsv2rgb

class color():
    # A python class that receives a name and a hexadecimal RGB value for a color,
    #  The class has as attributes the color value in RGB, CIELAB, and HSV
    def __init__(self, name: str, hex_rgb: list):
        self.name = name
        pass

    def __repr__(self) -> str:
        pass

    def set_values(self):
        # Set all the values
        pass

    def name(self) -> str:
        # return the color name
        return self.name

    def rgb(self) -> list[int]:
        # return the color in RGB space
        # in float, so range\in(0,1)
        return self.rgb

    def hsv(self) -> list[int]:
        # return the color in HSV space
        # in float, so range\in(0,1)
        return self.hsv

    def lab(self) -> list[int]:
        # return the color in CIElab space
        # in float, so range[l]\in(0,100); range[a,b]\in(-128,127)
        return self.lab

    def patch(self) -> np.array:
        # make a small path of the color
        pass


# a function that receives a csv file with two columns and one header
# First column is a color name, second the RGB value in hexadecimal

def get_df_qualia(file: TextIO) -> pd.DataFrame:
    # Load file as csv, somehow the file has an extra tab at the end
    data = pd.read_csv(
        file,
        skiprows=1,
        sep='\t',
        names=['name', 'hexa_rgb', ''],
        usecols=['name', 'hexa_rgb'],
    )
    # RGB columns
    rgb = (
        # remove the # from the hexa val
        data.loc[:, 'hexa_rgb'].str.replace('#', '')
        # turn the string into 3 values within 0-255
        .apply(lambda x: np.array([int(x[i:i+2], 16) for i in [0, 2, 4]])/255)
    ).apply(pd.Series, index=['r', 'g', 'b'])
    # HSV columns
    hsv = (
        rgb.apply(
            lambda x: rgb2hsv(x.values), axis=1)
    ).apply(pd.Series, index=['h', 's', 'v'])
    # CIElab columns
    cie = (
        rgb.apply(
            lambda x: rgb2lab(x.values, illuminant='D65', observer='2'),
            axis=1
        )
    ).apply(pd.Series, index=['c', 'i', 'e'])
    # Now get colors family and light family if there any
    # Keep first color if more than one, and same for light
    names = data.loc[:, 'name'].str.split(r'[\s,\/]').apply(pd.Series)
    lab, cnts = np.unique(names.stack().values.ravel(), return_counts=1)
    # Keep labels that appear more than 20 times: lab[cnts>20]
    #   this list:
    #       ['blue', 'bright', 'brown', 'dark', 'green', 'grey', 'light',
    #        'orange', 'pale', 'pink', 'purple', 'red', 'yellow']
    # use the color names to make a color family column
    col_fam = ['blue', 'brown', 'green', 'grey', 'orange',
               'pink', 'purple', 'red', 'yellow']
    col_col = (
        # Get all names in the color list
        names[names.isin(col_fam)]
        # drop rows and cols without values
        .dropna(how='all',axis=0).dropna(how='all', axis=1)
        # Fill with first appearance, and keep that
        .bfill(axis=1).loc[:, 0]
    ).rename('color_family')
    light_fam = ['bright', 'dark', 'light', 'pale']
    light_col = (
        # Get all names in the color list
        names[names.isin(light_fam)]
        # drop rows and cols without values
        .dropna(how='all',axis=0).dropna(how='all', axis=1)
        # Fill with first appearance, and keep that
        .bfill(axis=1).loc[:, 0]
    ).rename('light_family')
    # Concat them all
    ret = pd.concat([
        data.loc[:, 'name'],
        # Color space columns
        rgb.astype(float), hsv.astype(float), cie.astype(float),
        # The family columns
        col_col, light_col
    ],
                    axis=1).set_index('name')
    return ret


def inv_sat(df_qual: pd.DataFrame) -> pd.DataFrame:
    # return color DataFrame with inverted qualia by saturation on HSV space
    df_inv = df_qual.copy()
    df_inv.loc[:, 's'] = np.abs(df_inv.loc[:, 's'] - 1).values
    df_inv.loc[:, ['r', 'g', 'b']]= (df_inv.loc[:, ['h', 's', 'v']].apply(hsv2rgb, axis=1).apply(pd.Series).values)
    df_inv.loc[:, ['c', 'i', 'e']]= (df_inv.loc[:, ['r', 'g', 'b']].apply(rgb2lab, axis=1).apply(pd.Series).values)
    return df_inv


def inv_val(df_qual: pd.DataFrame) -> pd.DataFrame:
    df_inv = df_qual.copy()
    # A function to invert data by luminosity
    df_inv.loc[:, 'v'] = np.abs(df_inv.loc[:, 'v'] - 1).values
    df_inv.loc[:, ['r', 'g', 'b']]= (df_inv.loc[:, ['h', 's', 'v']].apply(hsv2rgb, axis=1).apply(pd.Series).values)
    df_inv.loc[:, ['c', 'i', 'e']]= (df_inv.loc[:, ['r', 'g', 'b']].apply(rgb2lab, axis=1).apply(pd.Series).values)
    return df_inv


def inv_hue(df_qual: pd.DataFrame) -> pd.DataFrame:
    df_inv = df_qual.copy()
    # A function that inverts the hues, R2L
    df_inv.loc[:, 'h'] = np.abs(df_inv.loc[:, 'h'] - 1).values
    df_inv.loc[:, ['r', 'g', 'b']]= (df_inv.loc[:, ['h', 's', 'v']].apply(hsv2rgb, axis=1).apply(pd.Series).values)
    df_inv.loc[:, ['c', 'i', 'e']]= (df_inv.loc[:, ['r', 'g', 'b']].apply(rgb2lab, axis=1).apply(pd.Series).values)
    return df_inv


def rand_hue(df_qual: pd.DataFrame) -> pd.DataFrame:
    df_inv = df_qual.copy()
    # A function to randomize hues
    df_inv.loc[:, 'h'] = df_inv.loc[:, 'h'].sample(frac=1., replace=0).values
    df_inv.loc[:, ['r', 'g', 'b']]= (df_inv.loc[:, ['h', 's', 'v']].apply(hsv2rgb, axis=1).apply(pd.Series).values)
    df_inv.loc[:, ['c', 'i', 'e']]= (df_inv.loc[:, ['r', 'g', 'b']].apply(rgb2lab, axis=1).apply(pd.Series).values)
    return df_inv


def rand_all(df_qual: pd.DataFrame) -> pd.DataFrame:
    df_inv = df_qual.copy()
    # A function that randomizes all value pairs
    # Just change the index, ceteris paribus
    ixs = pd.Index(df_inv.index.to_frame().sample(frac=1., replace=0).values)
    df_inv = df_inv.set_index(ixs)
    return df_inv
