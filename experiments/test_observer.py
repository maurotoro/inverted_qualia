"""Make perceivers that cluster colors in an internally consistent ways.

The perceiver learns 3 clustering, a ML version of assigning one label to an
observation: "perception as discrimination", one clustering per color space.
Each is an instance of a k-means clustering, trained with 9-color clusters,
with 21 examples of each, and evaluated on 5 non observed examples. We repeat
the process 200 times to evaluate the consistency of the results.


"""
from pathlib import Path
import numpy as np

from inverted_qualia.perceiver import Perceiver


def instantiate_observers(ftoken_color=Path("assets/rgb.txt")) -> Perceiver:
    observer = Perceiver(ftoken_color)
    observer.set
    pass


def train_observers():
    pass


def eval_observers():
    pass


def results_consistency():
    pass
