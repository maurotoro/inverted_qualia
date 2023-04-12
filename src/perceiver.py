"""A Perceiver reads the colors, and has a qualia mapping

The subject needs to be able to cluster the colors by some algo and have a way
to tell something about new colors.
"""

# Class that works as a collection of colors
class Perceiver():

    def __init__(
        self,
        file: TextIO,
        inversion: Union[Callable[[pd.DataFrame], pd.DataFrame], None] = None,
    ):
        self.qualia = get_df_qualia(file)
        if inversion == None:
            pass
        else:
            try:
                self.qualia = inversion(self.qualia)
            except Exception as er:
                print(er)

    def get_palette(self) -> np.array:
        # return a palette of colors
        pass

    def cluster_perception(self):
        pass
