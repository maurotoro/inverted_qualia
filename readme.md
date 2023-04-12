# What?

An exploration on the thought experiment of the [inverted qualia](https://plato.stanford.edu/entries/qualia-inverted/).

The general idea is that it's possible for there to exist a person with the same physical and behavioral responses as someone without any vision impairtent. Yet, on their internal forum, they have opposite subjective experiences.

In principle, given that *qualia* are personal, subjective, ineffable, and inaccessible, this should be possible. But, it's not the case that internal experiences have these attributes. From a long time in experimental psychology we have developed tools to look at, and get correlates of internal experiences. One behavioral framework from where we can get access to this internal experiences is [psychophysics](https://en.wikipedia.org/wiki/Psychophysics). Within this framework, one can get access to the properties of perceptual systems and observer experiences.

As a first step, we take the empirical color-names $\to$ RGB value mappings that XKCD got from a large enough sample of nerds[1], and evaluate get some clusterings in color space. We later map colors into vector spaces, using BERT because it's cheap and GoodEnoug™.

We will try to show what happens for different kinds of inverted qualia scenarios. The RGB colorspace is comfortable to map colors as if they where paint, but there are other [colorspaces](https://en.wikipedia.org/wiki/Color_space) that are more meaninful with respect to our color experience. For example the [HSV space](https://en.wikipedia.org/wiki/HSL_and_HSV), Hue-Saturation-Value, maps colors into a 3D space defined by a Hue, what we regularly call colors, Saturation, how present is the color, and Value, how much present are these value. Another relevant space for our exploration is the [CIEXYZ](https://en.wikipedia.org/wiki/CIE_1931_color_space) spaces, a set of spaces defined quatintavely to link wavelenght distribution into physiologically perceived colors in human vision.

By changing the original RGB mappings into new ones, but respecting the original labeling, we can model how the hypothetical inverted-spectrum subject would behave. We should be capable of modelling behavioral responses for him using different ML techniques and maybe detect which cases would allow for her to really be undetectable.



[1]: [This is the original post](https://xkcd.com/color/rgb), and [this the dataset](https://xkcd.com/color/rgb.txt)
