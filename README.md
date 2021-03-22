# Computational Methods for Data Science

[From three-ml-mnist-models](./three-ml-mnist-models/figs/10projections.png)

This repository contains a few open-ended data-driven projects I undertook in Winter 2021 under the supervision of [Dr. Jason Bramburger](https://faculty.washington.edu/jbrambur/) and [Dr. Nathan Kutz](https://faculty.washington.edu/kutz/) at UW. Topics include:

* noise cancellation
* automated music transcription
* motion tracking
* image recognition
* video object separation.


These projects are unified under the common theme of acoustic/image signal processing and dimensionality reduction with different flavors of PCA.

Each directory has a `report.pdf`, the abstracts of which are reproduced below in chronological order.

----

**submarine-hunting**: This paper contains the analysis of 3-D spatial-domain data where each point indicates the acoustic pressure measured at the corresponding coordinate in the Puget Sound. There exists a mysterious submarine emitting an unknown frequency through the waters. Using digital signal processing techniques, I identify this submarine's sonic signature and determine its path through the Puget Sound. These mechanisms can be easily re-applied in the future to locate any other submarine of the same class at any time.

**short-time-fast-time**: I use  the  Gabor  Transform  to  generate  automatic  music  transcription.   Samples  are excerpted  from  classic  rock,  and  include  guitar  tones  with  lots  of  harmonic  distortion.   The  Gabor Transform is a useful tool, and is easily applied to isolated parts.  The principal challenge for automated transcription of a dense piece of music then comes from isolating the part sonically.

**watching-oscillations**: An object undergoing simple harmonic motion is filmed by three cameras in subpar lab conditions. The method of Principal Component Analysis is used to isolate the simple harmonic motion from other factors, particularly noise and independent movement in extraneous dimensions. Principal Component Analysis is found to be an effective technique to identify patterns of variance in multi-dimensional data and isolate the most interesting feature.

**three-ml-mnist-models**: Principal Component Analysis is used to reduce dimension and mine features from the MNIST database of handwritten digits. I then compare how three different classifiers perform when given these labeled features, i.e. linear discriminant analysis (LDA), decision tree classification, and support vector machines.

**background-subtraction**: Sequential frames of a video can be regarded as a transformation of the previous. In the process of Dynamic Mode Decomposition, this transformation is assumed to be linear and decomposed. Out of the resulting components, the constant-order ones can be used to isolate the background of the video. The background is then subtracted from the original video to successfully isolate the foreground.

If you've made it this far, my two favorite projects are `short-time-fast-time` and `three-ml-mnist-models`. They have the prettiest results.