---
layout: post
title: "Cardiac MRI Segmentation"
published: true
mathjax: true
---

A human heart is an astounding machine that is designed to continually function
for up to a century without failure. One of the key ways to measure how well
your heart is functioning is to compute its [ejection
fraction](https://en.wikipedia.org/wiki/Ejection_fraction): after your heart
relaxes at its diastole to fully fill with blood, what percentage does it pump
out upon contracting to its systole? Getting at this metric relies on tracing
the outlines of the ventricles from cardiac images.

During my time at the [Insight AI Program](http://insightdata.ai) in NYC, I
decided to tackle the [right ventricle segmentation
challenge](http://ai-on.org/projects/cardiac-mri-segmentation.html) from the
calls for research hosted by the [AI Open Network](http://ai-on.org/).

## Problem description

From the call for research: 

> Develop a system capable of automatic segmentation of the right ventricle in
> images from cardiac magnetic resonance imaging (MRI) datasets. Until now,
> this has been mostly handled by classical image processing methods. Modern
> deep learning techniques have the potential to provide a more reliable,
> fully-automated solution.

All three winners of the [left ventricle segmentation
challenge](https://www.kaggle.com/c/second-annual-data-science-bowl) sponsored
by Kaggle in 2016 were deep learning solutions. However, segmenting the right
ventricle (RV) is more challenging, because of:

> [the] presence of trabeculations in the cavity with signal intensities similar to
> that of the myocardium; the complex crescent shape of the RV, which varies
> from the base to the apex; difficulty in segmenting the apical image slices;
> considerable variability in shape and intensity of the chamber among
> subjects, notably in pathological cases, etc.

Medical jargon aside, it's just more difficult to identify the RV. The left
ventricle is a thick-walled circle while the right ventricle is an irregularly
shaped object:

![data-easy](/images/data-easy.png)

That was an easy example. This one is more difficult:

![data-medium](/images/data-medium.png)

And this one is downright challenging to the untrained eye:

![data-hard](/images/data-hard.png)

Human physicians in fact take twice as long to determine the RV volume and
produce results that have 2-3 times the variability as compared to the left
ventricle [[1](https://dx.doi.org/10.1007/s00330-011-2152-0)]. The goal of this
work is to build a deep learning model that automates right ventricle
segmentation with high accuracy.

## The dataset

The dataset (accessible
[here](http://www.litislab.fr/?sub_project=how-to-download-the-data)) contains
243 images like those shown above along with an RV segmentation mask produced
by a physician. There are 3697 additional unlabeled images, which may be useful
for unsupervised or semi-supervised techniques, but are set aside in this work.
The images have dimensions 216 x 256 pixels, and we rotate them all into
landscape format.

Given the small dataset size, one would suspect generalization to unseen images
would be hopeless! This unfortunately is the typical situation in medical
settings where labeled data is expensive to come by. The standard procedure is
to apply affine transformations to the data: random rotations, translations,
zooms and shears. We also implemented elastic deformations, which locally
stretch and compress the image
[[2](https://www.microsoft.com/en-us/research/publication/best-practices-for-convolutional-neural-networks-applied-to-visual-document-analysis/)].

![data-augmentation](/images/data-augmentation.png)

In our training framework, we apply these transformations on the fly so the
network sees new random transformations during each epoch.

As is also common, there is a large class imbalance since most of the pixels
are background. Normalizing the pixel intensities to lie between 0 and 1, we
see that across the entire dataset, only 5.1% of the pixels are part of the RV
cavity.

![pixel-stats](/images/pixel-statistics.png)

In constructing our loss functions, we experimented with reweighting schemes to
balance the class distributions, but ultimately found that a simple average
performed best.

## U-net: the baseline model

In the medical segmentation domain, most architectures have been based on
convolutional networks which construct feature maps at varying scales and then
combine them to form the pixel-wise segmentation mask (see
[[3](https://arxiv.org/abs/1701.03056)] for a recent review). We selected as
our baseline model the u-net, proposed by Ronneberger, Fischer and Brox
[[4](https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28)], since it
has been quite successful in biomedical segmentation tasks. The authors were
able to successful train their network with only 30 images combined with
aggressive image augmentation. (I suspect their task was less complex, as it
consisted of delineating cells in microscopy images.)

The u-net architecture consists of a contracting path, which collapses an image
down into a set of high level features, followed by an expanding path which
applies the feature information to construct a pixel-wise segmentation mask.
The unique aspect of the u-net are its "copy and concatenate" connections which
pass information from early feature maps to the later portions of the network
tasked with constructing the segmentation mask. The authors propose that these
connections allow the network to incorporate high level features and pixel-wise
detail simultaneously.

The architecture we used is shown here:

![unet-architecture](/images/unet-architecture.png)

We adapted the u-net to our purposes by reducing the number of downsampling
layers from four to three, since our images were roughly half the size as those
considered by the original authors. We also zero pad our convolutions (as
opposed to unpadded) to keep the images the same size. The model was
implemented in Keras.

To talk about loss functions, we need to introduce some notation. The output of
the model is a (width $\times$ height $\times$ nclasses) mask which we denote
by $\hat{y}_{nk}$, where $n$ runs over all pixels and $k$ runs over the classes
(in our case, background and right ventricle). The ground truth is $y_{nk}$,
which is one-hot encoded across the classes. We considered the pixel-wise cross
entropy

$$ L(y, \hat{y}) = -\sum_{nk} w_k y_{nk} \log \hat{y}_{nk} $$

and the soft dice loss

$$ L(y, \hat{y}) = 1 - \sum_k w_k \frac{\sum_n y_{nk} \hat{y}_{nk}}{\sum_n
y_{nk} + \sum_n \hat{y}_{nk}} $$

They have been modified from 

What about different numbers of feature maps?

## Results

We use the Dice coefficient to compare the pixel-wise agreement between the
predicted segmentation and the corresponding ground truth. If $X$ and $Y$ are
the predicted and ground truth areas,

![dice-venn](/images/dice-venn.png)

the [Dice
coefficient](https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient)
is given by twice the intersection over the sum of areas,

$$ \mathrm{dice}(X, Y) = \frac{2 X \cap Y}{X + Y} $$

It is 0 for disjoint areas, and 1 for perfect agreement.

| Tables        | Are           | Cool  |
| ------------- |:-------------:| -----:|
| col 3 is      | right-aligned | $1600 |
| col 2 is      | centered      |   $12 |
| zebra stripes | are neat      |    $1 |


## Dilated u-nets: global receptive fields

![receptive-field-unet](/images/receptive-field-unet.png)
![receptive-field-dilated-unet](/images/receptive-field-dilated-unet.png)

![dilated-convs](/images/dilated-convs.png)

## Dilated DenseNets

![dilated-densenet](/images/dilated-densenet.png)

## Other stuff

![predicted-masks](/images/predicted-masks.png)

## References

* Overview of state-of-the-art of CNN applications in medical segmentation [link](https://arxiv.org/abs/1701.03056).
* Right Ventricle Segmentation From Cardiac MRI: A Collation Study
* A survey of shaped-based registration and segmentation techniques for cardiac images
* Right ventricular segmentation in cardiac MRI with moving mesh correspondences
* Segmentation of RV in 4D cardiac mr volumes using region-merging graph cuts

Note: this entire library is written with the Tensorflow backend in mind --
(batch, height, width, channels) ordered is assumed and is not portable to
Theano.

## About this project

The [Insight Artificial Intelligence Fellows Program](http://insightdata.ai) is
an intense, 7-week experience where fellows already possessing deep technical
expertise are provided an environment to bridge the small skills gap needed to
enter the AI field. A major component of bridging that gap involves completing
a project in deep learning in order to gain hands-on experience in AI.

As a fellow in the 2017 NYC program with little prior experience in deep
learning, I approached project selection like a new graduate student: choose a
clearly motivated and well scoped problem that could be tackled in 4 weeks. I
deliberately selected a project with small a dataset to allow me to quickly
iterate and gain experience in deep learning.