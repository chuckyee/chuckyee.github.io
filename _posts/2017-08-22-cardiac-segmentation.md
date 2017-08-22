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
shaped object with thin walls that sometimes blends in with the surrounding
tissue. Here are the manually drawn contours for the inner and outer walls
(endocardium and epicardium) of the right ventricle:

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

The organizers of the segmentation challenge chose to use the [dice
coefficient](https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient)
to quantify performance. The deep learning model will output a mask $$X$$
delineating what it thinks is the RV, and the dice coefficient compares it to
the mask $$Y$$ produced by a physician:

![dice-venn](/images/dice-venn.png)

$$ \mathrm{dice}(X, Y) = \frac{2 X \cap Y}{X + Y} $$

The metric is (twice) the ratio of the intersection over the sum of areas. It
is 0 for disjoint areas, and 1 for perfect agreement.

## The dataset

The dataset (accessible
[here](http://www.litislab.fr/?sub_project=how-to-download-the-data)) contains
243 images like those shown above along with an RV segmentation mask produced
by a physician from the MRIs of 16 patients. There are 3697 additional
unlabeled images, which may be useful for unsupervised or semi-supervised
techniques, but are set aside in this work. The images have dimensions 216 x
256 pixels, and we rotate them all into landscape format.

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

During training, we split out 20% of the images as a validation set. The
organizers of the RV segmentation challenge have a separate test set consisting
of MRI images derived from an additional 32 patients, for which we submit
predicted contours for evaluation.

Let's look at model architectures.

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

## Dilated u-nets: global receptive fields

Segmenting organs requires some knowledge of the global context of how organs
are arranged relative to one another. Do any of the neurons in the u-net have a
receptive field that covers the entire image? If not, no part of the network
can "see" the entire image and integrate global context in producing the
segmentation mask.

![receptive-field-unet](/images/receptive-field-unet.png)

It turns out the units in the deepest part of our network had receptive fields
that only spanned 68$$\times$$68 pixels. The reasoning is that the network
would have no understanding that there is only one right ventricle in a human,
and it misclassifies the blob marked with an arrow above.

Rather than adding two more downsampling layers at the cost of a huge increase
in network parameters, we use dilated convolutions
[[5](https://arxiv.org/abs/1511.07122)] to increase the receptive fields of our
network.

![dilated-convs](/images/dilated-convs.png)

Dilated convolutions space out the pixels summed over in the convolution by a
dilation factor. In the diagram above, the convolutions in the bottom layer are
regular 3$$\times$$3 convolutions. The next layer up, we have dilated the
convolutions by a factor of 2, so their effective receptive field in the
original image is 7$$\times$$7. The top layer convolutions are dilated by 4,
producing 15$$\times$$15 receptive fields. Dilated convolutions produce
*exponentially* expanding receptive fields with depth, in contrast to linear
expansion for stacked conventional convolutions.

![receptive-field-dilated-unet](/images/receptive-field-dilated-unet.png)

Schematically, we replace the convolutional layers producing the feature maps
marked in yellow with dilated convolutions in our u-net. The innermost neurons
now have receptive fields spanning the entire input image.

## Dilated densenets: multiple scales at once

The final architecture we considered is a "dilated densenet". The idea is
simple: combine dilated convolutions with a dense convolutional network
[[6](https://arxiv.org/abs/1608.06993)]. In densenets, the output of the first
convolutional layer is fed as input into all subsequent layers, and similarly
with the second, third, and so forth. The authors show that densenets have
several advantages:

> they alleviate the vanishing-gradient problem, strengthen feature
> propagation, encourage feature reuse, and substantially reduce the number of
> parameters.

At publication, densenets had surpassed the state of the art on the CIFAR and
ImageNet classification benchmarks.

However, densenets are extremely memory intensive since the number of feature
maps grow linearly with the depth. The authors used "transition layers" to cut
down on the number of feature maps in order to train their 40, 100 and 250
layer densenets.

What if we used dilated convolutions instead? Then a network could "see" an
entire 256$$\times$$256 image with just 8 layers. A schematic of a 5-layer
dilated densenet is shown below:

![dilated-densenet](/images/dilated-densenet.png)

In contrast to the u-net architecture, a dilated densenet does not smash an
image down into small feature maps to generate high-level representations.
Rather, it retains the same feature map size throughout (we use zero padding),
and relies on dilated convolutions. In the final convolutional layer, the
neurons have access to global context as well as features produced at every
prior scale in the network. In our work, we use an 8-layer dilated densenet and
vary the growth rate from 12 to 32.

Let's see how these architectures perform when segmenting right ventricles.

## Training: what loss and hyperparameters to use?

We used the standard pixel-wise cross entropy loss, but also experimented with
using a "soft" dice loss. To talk about loss functions, we need to introduce
some notation. We'll denote by $$\hat{y}_{nk}$$ the output of the model, where
$$n$$ runs over all pixels and $$k$$ runs over the classes (in our case,
background vs. right ventricle). The ground truth masks are one-hot encoded and
denoted by $$y_{nk}$$.

For the pixel-wise cross entropy, we include weights $$w_k$$ to allow for
reweighting the strongly imbalanced classes:

$$ L(y, \hat{y}) = -\sum_{nk} w_k y_{nk} \log \hat{y}_{nk} $$

Simple averaging corresponds to $$w_\mathrm{background} = w_\mathrm{RV} =
0.5$$. We use $$w_\mathrm{background} = 0.1$$ and $$w_\mathrm{RV} = 0.9$$ to
bias the model to pay more attention to the RV pixels.

We also use a "soft" dice loss summed over the classes, again with weights
$$w_k$$ to allow for rebalancing:

$$ L(y, \hat{y}) = 1 - \sum_k w_k \frac{\sum_n y_{nk} \hat{y}_{nk}}{\sum_n
y_{nk} + \sum_n \hat{y}_{nk}} $$

We take one minus the dice coefficient so the loss tends towards zero. This
dice coefficient is "soft" in the sense that the output probabilities at each
pixel $$\hat{y}_{nk}$$ aren't rounded to be $$[\hat{y}_{nk}] \in \{0, 1\}$$.
Rounding isn't a differentiable operation and cannot be used as a loss
function. We use the usual "hard" dice coefficient to report the classification
performance.

In our training runs, we varied the following hyperparameters:
* Batch normalization
* Dropout
* Learning rate
* Growth rate (for the dilated densenets)

## Results

We estimated human-level performance to be about 0.95 by looking at the quality
of the provided manual segmentation contours. The leading published model is a
fully convolutional network (FCN) by Tran
[[7](https://arxiv.org/abs/1604.00494)] with 0.84 accuracy on the test set. The
u-net base model performs well, the dilated u-net does better, and the dilated
densenet performs the best, reaching a dice coefficient of 0.87 on the
validation dataset with only 190K parameters.

The dice coefficients, along with their uncertainties in parentheses, are
summarized in the following table. For the endocardium:

| **Method**       | **Train**   | **Val**     | **Test**    | **Params** |
| :--------------- | :---------- | :---------- | :-------    | :--------- |
| Human            | 0.95 (0.05) | --          | --          | --         |
| FCN (Tran 2017)  | --          | --          | 0.84 (0.21) | ~11M       |
| U-net            | 0.91 (0.06) | 0.82 (0.23) | TBD         | 1.9M       |
| Dilated u-net    | 0.92 (0.08) | 0.85 (0.19) | TBD         | 3.7M       |
| Dilated densenet | 0.91 (0.10) | 0.87 (0.15) | TBD         | 0.19M      |

For the epicardium:

| **Method**       | **Train**   | **Val**     | **Test**    | **Params** |
| :--------------- | :---------- | :---------- | :-------    | :--------- |
| Human            | 0.95 (0.05) | --          | --          | --         |
| FCN (Tran 2017)  | --          | --          | 0.86 (0.20) | ~11M       |
| U-net            | 0.93 (0.07) | 0.86 (0.17) | TBD         | 1.9M       |
| Dilated u-net    | 0.94 (0.05) | 0.90 (0.14) | TBD         | 3.7M       |
| Dilated densenet | 0.94 (0.04) | 0.89 (0.15) | TBD         | 0.19M      |

Typical learning curves are shown below. For all architectures, we used the
adam optimizer with the default initial learning rate of $$10^{-3}$$ and
trained for 500 epochs. Each image in the dataset was individually normalized
by subtracting its mean and dividing by its standard deviation. We used the
unweighted pixel-wise cross entropy as the loss function. Dropout was turned
off, and pre-activation batch normalization was used only for the dilated
densenet. The dilated densenet had a growth rate of 24. We used a batch size of
32, except for the dilated densenet, which required a batch size of 3 on our
16GB GPU due to memory constraints. In all cases, the validation loss plateaus
and does not exhibit an upturn characteristic of overfitting.

![learning-curves](/images/learning-curves-aaug.png)

On a per epoch basis, we were astounded to find that the dilated densenet
learns extremely quickly relative to the u-net and dilated u-net, likely
because the dense connections facilitate gradient propagation. It is also
extremely parameter efficient, reaching state of the art performance with
60$$\times$$ less parameters than the FCN.

In our results, as well as in the published literature, the accuracies exhibit
large uncertainties. Boxplots show that for some images, the networks struggle
to segment the RV to any extent:

![boxplots](/images/boxplots-eaug.png)

Examining the outliers, we find they mostly arise from apical slices (near the
bottom tip) of the heart where the RV is difficult to identify.


[Create box plot, then look at the points which have zero dice and create
strategy to fix. Propose pre-training model on apical slices only.]

[Quantify variation with dropout and batch norm]

[Quantify variation with growth rate]

[Quantify finishing with dice coefficient]

## Summary and future directions

I was happy to be able to create deep learning models that could perform at the
state of the art in cardiac segmentation. Here are my opinions on future
directions:

* Explore models with greater expressive power to reduce bias, creating
  additional "breathing room" to reduce variance and improve performance.
* Perhaps pretrain model on subset of data consisting of apical slices, or more
  strongly weight this data, to avoid failure on outliers.
* Explore quasi-3D models where the entire stack of registered cardiac slices
  are simultaneously fed into the model.
* Explore multistep (localization, registration, segmentation) pipelines.
* In developing a production system, we will need to optimize for the final
  figure of merit, which is the systolic and diastolic right ventricle volumes,
  rather than the dice coefficients of the individual slices.

Regarding model architectures, I was astounded by the parameter efficiency and
accuracy of dilated densenets, and I am particularly excited to see how well
they perform on standard segmentation tasks.

* Explore dilated densenet architectures: additional convolutional layers at
  each scale, or a mirror stack of convolution layers with dilation factors may
  help performance.
* Memory efficient dilated densenets: densely connected networks are notorious
  for requiring immense ammounts of memory. The raw TensorFlow is particularly
  egregious, limiting us to 8 layers with a batch size of 3 images on a 16GB
  GPU. Switching to the recently-proposed memory efficient implementation
  [[8](https://arxiv.org/abs/1707.06990)], would allow for deeper
  architectures.

## About this project

The Insight AI Fellows Program is an intense, 7-week experience where fellows
already possessing deep technical expertise are provided an environment to
bridge the small skills gap needed to enter the AI field. A major component of
bridging that gap involves completing a project in deep learning in order to
gain hands-on experience in AI.

As a relative novice to deep learning, I approached project selection like a
new graduate student: choose a clearly motivated and well scoped problem that
could be tackled in 4 weeks. I deliberately selected a project with small a
dataset to allow me to quickly iterate and gain experience. I'm happy that this
project met those goals and provided the intellectual playground to come up
with dilated densenets.
