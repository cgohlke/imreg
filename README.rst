FFT based image registration
============================

Imreg is a Python library that implements an FFT-based technique for
translation, rotation and scale-invariant image registration [1].

:Author: `Christoph Gohlke <https://www.cgohlke.com>`_
:License: BSD 3-Clause
:Version: 2022.9.27

Requirements
------------

* `CPython >= 3.7 <https://www.python.org>`_
* `Numpy 1.15 <https://www.numpy.org>`_
* `Scipy 1.5 <https://www.scipy.org>`_
* `Matplotlib 3.3 <https://www.matplotlib.org>`_  (optional for plotting)

Revisions
---------

2022.9.27

- Fix scipy.ndimage DeprecationWarning.

Notes
-----

Imreg is no longer being actively developed.

This implementation is mainly for educational purposes.

An improved version is being developed at https://github.com/matejak/imreg_dft.

References
----------

1. An FFT-based technique for translation, rotation and scale-invariant
   image registration. BS Reddy, BN Chatterji.
   IEEE Transactions on Image Processing, 5, 1266-1271, 1996
2. An IDL/ENVI implementation of the FFT-based algorithm for automatic
   image registration. H Xiea, N Hicksa, GR Kellera, H Huangb, V Kreinovich.
   Computers & Geosciences, 29, 1045-1055, 2003.
3. Image Registration Using Adaptive Polar Transform. R Matungka, YF Zheng,
   RL Ewing. IEEE Transactions on Image Processing, 18(10), 2009.

Examples
--------

>>> im0 = imread('t400')
>>> im1 = imread('Tr19s1.3')
>>> im2, scale, angle, (t0, t1) = similarity(im0, im1)
>>> imshow(im0, im1, im2)

>>> im0 = imread('t350380ori')
>>> im1 = imread('t350380shf')
>>> t0, t1 = translation(im0, im1)
>>> t0, t1
(20, 50)
