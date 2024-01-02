# -*- coding: utf-8 -*-
# imreg_examples.py

"""Imreg examples."""

import time

import imreg
import numpy
from matplotlib import pyplot

if 1:
    im0 = imreg.imread('data/t400')
    im1 = imreg.imread('data/Tr19s1.3')

if 0:
    im1 = imreg.imread('data/t350380ori')
    im0 = imreg.imread('data/t350380shf')

if 0:
    im0 = pyplot.imread('data/im1.png')[..., 0]
    im1 = pyplot.imread('data/im2.png')[..., 0]

if 0:
    im0 = pyplot.imread('data/im2.png')[..., 0]
    im1 = pyplot.imread('data/temp_1_1.png')[..., 0]
    im2 = numpy.zeros_like(im0)
    im2[: im1.shape[0], : im1.shape[1]] = im1
    im1 = im2

if 0:
    im0 = pyplot.imread('data/im2.png')[..., 0]
    im1 = pyplot.imread('data/temp_18_23.png')[..., 0]
    im2 = numpy.zeros_like(im0)
    im2[: im1.shape[0], : im1.shape[1]] = im1
    im1 = im2

if 0:
    im0 = pyplot.imread('data/1.png')[..., 0]
    im1 = pyplot.imread('data/2.png')[..., 0]
    im2 = numpy.zeros_like(im1)
    im2[: im0.shape[0], : im0.shape[1]] = im0
    im0 = im2

if 0:
    im0 = pyplot.imread('data/1_.png')[..., 0]
    im1 = pyplot.imread('data/2_.png')[..., 0]

t = time.perf_counter()
if 0:
    t0, t1 = imreg.translation(im0, im1)
    im2 = imreg.ndii.shift(im1, [t0, t1])
    print(t0, t1)
else:
    im2, scale, angle, (t0, t1) = imreg.similarity(im0, im1)
    print(t0, t1, scale, angle)
print(time.perf_counter() - t)

imreg.imshow(im0, im1, im2)
