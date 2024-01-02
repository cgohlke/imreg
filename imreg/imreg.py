# imreg.py

# Copyright (c) 2011-2024, Christoph Gohlke
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""FFT based image registration.

Imreg is a Python library that implements an FFT-based technique for
translation, rotation and scale-invariant image registration [1].

:Author: `Christoph Gohlke <https://www.cgohlke.com>`_
:License: BSD 3-Clause
:Version: 2024.1.2

Quickstart
----------

Install the imreg package and all dependencies from the
`Python Package Index <https://pypi.org/project/imreg/>`_::

    python -m pip install -U imreg

See `Examples`_ for using the programming interface.

Source code and support are available on
`GitHub <https://github.com/cgohlke/imreg>`_.

Requirements
------------

This revision was tested with the following requirements and dependencies
(other versions may work):

- `CPython <https://www.python.org>`_ 3.9.13, 3.10.11, 3.11.7, 3.12.1
- `NumPy <https://pypi.org/project/numpy/>`_ 1.26.2
- `Scipy <https://pypi.org/project/scipy>`_ 1.11.4
- `Matplotlib 3.8.2 <https://pypi.org/project/matplotlib>`_
  (optional for plotting)

Revisions
---------

2024.1.2

- Add type hints.
- Drop support for Python 3.8 and numpy < 1.23 (NEP29).

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

"""

from __future__ import annotations

__version__ = '2024.1.2'

__all__ = [
    'translation',
    'similarity',
    'similarity_matrix',
    'logpolar',
    'highpass',
    'imread',
    'imshow',
]

import math
import os
from typing import TYPE_CHECKING

import numpy
from numpy.fft import fft2, fftshift, ifft2
from scipy import ndimage

if TYPE_CHECKING:
    from typing import Any

    from numpy.typing import ArrayLike, NDArray


def translation(
    im0: ArrayLike,
    im1: ArrayLike,
    /,
) -> tuple[int, int]:
    """Return translation vector to register images."""
    im0 = numpy.asanyarray(im0)
    shape = im0.shape
    assert len(shape) == 2
    f0 = fft2(im0)
    f1 = fft2(im1)
    ir = abs(ifft2((f0 * f1.conjugate()) / (abs(f0) * abs(f1))))
    t0, t1 = numpy.unravel_index(numpy.argmax(ir), shape)
    if t0 > shape[0] // 2:
        t0 -= shape[0]
    if t1 > shape[1] // 2:
        t1 -= shape[1]
    return int(t0), int(t1)


def similarity(
    im0: ArrayLike,
    im1: ArrayLike,
    /,
) -> tuple[NDArray[Any], float, float, tuple[int, int]]:
    """Return similarity transformed image `im1` and transformation parameters.

    Transformation parameters are: isotropic scale factor, rotation angle (in
    degrees), and translation vector.

    A similarity transformation is an affine transformation with isotropic
    scale and without shear.

    Limitations:

    - Image shapes must be equal and square.
    - All image areas must have same scale, rotation, and shift.
    - Scale change must be less than 1.8.
    - No subpixel precision.

    """
    im0 = numpy.asanyarray(im0)
    im1 = numpy.asanyarray(im1)

    if im0.shape != im1.shape:
        raise ValueError('images must have same shapes')
    if len(im0.shape) != 2:
        raise ValueError('images must be two-dimensional')

    f0 = fftshift(abs(fft2(im0)))
    f1 = fftshift(abs(fft2(im1)))

    h = highpass(f0.shape)
    f0 *= h
    f1 *= h
    del h

    f0, log_base = logpolar(f0)
    f1, log_base = logpolar(f1)

    f0 = fft2(f0)  # type: ignore
    f1 = fft2(f1)  # type: ignore
    r0 = abs(f0) * abs(f1)
    ir = abs(ifft2((f0 * f1.conjugate()) / r0))
    i0, i1 = numpy.unravel_index(numpy.argmax(ir), ir.shape)
    angle: float = 180.0 * int(i0) / ir.shape[0]
    scale: float = log_base ** int(i1)

    if scale > 1.8:
        ir = abs(ifft2((f1 * f0.conjugate()) / r0))
        i0, i1 = numpy.unravel_index(numpy.argmax(ir), ir.shape)
        angle = -180.0 * int(i0) / ir.shape[0]
        scale = 1.0 / (log_base ** int(i1))
        if scale > 1.8:
            raise ValueError('images are not compatible. Scale change > 1.8')

    if angle < -90.0:
        angle += 180.0
    elif angle > 90.0:
        angle -= 180.0

    im2 = ndimage.zoom(im1, 1.0 / scale)
    im2 = ndimage.rotate(im2, angle)

    if im2.shape < im0.shape:
        t = numpy.zeros_like(im0)
        t[: im2.shape[0], : im2.shape[1]] = im2
        im2 = t
    elif im2.shape > im0.shape:
        im2 = im2[: im0.shape[0], : im0.shape[1]]

    f0 = fft2(im0)  # type: ignore
    f1 = fft2(im2)  # type: ignore
    ir = abs(ifft2((f0 * f1.conjugate()) / (abs(f0) * abs(f1))))
    t0, t1 = numpy.unravel_index(numpy.argmax(ir), ir.shape)

    if t0 > f0.shape[0] // 2:
        t0 -= f0.shape[0]
    if t1 > f0.shape[1] // 2:
        t1 -= f0.shape[1]

    im2 = ndimage.shift(im2, [t0, t1])

    # correct parameters for ndimage's internal processing
    if angle > 0.0:
        d = int(int(im1.shape[1] / scale) * math.sin(math.radians(angle)))
        t0, t1 = t1, d + t0
    elif angle < 0.0:
        d = int(int(im1.shape[0] / scale) * math.sin(math.radians(angle)))
        t0, t1 = d + t1, d + t0
    scale = (im1.shape[1] - 1) / (int(im1.shape[1] / scale) - 1)

    return im2, scale, angle, (int(-t0), int(-t1))


def similarity_matrix(
    scale: float,
    angle: float,
    vector: ArrayLike,
) -> NDArray[Any]:
    """Return homogeneous transformation matrix from similarity parameters.

    Transformation parameters are: isotropic scale factor, rotation angle
    (in degrees), and translation vector (of size 2).

    The order of transformations is: scale, rotate, translate.

    """
    S = numpy.diag([scale, scale, 1.0])
    R = numpy.identity(3)
    angle = math.radians(angle)
    R[0, 0] = math.cos(angle)
    R[1, 1] = math.cos(angle)
    R[0, 1] = -math.sin(angle)
    R[1, 0] = math.sin(angle)
    T = numpy.identity(3)
    T[:2, 2] = vector
    return numpy.dot(T, numpy.dot(R, S))


def logpolar(
    image: ArrayLike,
    /,
    *,
    angles: int | None = None,
    radii: int | None = None,
) -> tuple[NDArray[Any], float]:
    """Return log-polar transformed image and log base."""
    image = numpy.asanyarray(image)
    shape = image.shape
    center = shape[0] / 2, shape[1] / 2
    if angles is None:
        angles = shape[0]
    if radii is None:
        radii = shape[1]
    theta = numpy.empty((angles, radii), dtype='float64')
    theta.T[:] = numpy.linspace(0, numpy.pi, angles, endpoint=False) * -1.0
    # d = radii
    d = numpy.hypot(shape[0] - center[0], shape[1] - center[1])
    log_base = 10.0 ** (math.log10(d) / (radii))
    radius = numpy.empty_like(theta)
    radius[:] = (
        numpy.power(log_base, numpy.arange(radii, dtype='float64')) - 1.0
    )
    x = radius * numpy.sin(theta) + center[0]
    y = radius * numpy.cos(theta) + center[1]
    output = numpy.empty_like(x)
    ndimage.map_coordinates(image, [x, y], output=output)
    return output, log_base


def highpass(shape: tuple[int, ...]) -> NDArray[Any]:
    """Return highpass filter to be multiplied with Fourier transform."""
    x = numpy.outer(
        numpy.cos(numpy.linspace(-math.pi / 2.0, math.pi / 2.0, shape[0])),
        numpy.cos(numpy.linspace(-math.pi / 2.0, math.pi / 2.0, shape[1])),
    )
    return (1.0 - x) * (2.0 - x)


def imread(
    fname: str | os.PathLike,
    /,
    *,
    norm: bool = True,
) -> NDArray[Any]:
    """Return image data from img&hdr uint8 files."""
    fname = os.fspath(fname)
    with open(fname + '.hdr', encoding='utf-8') as fh:
        hdr = fh.readlines()
    img = numpy.fromfile(fname + '.img', numpy.uint8, -1)
    img.shape = int(hdr[4].split()[-1]), int(hdr[3].split()[-1])
    if norm:
        img = img.astype(numpy.float64)
        img /= 255.0  # type: ignore
    return img


def imshow(
    im0: ArrayLike,
    im1: ArrayLike,
    im2: ArrayLike,
    im3: ArrayLike | None = None,
    /,
    *,
    cmap: str | None = None,
    **kwargs,
) -> None:
    """Plot images using matplotlib."""
    from matplotlib import pyplot

    im0 = numpy.asanyarray(im0)
    im1 = numpy.asanyarray(im1)
    im2 = numpy.asanyarray(im2)
    if im3 is None:
        im3 = abs(im2 - im0)
    pyplot.subplot(221)
    pyplot.imshow(im0, cmap, **kwargs)
    pyplot.subplot(222)
    pyplot.imshow(im1, cmap, **kwargs)
    pyplot.subplot(223)
    pyplot.imshow(im3, cmap, **kwargs)
    pyplot.subplot(224)
    pyplot.imshow(im2, cmap, **kwargs)
    pyplot.show()


if __name__ == '__main__':
    import doctest

    try:
        os.chdir('data')
    except Exception:
        pass
    doctest.testmod()
