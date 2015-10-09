#!/usr/bin/env python
import h5py, os
import numpy as np
from functools import wraps

from plot import plot_gabors


def generate_gabor(pixels, x, y, theta, stdx, stdy, lamb, phase):
    """
    Generate a gabor filter based on given parameters.

    Parameters
    ----------
    pixels : tuple of ints
        Height and width of patch.
    x : float
        x Location of center of gabor in pixels.
    y : float
        y Location of center of gabor in pixels.
    theta : float
        Rotation of gabor in plane in degrees.
    stdx : float
        Width of gaussian window along rot(x) in pixels.
    stdy : float
        Width of gaussian window along rot(y) in pixels.
    lamb : float
        Wavelength of sine funtion in pixels along rot(x).
    phase : float
        Phase of sine function in degrees.

    Returns
    -------
    gabor : ndarray
        2d array of pixel values.
    """
    x_coords = np.arange(0, pixels[0])
    y_coords = np.arange(0, pixels[1])
    xx, yy = np.meshgrid(x_coords, y_coords)
    unit2rad = 2. * np.pi 
    deg2rad = 2. * np.pi / 360.
    xp = (xx - x) * np.cos(deg2rad * theta) - (yy - y) * np.sin(deg2rad * theta)
    yp = (xx - x) * np.sin(deg2rad * theta) - (yy - y) * np.cos(deg2rad * theta)
    gabor = (np.exp(-xp**2 / (2. * stdx**2) - yp**2 / (2. * stdy**2)) *
             np.sin(unit2rad * (xp / lamb) - deg2rad * phase))
    norm = (gabor**2).sum()
    return gabor/norm

def _generate_fixed_loc_set(pixels, x, y, theta, stdx, stdy, lamb, phase):
    """
    Generate gabor filters at a fixed x, y location
    for the parameters ranges given.

    Parameters
    ----------
    pixels : tuple of ints
        Height and width of patch.
    x : float
        x Location of center of gabor in pixels.
    y : float
        y Location of center of gabor in pixels.
    theta : array
        Rotations of gabor in plane in degrees.
    stdx : array
        Widths of gaussian window along rot(x) in pixels.
    stdy : array
        Widths of gaussian window along rot(y) in pixels.
    lambda : array
        Wavelengths of sine funtion in pixels along rot(x).
    phase : array
        Phases of sine function in degrees.

    Returns
    -------
    gabors : ndarray
        nd array of pixel values.
    """
    lengths = (len(theta), len(stdx), len(stdy), len(lamb), len(phase))
    gabors = np.zeros(lengths + pixels)
    for ii, th in enumerate(theta):
        for jj, sx in enumerate(stdx):
            for kk, sy in enumerate(stdy):
                for nn, la in enumerate(lamb):
                    for mm, ph in enumerate(phase):
                        gabors[ii, jj, kk, nn, mm] = generate_gabor(pixels,
                                                                    x, y, 
                                                                    th, sx, sy,
                                                                    la, ph)
    return gabors

@wraps(_generate_fixed_loc_set)
def generate_fixed_loc_set(pixels, x, y, *args):
    # Convert ranges to values.
    largs = tuple(np.linspace(*arg, endpoint=False) for arg in args)
    return _generate_fixed_loc_set(pixels, x, y, *largs)

if __name__ == '__main__':
    gabors = generate_fixed_loc_set((16, 16), 5, 5, (0, 360, 8), (1, 5, 5), (1, 5, 5), (2, 10, 2), (0, 360, 4))
    gabors = gabors.reshape(-1, 256)
    plot_gabors(gabors)
