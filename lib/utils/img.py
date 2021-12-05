"""
Zoom-In Code taken from:  https://github.com/LZGMatrix/CDPN_ICCV2019_ZhigangLi
"""
import torch
import numpy as np
import cv2
def zoom_in(im, c, s, res, channel=3, interpolate=cv2.INTER_LINEAR):
    """
    zoom in on the object with center c and size s, and resize to resolution res.
    :param im: nd.array, single-channel or 3-channel image
    :param c: (w, h), object center
    :param s: scalar, object size
    :param res: target resolution
    :param channel:
    :param interpolate:
    :return: zoomed object patch 
    """
    c_w, c_h = c
    c_w, c_h, s, res = int(c_w), int(c_h), int(s), int(res)
    if channel==1:
        im = im[..., None]
    h, w = im.shape[:2]
    u = int(c_h-0.5*s+0.5)
    l = int(c_w-0.5*s+0.5)
    b = u+s
    r = l+s
    if (u>=h) or (l>=w) or (b<=0) or (r<=0):
        return np.zeros((res, res, channel)).squeeze()
    if u < 0:
        local_u = -u
        u = 0 
    else:
        local_u = 0
    if l < 0:
        local_l = -l
        l = 0
    else:
        local_l = 0
    if b > h:
        local_b = s-(b-h)
        b = h
    else:
        local_b = s
    if r > w:
        local_r = s-(r-w)
    else:
        local_r = s
    im_crop = np.zeros((s, s, channel))
    im_crop[local_u:local_b, local_l:local_r, :] = im[u:b, l:r, :]
    im_crop = im_crop.squeeze()
    im_resize = cv2.resize(im_crop, (res, res), interpolation=interpolate)
    c_h = 0.5*(u+b)
    c_w = 0.5*(l+r)
    s = s
    return im_resize, c_h, c_w, s

