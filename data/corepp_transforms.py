"""CoRe++ image transforms (Pad only — ported from corepp/dataloaders/transforms.py)."""

import numbers

import numpy as np
from PIL import Image
from torchvision.transforms import v2


def get_padding(image, size):
    w, h = image.size
    max_wh = size
    h_padding = (max_wh - w) / 2
    v_padding = (max_wh - h) / 2
    l_pad = h_padding if h_padding % 1 == 0 else h_padding + 0.5
    t_pad = v_padding if v_padding % 1 == 0 else v_padding + 0.5
    r_pad = h_padding if h_padding % 1 == 0 else h_padding - 0.5
    b_pad = v_padding if v_padding % 1 == 0 else v_padding - 0.5
    return (int(l_pad), int(t_pad), int(r_pad), int(b_pad))


class Pad:
    def __init__(self, size=512, fill=0, padding_mode="constant"):
        assert isinstance(fill, (numbers.Number, str, tuple))
        assert padding_mode in ["constant", "edge", "reflect", "symmetric"]
        self.fill = fill
        self.padding_mode = padding_mode
        self.size = size

    def __call__(self, img):
        img = Image.fromarray(img)
        return v2.functional.pad(img, get_padding(img, self.size), self.fill, self.padding_mode)
