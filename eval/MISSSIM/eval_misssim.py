import pathlib
import numpy as np
import torch
from imageio import imread
from pytorch_msssim import ms_ssim

def read_img_tensor(path):
    path = pathlib.Path(path)
    files = list(path.glob('*.jpg')) + list(path.glob('*.png'))
    images = np.array([imread(str(f)).astype(np.float32) for f in files])
    # Reshape to (n_images, 3, height, width)
    images = images.transpose((0, 3, 1, 2))
    images /= 255
    return torch.from_numpy(images).type(torch.FloatTensor)

real_path = '../real/'
fake_path = '../fake/'
reals = read_img_tensor(real_path)
fakes = read_img_tensor(fake_path)

ms_ssim_val = ms_ssim(reals, fakes, data_range=255, size_average=True)
print(ms_ssim_val)