import matplotlib.image as mpimg
import os
import shutil

import numpy as np


def generate_great_image(params):

    h=params['generate_video_params']['height']
    w=params['generate_video_params']['width']
    patch_size=params['generate_video_params']['patch_size']
    stride=params['generate_video_params']['strdie']

    idx = 0
    for i in range(params['']):
        gen_image = np.zeros((h, w))
        count = np.zeros((h, w))
        r = 0
        while r + patch_size <= h:
            c = 0
            while c + patch_size <= w:
                gen_image[r:r + patch_size, c:c + patch_size] += gen_data[idx, 0]
                count[r:r + patch_size, c:c + patch_size] += 1
                idx += 1
                c = c + stride
            c = w - patch_size
            gen_image[r:r + patch_size, c:c + patch_size] += gen_data[idx, 0]
            count[r:r + patch_size, c:c + patch_size] += 1
            idx += 1
            r = r + stride

        r = h - patch_size
        c = 0
        while c + patch_size <= w:
            gen_image[r:r + patch_size, c:c + patch_size] += gen_data[idx, 0]
            count[r:r + patch_size, c:c + patch_size] += 1
            idx += 1
            c = c + stride
        c = w - patch_size
        gen_image[r:r + patch_size, c:c + patch_size] += gen_data[idx, 0]
        count[r:r + patch_size, c:c + patch_size] += 1
        gen_image = gen_image / count
        idx += 1
        mpimg.imsave('gen'+str(i)+'.png', gen_image, cmap='gray', vmin=0, vmax=1)
def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        print('Dir created')
        return True
    else:
        print('Dir already exists')
        return False
def rmdir(path):
    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)
    if isExists:
        shutil.rmtree(path)
        print('Dir removed')
        return True
    else:
        print('Dir not exists')
        return False


