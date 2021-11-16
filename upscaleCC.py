#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 18:45:31 2021

@author: Luraminaki
"""

#===================================================================================================
import os
import time
import glob
import json
import inspect

import cv2
import numpy as np
#===================================================================================================
__version__ = "0.1.0"

data_folder = "./data/"

with open("config.json", encoding='utf-8') as f:
    conf = json.load(f)

BD_name = conf["BD_name"]
ext = conf["ext"]
folder_save_dl = data_folder + conf["folder_save_dl"]
folder_models = data_folder + conf["folder_models"]
folder_save_upscale = data_folder + conf["folder_save_upscale"]
model = conf["model_nbr"]

os.makedirs(folder_save_dl + BD_name + "/", exist_ok=True)
os.makedirs(folder_save_upscale + BD_name + "/", exist_ok=True)


def area_posterise(input_image: np.ndarray, nbr_cluster=32, nbr_iterations=10) -> np.ndarray:
    """Quantization function.

    Args:
        input_image (np.ndarray): Input image (BGR).
        nbr_cluster (int, optional): Number of "color(s)" to keep. Defaults to 32.
        nbr_iterations (int, optional): Number of loop(s) to make the cluster(s). Defaults to 10.

    Returns:
        np.ndarray: Quantized image.
    """
    curr_func = inspect.currentframe().f_code.co_name

    if nbr_cluster > 1 and nbr_cluster < 255:
        area_to_posterise_line = input_image.reshape((-1, 3))
        area_to_posterise_line = np.float32(area_to_posterise_line)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        flags = cv2.KMEANS_RANDOM_CENTERS
        _, labels, centers = cv2.kmeans(area_to_posterise_line, nbr_cluster, None, criteria, nbr_iterations, flags)
        centers = np.uint8(centers)
        res = centers[labels.flatten()]
        posterised_area_to_posterise = res.reshape((np.shape(input_image)))

        return posterised_area_to_posterise

    else:
        print("{} -- Change the value of the following parameter 'nbr_cluster' (1 ~ 255)".format(curr_func))

        return input_image

def sharpen_image(image: np.ndarray, ksize_1=7, ksize_2=7, sigma_1=1, sigma_2=2) -> np.ndarray:
    """Function that sharpen an image.

    Args:
        image (np.ndarray): Input image (BGR).
        ksize_1 (int, optional): Kernel n°1 size (Odd number). Defaults to 7.
        ksize_2 (int, optional): Kernel n°2 size (Odd number). Defaults to 7.
        sigma_1 (int, optional): Blur value n°1. Defaults to 1.
        sigma_2 (int, optional): Blur value n°2. Defaults to 2.

    Returns:
        np.ndarray: Sharpned image.
    """
    curr_func = inspect.currentframe().f_code.co_name

    blur1 = cv2.GaussianBlur(image, (ksize_1, ksize_1), sigma_1)
    blur2 = cv2.GaussianBlur(image, (ksize_2, ksize_2), sigma_2)

    temp_1 = cv2.addWeighted(image, 1.5, blur1, -0.5, 0)
    temp_2 = cv2.addWeighted(image, 1.5, blur2, -0.5, 0)
    sharpenned_image = cv2.addWeighted(temp_1, 0.5, temp_2, 0.5, 0)

    return sharpenned_image


def main() -> None:
    curr_func = inspect.currentframe().f_code.co_name

    # Get small pictures
    originals = glob.glob(folder_save_dl + BD_name + '/*' + ext)
    originals.sort()

    # Get completed pictures
    completed = glob.glob(folder_save_upscale + BD_name + '/*' + ext)

    # Get trained models
    models = glob.glob(folder_models + "*.pb")
    models.sort()

    if len(models) >= model:

        # Select model
        model_path = models[model]
        model_name = os.path.basename(model_path).split(".", maxsplit=1)[0].split("_", maxsplit=1)[0].lower()
        model_scale = int(os.path.basename(model_path).split(".", maxsplit=1)[0].split("_", maxsplit=1)[1].replace("x", ""))
        print("{} -- Using model {} with {} scaling...".format(curr_func, model_name, model_scale))

        # Prepare model
        sup_res = cv2.dnn_superres.DnnSuperResImpl_create()
        sup_res.readModel(model_path)
        sup_res.setModel(model_name, model_scale)

        # Compute / Upscale / Clean each small image
        for cptr, img_path in enumerate(originals[len(completed):]):
            tic = time.time()

            img_name = os.path.basename(img_path)
            print("{} -- Loading image {}".format(curr_func, img_name))
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)

            print("{} -- Upscaling image {}".format(curr_func, img_name))
            result = sup_res.upsample(img)

            print("{} -- Cleaning image {}".format(curr_func, img_name))
            img = area_posterise(img)
            img = sharpen_image(img)

            print("{} -- Saving image {}".format(curr_func, img_name))
            cv2.imwrite((folder_save_upscale + BD_name + "/" + img_name), result)

            tac = time.time()
            toc = int((tac - tic)*100)
            toc = toc / 100

            # Estimate remaining time to complete the task
            total = toc * (len(originals[len(completed):]) - cptr)

            day = int(total/(60*60*24))
            hour = int(total/(60*60))%24
            min = int(total/(60))%60
            sec = int(total)%60

            print("{} -- Image {} upscaled in {} second(s). \n\t Estimated remaining time: {}d - {}h:{}m:{}s\n".format(curr_func, img_name, toc, day, hour, min, sec))

    else:
        print("{} -- Model n°{} not found in the `data/models/` folder...".format(curr_func, model))


if __name__ == '__main__':
    main()
