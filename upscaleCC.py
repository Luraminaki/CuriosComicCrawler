#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 18:45:31 2021

@author: Luraminaki
"""

#===================================================================================================
import time
import json
import inspect
import pathlib

import cv2
import numpy as np
#===================================================================================================
__version__ = "0.1.1"

CWD = pathlib.Path.cwd()

DATA_FOLDER = "data/"
CONFIG_FILE = "config.json"

with open(CWD/CONFIG_FILE, encoding='utf-8') as f:
    conf = json.load(f)

BD_NAME: str = str(conf["BD_name"])
EXT: str = str(conf["ext"])

MODEL: int = abs(int(conf["model_nbr"]))
FOLDER_MODELS: pathlib.Path = CWD/DATA_FOLDER/str(conf["folder_models"])

FOLDER_SAVE_DL: pathlib.Path = CWD/DATA_FOLDER/str(conf["folder_save_dl"])
FOLDER_SAVE_UPSCALE: pathlib.Path = CWD/DATA_FOLDER/str(conf["folder_save_upscale"])

FOLDER_SAVE_DL_BD_NAME: pathlib.Path = FOLDER_SAVE_DL/BD_NAME
FOLDER_SAVE_UPSCALE_BD_NAME: pathlib.Path = FOLDER_SAVE_UPSCALE/BD_NAME

FOLDER_SAVE_DL_BD_NAME.mkdir(exist_ok=True)
FOLDER_SAVE_UPSCALE_BD_NAME.mkdir(exist_ok=True)


def area_posterise(input_image: np.ndarray, nbr_cluster: int=32, nbr_iterations: int=10) -> np.ndarray:
    """Quantization function.

    Args:
        input_image (np.ndarray): Input image (BGR).
        nbr_cluster (int, optional): Number of "color(s)" to keep. Defaults to 32.
        nbr_iterations (int, optional): Number of loop(s) to make the cluster(s). Defaults to 10.

    Returns:
        np.ndarray: Quantized image.
    """
    curr_func = inspect.currentframe().f_code.co_name

    if 1 < nbr_cluster < 255:
        area_to_posterise_line = input_image.reshape((-1, 3))
        area_to_posterise_line = np.float32(area_to_posterise_line)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        flags = cv2.KMEANS_RANDOM_CENTERS
        _, labels, centers = cv2.kmeans(area_to_posterise_line, nbr_cluster, None, criteria, nbr_iterations, flags)
        centers.astype(np.uint8)
        res = centers[labels.flatten()]
        posterised_area_to_posterise = res.reshape((np.shape(input_image)))

        return posterised_area_to_posterise

    print(f"{curr_func} -- Change the value of the following parameter 'nbr_cluster' (1 ~ 255)")

    return input_image

def sharpen_image(image: np.ndarray, ksize_1: int=7, ksize_2: int=7, sigma_1: float=1., sigma_2: float=2.) -> np.ndarray:
    """Function that sharpen an image.

    Args:
        image (np.ndarray): Input image (BGR).
        ksize_1 (int, optional): Kernel n°1 size (Odd number). Defaults to 7.
        ksize_2 (int, optional): Kernel n°2 size (Odd number). Defaults to 7.
        sigma_1 (float, optional): Blur value n°1. Defaults to 1.
        sigma_2 (float, optional): Blur value n°2. Defaults to 2.

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


def main() -> int:
    """main
    """
    curr_func = inspect.currentframe().f_code.co_name

    # Get small pictures
    originals = list(FOLDER_SAVE_DL_BD_NAME.glob("*" + EXT))
    originals.sort()

    # Get completed pictures
    completed = list(FOLDER_SAVE_UPSCALE_BD_NAME.glob("*" + EXT))

    # Get trained models
    models = list(FOLDER_MODELS.glob("*.pb"))

    if not models:
        print(f"{curr_func} -- No Model found in the {FOLDER_MODELS} folder...")
        return 1

    models.sort()
    selected_model = MODEL
    nbr_models = len(models)

    if not nbr_models >= selected_model:
        print(f"{curr_func} -- Model n°{selected_model} not found in the {FOLDER_MODELS} folder...")
        selected_model = nbr_models - 1
        print(f"{curr_func} -- Found {nbr_models} model(s)... Proceeding with Model n°{selected_model} instead...")

    # Select model
    model_path = models[selected_model]
    model_name = model_path.stem.split("_", maxsplit=1)[0].lower()
    model_scale = model_path.stem.split("_")[-1].replace("x", "")
    print(f"{curr_func} -- Using model {model_name} with {model_scale} scaling...")

    # Prepare model
    sup_res = cv2.dnn_superres.DnnSuperResImpl_create()
    sup_res.readModel(str(model_path))
    sup_res.setModel(model_name, int(model_scale))

    # Compute / Upscale / Clean each small image
    for cptr, img_path in enumerate(originals[len(completed):]):
        tic = time.time()

        print(f"{curr_func} -- Loading image {img_path.name}")
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)

        print(f"{curr_func} -- Upscaling image {img_path.name}")
        result = sup_res.upsample(img)

        print(f"{curr_func} -- Cleaning image {img_path.name}")
        img = area_posterise(img)
        img = sharpen_image(img)

        print(f"{curr_func} -- Saving image {img_path.name}")
        cv2.imwrite(str(FOLDER_SAVE_UPSCALE_BD_NAME/img_path.name), result)

        tac = time.time()
        toc = int((tac - tic)*100)
        toc = toc / 100

        # Estimate remaining time to complete the task
        total = toc * ((len(originals[len(completed):]) - cptr) - 1)
        total = total if total > 0 else 0

        days = int(total/(60*60*24))
        hours = int(total/(60*60))%24
        minutes = int(total/(60))%60
        seconds = int(total)%60

        print(f"{curr_func} -- Image {img_path.name} upscaled in {toc} second(s). \n\t Estimated remaining time: {days}d - {hours}h:{minutes}m:{seconds}s\n")

    return 0


if __name__ == '__main__':
    main()
