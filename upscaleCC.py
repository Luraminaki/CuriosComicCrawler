#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 18:45:31 2021

@author: Luraminaki
"""

#===================================================================================================
import sys
import time
import json
import inspect
import pathlib
import argparse

import cv2
import numpy as np

#pylint: disable=no-name-in-module
from __init__ import __version__
#pylint: enable=no-name-in-module
#===================================================================================================

CWD = pathlib.Path.cwd()
CONFIG_FILE = CWD/"config.json"


def area_posterise(input_image: np.ndarray, nbr_cluster: int=32, nbr_iterations: int=10) -> np.ndarray:
    """Quantization function.

    Args:
        input_image (np.ndarray): Input image (BGR).
        nbr_cluster (int, optional): Number of "color(s)" to keep. Defaults to 32.
        nbr_iterations (int, optional): Number of loop(s) to make the cluster(s). Defaults to 10.

    Returns:
        np.ndarray: Quantized image.
    """
    curr_func = (cf.f_code.co_name
                 if (cf := inspect.currentframe()) is not None
                 else 'None')

    if not 1 < nbr_cluster <= 255:
        print(f"{curr_func} -- Change the value of {nbr_cluster} for 'nbr_cluster' should be (1 ~ 255)")
        return input_image

    unique_values = np.unique(input_image)

    if len(unique_values) <= nbr_cluster:
        print(f"{curr_func} -- WARNING -- Requested clusters {nbr_cluster} can't be highter than the number of unique elements {len(unique_values)} to organise")
        return input_image

    if len(input_image.shape) > 1:
        area_to_posterise_line = input_image.reshape((-1, input_image.size))
    else:
        area_to_posterise_line = input_image

    area_to_posterise_line = np.float32(area_to_posterise_line)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    _, labels, centers = cv2.kmeans(area_to_posterise_line, nbr_cluster, None, criteria, nbr_iterations, flags)
    centers.astype(np.uint8)

    res: np.ndarray = centers[labels.flatten()]

    if len(input_image.shape) > 1:
        return res.reshape((np.shape(input_image)))

    return res.reshape((np.shape(-1)))


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
    # curr_func = inspect.currentframe().f_code.co_name

    blur1 = cv2.GaussianBlur(image, (ksize_1, ksize_1), sigma_1)
    blur2 = cv2.GaussianBlur(image, (ksize_2, ksize_2), sigma_2)

    temp_1 = cv2.addWeighted(image, 1.5, blur1, -0.5, 0)
    temp_2 = cv2.addWeighted(image, 1.5, blur2, -0.5, 0)
    sharpenned_image = cv2.addWeighted(temp_1, 0.5, temp_2, 0.5, 0)

    return sharpenned_image


def main(config: dict) -> int:
    """Main

    Args:
        config (dict): Content of the config file

    Returns:
        int: 0 if success, 1 if failed
    """
    curr_func = (cf.f_code.co_name
                 if (cf := inspect.currentframe()) is not None
                 else 'None')

    selected_model: int = abs(int(conf['model_nbr']))
    folder_models: pathlib.Path = CWD/f"{config['folder_data']}{config['folder_models']}"

    folder_save_dl_bd_name: pathlib.Path = CWD/f"{config['folder_data']}{config['folder_save_dl']}{config['BD_name']}"
    folder_save_upscale_bd_name: pathlib.Path = CWD/f"{config['folder_data']}{config['folder_save_upscale']}{config['BD_name']}"
    folder_save_dl_bd_name.mkdir(exist_ok=True, parents=True)
    folder_save_upscale_bd_name.mkdir(exist_ok=True, parents=True)

    # Get small pictures
    originals = sorted(folder_save_dl_bd_name.glob(f"*{config['ext']}"))

    # Get completed pictures
    completed = sorted(folder_save_upscale_bd_name.glob(f"*{config['ext']}"))

    # Get trained models
    models = sorted(folder_models.glob('*.pb'))

    if not models:
        print(f"{curr_func} -- No Model found in the {folder_models} folder...")
        return 1

    nbr_models = len(models)

    if  selected_model > (nbr_models - 1):
        print(f"{curr_func} -- Model n°{selected_model} not found in the {folder_models} folder...")
        selected_model = nbr_models - 1
        print(f"{curr_func} -- Found {nbr_models} model(s)... Proceeding with Model n°{selected_model} instead...")

    # Select model
    model_path = models[selected_model]
    model_name = model_path.stem.split('_', maxsplit=1)[0].lower()
    model_scale = model_path.stem.split('_')[-1].replace('x', '')
    print(f"{curr_func} -- Using model {model_name} with {model_scale} scaling...")

    # Prepare model
    sup_res = cv2.dnn_superres.DnnSuperResImpl_create()
    sup_res.readModel(str(model_path))
    sup_res.setModel(model_name, int(model_scale))

    avg_completion_time = 0
    # Compute / Upscale / Clean each small image
    for cptr, img_path in enumerate(originals[len(completed):]):
        tic = time.time()

        print(f"{curr_func} -- Loading image {img_path.name}")
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)

        print(f"{curr_func} -- Upscaling image {img_path.name}")
        result = sup_res.upsample(img)

        print(f"{curr_func} -- Cleaning image {img_path.name}")
        img = area_posterise(img, config['gray_values'])
        img = sharpen_image(img)

        print(f"{curr_func} -- Saving image {img_path.name}")
        cv2.imwrite(str(folder_save_upscale_bd_name/img_path.name), result)

        tac = time.time()
        toc = int((tac - tic)*100)
        toc = toc / 100

        avg_completion_time = avg_completion_time + toc

        # Estimate remaining time to complete the task
        total = (avg_completion_time / (cptr + 1)) * ((len(originals[len(completed):]) - cptr) - 1)
        total = total if total > 0 else 0

        days = int(total/(60*60*24))
        hours = int(total/(60*60))%24
        minutes = int(total/(60))%60
        seconds = int(total)%60

        print(f"{curr_func} -- Image {img_path.name} processed in {toc} second(s). \n\t Estimated remaining time: {days}d - {hours}h:{minutes}m:{seconds}s\n")

    return 0


if __name__ == '__main__':
    c_func = (cf.f_code.co_name
              if (cf := inspect.currentframe()) is not None
              else 'None')
    m_tic = time.perf_counter()

    print(f"{c_func} -- Version {__version__}")

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--configuration', help='Configuration file location', required=False)
    args = vars(parser.parse_args())

    config_file = args.get('configuration', None)
    if config_file is None:
        config_file = CONFIG_FILE
    else:
        config_file = pathlib.Path(config_file)

    if not config_file.is_file():
        print(f"{c_func} -- ERROR -- {config_file} does not exist -- Aborting")
        sys.exit(1)

    try:
        with config_file.open('r', encoding='utf-8') as f:
            conf: dict = json.load(f)
    except Exception as err:
        print(f"{c_func} -- ERROR -- Loading {config_file} failed -- {repr(err)}")
        sys.exit(1)

    print(f"{c_func} -- Current time is: {time.asctime(time.localtime())}")
    print(f"{c_func} -- {config_file} acquired")
    crash = False

    try:
        main(config=conf)
    except Exception as err:
        crash = True
        print(f"{c_func} -- ERROR -- App chrashed at {time.asctime(time.localtime())} -- {repr(err)}")

    m_tac = time.perf_counter() - m_tic
    print(f"{c_func} -- Ellapsed time: {round(m_tac, 3)}")

    if crash:
        sys.exit(1)

    sys.exit(0)
