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

import requests
#===================================================================================================
__version__ = "0.1.1"

CWD = pathlib.Path.cwd()

DATA_FOLDER = "data/"
CONFIG_FILE = "config.json"

with open(CWD/CONFIG_FILE, encoding='utf-8') as f:
    conf = json.load(f)

BD_NAME: str = conf["BD_name"]
EXT: str = conf["ext"]

ROOT_SITE: str = conf["root_site"]

ENDS: str = conf["ends"]
EXTRA: str = conf["extra"]

FOLDER_SAVE_DL: pathlib.Path = CWD/DATA_FOLDER/conf["folder_save_dl"]

FOLDER_SAVE_DL_BD_NAME: pathlib.Path = FOLDER_SAVE_DL/BD_NAME

FOLDER_SAVE_DL_BD_NAME.mkdir(exist_ok=True)


def get_last_saved(save_path: pathlib.Path) -> int:
    """Function that will retrieve the last saved file's number.

    Args:
        save_path (pathlib.Path): Images path.

    Returns:
        int: Last saved number.
    """
    curr_func = inspect.currentframe().f_code.co_name

    available = list(save_path.glob("*.jpg"))
    if len(available) > 0:
        available.sort()
        last_nbr_available = available[-1].stem.split('_')[1]

        cptr = 0
        for cptr, digit in enumerate(last_nbr_available):
            if digit != '0':
                break

        return int(last_nbr_available[cptr:])
    return 0


def dl_and_save_img(link: str, save_path: pathlib.Path) -> bool:
    """Function that fetch an image from a provided link and saves it.

    Args:
        link (str): Image source link.
        save_path (str): Destination path.

    Returns:
        bool: True if Success. False if Failed.
    """
    curr_func = inspect.currentframe().f_code.co_name

    try:
        resource = requests.get(link, stream=True, timeout=5)

    except Exception as error:
        print(f"Distant ressource {link} unreachable: {repr(error)}")
        return False

    if resource.ok:
        print(f"{curr_func} -- Saving: {save_path.name}\n")

        with open(save_path, "wb") as img:
            for chunk in resource.iter_content(1024):
                img.write(chunk)
        return True

    print(f"{curr_func} -- Image not found")
    return False


def main() -> None:
    """main function. Simply call it.

    Args:

    Returns:
        None: None.
    """
    curr_func = inspect.currentframe().f_code.co_name

    last_nbr_available = get_last_saved(FOLDER_SAVE_DL_BD_NAME)
    print(f"{curr_func} -- Last downloaded: {last_nbr_available}")

    cptr = last_nbr_available + 1
    nbr_zeros = 4

    while True:
        len_cptr = len(str(cptr))
        pic_nbr = ''
        i = 0

        while i < (nbr_zeros - len_cptr):
            pic_nbr = '0' + pic_nbr
            i = i + 1

        pic_nbr = pic_nbr + str(cptr)
        image_name = BD_NAME + "_" + pic_nbr + "_" + ENDS + EXT
        image_name_b = BD_NAME + "_" + pic_nbr + "_" + ENDS + "_" + EXTRA + EXT

        print(f"{curr_func} -- Downloading: {image_name}")
        ret = dl_and_save_img(ROOT_SITE + image_name, FOLDER_SAVE_DL_BD_NAME/image_name)

        if not ret:
            print(f"{curr_func} -- Failed with {image_name} Retrying with {image_name_b}")
            ret_b = dl_and_save_img(ROOT_SITE + image_name_b, FOLDER_SAVE_DL_BD_NAME/image_name)

            if not ret_b:
                print((f"{curr_func} -- Nothing to download. Process aborting..."))
                break

        time.sleep(1)
        cptr = cptr + 1


if __name__ == '__main__':
    main()
