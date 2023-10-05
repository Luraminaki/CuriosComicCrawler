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

import requests
#===================================================================================================
__version__ = "0.1.0"

data_folder = "./data/"

with open("config.json", encoding='utf-8') as f:
    conf = json.load(f)

root_site = conf["root_site"]
BD_name = conf["BD_name"]
ends = conf["ends"]
extra = conf["extra"]
ext = conf["ext"]
folder_save_dl = data_folder + conf["folder_save_dl"]

os.makedirs(folder_save_dl + BD_name + "/", exist_ok=True)


def get_last_saved(save_path: str) -> int:
    """Function that will retrieve the last saved file's number.

    Args:
        save_path (str): Images path.

    Returns:
        int: Last saved number.
    """
    curr_func = inspect.currentframe().f_code.co_name

    available = glob.glob(save_path + '*.jpg')
    if len(available) > 0:
        available.sort()
        last_nbr_available = os.path.basename(available[-1]).split('_')[1]

        for cptr, digit in enumerate(last_nbr_available):
            if digit != '0':
                break

        return int(last_nbr_available[cptr:])
    return 0


def dl_and_save_img(link: str, save_path: str) -> bool:
    """Function that fetch an image from a provided link and saves it.

    Args:
        link (str): Image source link.
        save_path (str): Destination path.

    Returns:
        bool: True if Success. False if Failed.
    """
    curr_func = inspect.currentframe().f_code.co_name

    resource = requests.get(link, stream=True)
    if resource.ok:
        print(f"{curr_func} -- Saving: {os.path.basename(save_path)}\n")

        with open(save_path, "wb") as img:
            for chunk in resource.iter_content(1024):
                img.write(chunk)
        return True

    else:
        print(f"{curr_func} -- Image not found")
        return False


def main() -> None:
    curr_func = inspect.currentframe().f_code.co_name

    last_nbr_available = get_last_saved(folder_save_dl + BD_name + "/")
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
        image_name = BD_name + "_" + pic_nbr + "_" + ends + ext
        image_name_b = BD_name + "_" + pic_nbr + "_" + ends + "_" + extra + ext

        print(f"{curr_func} -- Downloading: {image_name}")
        ret = dl_and_save_img(root_site + image_name, folder_save_dl + BD_name + "/" + image_name)

        if not ret:
            print(f"{curr_func} -- Failed with {image_name} Retrying with {image_name_b}")
            ret_b = dl_and_save_img(root_site + image_name_b, folder_save_dl + BD_name + "/" + image_name)

            if not ret_b:
                print((f"{curr_func} -- Nothing to download. Process aborting..."))
                break

        time.sleep(1)
        cptr = cptr + 1


if __name__ == '__main__':
    main()
