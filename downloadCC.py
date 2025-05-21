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

import requests

#pylint: disable=no-name-in-module
from __init__ import __version__
#pylint: enable=no-name-in-module
#===================================================================================================

CWD = pathlib.Path.cwd()
CONFIG_FILE = CWD/"config.json"


def get_last_saved(save_path: pathlib.Path, ext: str) -> int:
    """Function that will retrieve the last saved file's number.

    Args:
        save_path (pathlib.Path): Images path.
        ext (str): Image extention.

    Returns:
        int: Last saved number.
    """
    curr_func = (cf.f_code.co_name
                 if (cf := inspect.currentframe()) is not None
                 else 'None')

    if not(available := sorted(save_path.glob(f'*{ext}'), reverse=True)):
        print(f"{curr_func} -- No prior save")
        return 0

    for latest in available:
        last_nbr_available = latest.stem.split('_', maxsplit=2)[1]

        if last_nbr_available.isalnum():
            return int(last_nbr_available)

    return 0


def dl_and_save_img(link: str, save_path: pathlib.Path, headers: dict) -> bool:
    """Function that fetch an image from a provided link and saves it.

    Args:
        link (str): Image source link.
        save_path (str): Destination path.
        headers (dict): Headers for the request.

    Returns:
        bool: True if Success. False if Failed.
    """
    curr_func = (cf.f_code.co_name
                 if (cf := inspect.currentframe()) is not None
                 else 'None')

    try:
        resource = requests.get(link, headers=headers, stream=True, timeout=5, verify=True)

    except Exception as error:
        print(f"Distant ressource {link} unreachable: {repr(error)}")
        return False

    if resource.ok:
        print(f"{curr_func} -- Saving: {save_path.name}\n")

        with open(save_path, 'wb') as img:
            for chunk in resource.iter_content(1024):
                img.write(chunk)
        return True

    print(f"{curr_func} -- Image not found -- HTTP_CODE: {resource.status_code}")
    return False


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

    folder_save_dl_bd_name: pathlib.Path = CWD/f"{config['folder_data']}{config['folder_save_dl']}{config['BD_name']}"
    folder_save_dl_bd_name.mkdir(exist_ok=True, parents=True)

    padding = int(config['padded'])
    fail_counter = int(config['fails'])

    last_page_available = get_last_saved(folder_save_dl_bd_name, config['ext'])
    print(f"{curr_func} -- Last downloaded: {last_page_available}")
    print(f"{curr_func} -- ========================================================================================")

    next_page = last_page_available + 1
    fail = 0

    while fail < fail_counter:
        base_image_name: str = f"{config['BD_name']}_{str(next_page).zfill(padding)}_{config['ends']}"
        possible_image_names: list[str] = [f"{base_image_name}{config['ext']}",
                                           f"{base_image_name}_{conf['extra']}{conf['ext']}"]

        for image_name in possible_image_names:
            print(f"{curr_func} -- Downloading: {image_name}")

            if not dl_and_save_img(config['root_site'] + image_name, folder_save_dl_bd_name/image_name, config['headers']):
                print(f"{curr_func} -- Failed with {image_name}")
                fail = fail + 1
                time.sleep(1)

            else:
                break

        time.sleep(1)
        next_page = int(next_page) + 1

    print((f"{curr_func} -- Nothing to download. Process aborting..."))

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
