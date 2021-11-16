# CC DL-UPSCALER

## Version 0.1.0

## Requirements

- Python3 (`apt install python3-dev` & `apt install python3-setuptools` & `apt instal python3-pip`)
- Numpy (`apt install python3-numpy`)
- OpenCV (⩾4.5) (`pip3 install opencv-contrib-python`)
- Model links (`https://github.com/Saafke/EDSR_Tensorflow/tree/master/models`, `https://github.com/fannymonori/TF-ESPCN/tree/master/export`, `https://github.com/Saafke/FSRCNN_Tensorflow/tree/master/models`, `https://github.com/fannymonori/TF-LapSRN/tree/master/export`)

## Overview

I love `Sequential Art` from `https://www.collectedcurios.com/`. But I don't always have a stable connection (And it seems the author has some issues with his server every once in a while)... And my sight isn't as good as it used to be... So I made this... Please bear in mind that I made this for entertaining and educative purpose only. Consider supporting (financially) the author of this comic if you like his work.
Just edit the `config.json` file if you want to play around with the script...

## Start Comic Crawler Downloader

```sh
 python3 downloadCC.py
```

## Start Comic Crawled Upscaler

```sh
 python3 upscaleCC.py
```

## About The Models

As you saw in the `Requirements` section, I'm using pre-trained models. Be sure to put these in the `data/models/` folder.
