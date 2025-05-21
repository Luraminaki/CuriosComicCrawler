# CC DL-UPSCALER

I love [Sequential Art](https://www.collectedcurios.com/). But I don't always have a stable connection (And it seems the author has some issues with his server every once in a while)... And my sight isn't as good as it used to be... So I made this... Please bear in mind that I made this for entertaining and educative purpose only. Consider [supporting](https://www.patreon.com/collectedcurios) the author of this comic if you like his work.
Just edit the `config.json` file if you want to play around with the scripts...

## Versions

- 0.1.0-alpha: First release

## Table of content

<!-- TOC -->

- [CC DL-UPSCALER](#cc-dl-upscaler)
  - [Versions](#versions)
  - [Table of content](#table-of-content)
  - [Requirements](#requirements)
  - [Install](#install)
    - [About The Models](#about-the-models)
  - [Start Comic Crawler Downloader](#start-comic-crawler-downloader)
  - [Start Comic Crawled Upscaler](#start-comic-crawled-upscaler)

<!-- /TOC -->

## Requirements

- Model links [EDSR_Tensorflow](https://github.com/Saafke/EDSR_Tensorflow/tree/master/models), [TF-ESPCN](https://github.com/fannymonori/TF-ESPCN/tree/master/export), [FSRCNN_Tensorflow](https://github.com/Saafke/FSRCNN_Tensorflow/tree/master/models), [TF-LapSRN](https://github.com/fannymonori/TF-LapSRN/tree/master/export)

## Install

For `Python 3` installation, consult the following [link](https://www.python.org/downloads/)

Once done, open a new terminal in the directory `CuriosComicCrawler` and type the following command to create the python virtual environment.

```sh
python -m venv .venv
```

In the same terminal, activate the `.venv` previously created as follow, or as shown in [HowTo](https://docs.python.org/3/tutorial/venv.html#creating-virtual-environments), and install the project's dependencies.

- **Windows**

```sh
.venv\Scripts\activate
pip install -U -r requirements.txt
```

- **Unix** or **MacOS**

```sh
source .venv/bin/activate
pip install -U -r requirements.txt
```

### About The Models

As you saw in the [Requirements](#requirements) section, I'm using pre-trained models. Be sure to put these in the `data/models/` folder.


## Start Comic Crawler Downloader

```sh
 python3 downloadCC.py
```

## Start Comic Crawled Upscaler

```sh
 python3 upscaleCC.py
```
