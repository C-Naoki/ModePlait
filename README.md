# ModePlait

- [Introduction](#introduction)
- [Requirements](#requirements)
- [Usage](#usage)
- [Datasets](#datasets)
- [Experiments](#experiments)
  - [Baselines](#baselines)
  - [Q1. Causal discovering](#q1-causal-discovering)
  - [Q2. Forecasting](#q2-forecasting)
  - [Q3. Ablation study](#q3-ablation-study)
  - [Experimental setup](#experimental-setup)
- [Contact](#contact)

## Introduction
This is an official implementation of ModePlait. We focus on causal relationships that evolve over time in data streams and refer such relationships as "time-evolving causality." We presented ModePlait, which aims to discover time-evolving causalities in multivariate co-evolving data streams, and forecast future values in a stream fashion simultaneously. The overview of our proposed model is following:

<p align="center">
  <img src=".\docs\assets\model.png" align=center />
</p>

## Requirements
- Python 3.9.15
- numpy == 1.23.5
- pandas == 1.5.3
- matplotlib == 3.8.2
- scikit-learn == 1.1.3
- scipy == 1.11.4

## Usage
1. Clone this repository.
    ```bash
    curl -o modeplait.zip https://anonymous.4open.science/api/repo/ModePlait-CB24/zip
    ```
2. Construct a virtual environment and install the required packages.
    ```bash
    make install
    ```
    - However, you are required to [pyenv](https://github.com/pyenv/pyenv#installation) and [poetry](https://python-poetry.org/docs/#installation) for above command to work.
    - If you prefer not to use pyenv or poetry, you can also use [`requirements.txt`](https://github.com/C-Naoki/ModePlait/blob/main/requirements.txt) created based on pyproject.toml.

    Specifically, the above command performs the following steps:
    1. if necessary, install Python 3.9.15 using pyenv, and then switch to this version.
    2. tell poetry to use python 3.9.15.
    3. install packages in `pyproject.toml`.
    4. attach the path file (i.e., `*.pth`) in the `site-packages/` for extending module search path.

    Please check the [`Makefile`](https://github.com/C-Naoki/ModePlait/blob/main/Makefile) for more details.

3. Run quick demos of ModePlait
    ```bash
    sh bin/google.sh
    ```
    If you want the command to continue running after logging out, you prepare `nohup/` directory and use `-n` option as shown below (using nohup).
    ```bash
    mkdir nohup
    sh bin/google.sh -n
    ```
    - The execution log is saved in `nohup/` directory.

## Datasets
1. covid19 [[link]](https://health.google.com/covid-19/open-data/)
2. web-search [[link]](https://trends.google.co.jp/trends/)
3. chicken-dance, exercise [[link]](http://mocap.cs.cmu.edu/)

- Other than `1. covid19`, everything is placed in the folder [`./data`](https://github.com/C-Naoki/ModePlait/blob/main/data)
- If you execute the command `sh bin/covid19.sh`, the `1. covid19` is automatically downloaded from Google COVID-19 Open Data Repository and placed in the folder `./data`.

## Experiments
### Baselines
We compared our algorithm with the following seven state-of-the-art baselines for causal discovering, namely CASPER, DARING, NoCurl, NOTEARS-MLP (NO-MLP), NOTEARS, LiNGAM, and GES.
We also compared with the following five leading competitors in time series forecasting, namely TimesNet, PatchTST, DeepAR, OrbitMap, and ARIMA.

### Q1. Causal discovering
We ran experiments on synthetic datasets with multiple temporal sequences to encompass various types of scenarios and ModePlait outperformed all competitors for every setting.

<p align="center">
  <img src=".\docs\assets\causal.png" align=center />
</p>

### Q2. Forecasting
ModePlait achieved a high forecasting accuracy for every dataset, including synthetic and real-world datasets.

<p align="center">
  <img src=".\docs\assets\forecasting.png" align=center />
</p>

### Q3. Ablation study
We can see that discovering the time-evolving causality adaptively is very helpful when forecasting in a streaming fashion.

<p align="center">
  <img src=".\docs\assets\ablation-study.png" align=center />
</p>

### Experimental setup
We conducted all above experiments on an Intel Xeon Platinum 8268 2.9GHz quad core CPU with 512GB of memory and running Linux.

## Contact
If you have any questions or concerns, please submit an [issue](https://github.com/C-Naoki/ModePlait/issues) or contact us (naoki88@sanken.osaka-u.ac).
