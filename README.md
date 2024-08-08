# ModePlait

- [Introduction](#introduction)
- [Requirements](#requirements)
- [Usage](#usage)
- [Datasets](#datasets)
- [Experiments](#experiments)
  - [Baselines](#baselines)
  - [Causal discovering accuracy score](#causal-discovering-accuracy-score)
  - [Forecasting accuracy score](#forecasting-accuracy-score)

## Introduction
This is an official implementation of ModePlait. We focus on causal relationships that evolve over time in data streams and refer such relationships as "time-evolving causalities." We presented ModePlait, which aims to discover time-evolving causalities in multivariate co-evolving data streams, and forecast future values in a stream fashion simultaneously. The overview of our proposed model is following:

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
    git clone https://github.com/C-Naoki/ModePlait.git
    ```
2. Construct a virtual environment and install the required packages.
    ```bash
    make install
    ```
    - However, you are required to [pyenv](https://github.com/pyenv/pyenv#installation) and [poetry](https://python-poetry.org/docs/#installation) for above command to work.
    - If you prefer not to use `pyenv` or `poetry`, you can also use [`requirements.txt`](https://github.com/C-Naoki/ModePlait/blob/main/requirements.txt) created based on pyproject.toml.

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

(Other than `1. covid19`, everything is placed in the folder [`./data`](https://github.com/C-Naoki/ModePlait/blob/main/data))

## Experiments
### Baselines
We compared our algorithm with the following baselines for causal discovering, including CASPER, DARING, NoCurl, NOTEARS-MLP (NO-MLP), NOTEARS, LiNGAM, and GES.
We also compared with TimesNet, PatchTST, DeepAR, OrbitMap, and ARIMA for forecasting.

### Causal discovering accuracy score
We ran experiments on synthetic datasets with multiple temporal sequences to encompass various types of scenarios and ModePlait outperformed all competitors for every setting.

<p align="center">
  <img src=".\docs\assets\causal.png" align=center />
</p>

### Forecasting accuracy score
ModePlait achieved a high forecasting accuracy for every dataset, including synthetic and real-world datasets. ModePlait† is a limited version of our proposed method.

<p align="center">
  <img src=".\docs\assets\forecasting.png" align=center />
</p>
