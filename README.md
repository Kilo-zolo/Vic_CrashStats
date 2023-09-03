# CrashStats Case Study

## Table of Contents

1. [Introduction](#introduction)
2. [Audience](#audience)
3. [EDA Questions Answered](#eda-questions-answered)
4. [Problem Statement](#problem-statement)
5. [Project Structure](#project-structure)
    - [models](#models)
    - [notebooks](#notebooks)
    - [src](#src)
6. [Setup](#setup)
7. [Usage](#usage)


## Introduction

This project aims to address road safety issues by using machine learning to analyze various factors that contribute to the severity of accidents and predict the severity of an accident on the basis of chosen variables.

## Audience

The primary audience for this project includes Victorian Local Councils and Victorian Police.

## EDA Questions Answered

The exploratory data analysis (EDA) aims to identify factors that contribute to the severity of accidents.

## Problem Statement

The project attempts to mitigate the number of collisions occurring due to specific road geometries, such as T-intersections and cross intersections.

## Project Structure

The project has a variety of directories, each with a specific purpose:

### `models`

This directory contains subdirectories for different machine learning models. Each subdirectory has:

- A Joblib file (`.joblib`): Serialized machine learning models.
- `plts`: Directory for plots related to the model.
- `reports`: Directory for evaluation reports or other relevant information about the model.

#### Models Included:

- `binary_lr_clf`: Likely a binary Logistic Regression classifier model.
- `binary_rf_clf`: Likely a binary Random Forest classifier model.
- `olr_clf`: Possibly some sort of Ordinal Logistic Regression classifier or another type of model.

### `notebooks`

Contains Jupyter notebooks for exploratory data analysis and possibly other tasks.

#### Files and Directories:

- `eda.ipynb`: Notebook for Exploratory Data Analysis.
- `outputs`: Directory containing outputs like heatmaps, CSV files, and other visualizations.

### `src`

The source code directory.

#### Python Files:

- `__init__.py`: Empty file to indicate that this directory should be considered a Python package.
- `data_prep.py`: Functions for loading and preparing data.
- `lr_model.py`: Script for training and evaluating a Logistic Regression model.
- `or_model.py`: Script for training and evaluating an Ordinal Regression model.
- `plots.py`: Functions for plotting metrics and visualizations.
- `pre_proc.py`: Functions for feature engineering and pre-processing.
- `rf_model.py`: Script for training and evaluating a Random Forest model.

## Setup

1. Clone the repository.
2. Install dependencies using `pip install -r requirements.txt`.
3. Set up environment variables as per `.env_sample`.

## Usage

- To run the notebooks, navigate to the `notebooks` directory and start Jupyter Notebook.
- To run models, navigate to the `src` directory and run the respective Python files.




