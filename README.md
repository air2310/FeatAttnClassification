# Optimising classification of feature-based attention in frequency-tagged electroencephalography data

For a full and detailed description of this repository, see: Renton A.I., Painter D.R., & Mattingley J.B. (2021) Optimising classification of feature-based attention in frequency-tagged electroencephalography data. [link to be added upon publication]

## Author
Angela I. Renton
angie.renton23@gmail.com

## Note! 
This github repository contains code for the experimental task and analysis scripts. The data folders associated with these scripts are stored at [Insert Link on Publication]

## Background
Brain-computer interface (BCI) and neurofeedback training protocols are a rapidly expanding field of study, and require accurate and reliable real-time decoding of patterns of neural activity1. These protocols often exploit selective attention, a neural mechanism to prioritise the processing of sensory inputs that either match task-relevant stimulus features (feature-based attention) or that occur at task-relevant spatial locations (spatial attention). Within the visual modality, attentional modulation of neural responses to different inputs is well indexed by steady-state visual evoked potentials (SSVEPs) evoked by frequency-tagged stimuli. These signals are reliably present in single-trial EEG data, are largely resilient to common EEG artifacts, and allow separation of neural responses to numerous concurrently presented visual stimuli. To date, efforts to use single-trial SSVEPs to classify visual attention for BCI control have largely focused on spatial- as opposed to feature-based attention. Here, we present a dataset to address this gap and to allow for the development and benchmarking of algorithms to classify feature-based attention using single-trial EEG data. The dataset includes EEG and behavioural responses from 30 healthy human participants who performed a feature-based motion-discrimination task on frequency tagged visual stimuli. The data are organised in the Brain Imaging Data Structure (BIDS).

## Overview
This repository contains the code and data associated with the “FeatAttnClassification” project, which is aimed towards optimising the classification of feature-based attention in frequency-tagged Electroencephalography (EEG) data. The repository is organised into four sub-folders. 
-	The “Data” folder contains EEG and behavioural responses from 30 healthy human participants who performed a feature-based motion-discrimination task on frequency tagged visual stimuli.
-	The “ExperimentalTask” folder contains the Matlab code used to run the feature-based motion-discrimination task and record the associated data. 
-	The “AnalysisScripts” folder contains Matlab scripts used to process the data and apply a number of different machine learning classification algorithms to classify the attended visual feature at any moment. 
-	The “Results” folder contains the files output by the “AnalysisScripts” folder
The data folder follows the EEG-BIDS specification for folder hierarchy. Critical information regarding the experimental task parameters, display settings, EEG recording settings and triggers is contained in the file “FeatAttnClassification\Data\helperdata.mat”.

## Data
The EEG data are organised according to the BIDS architecture within the “FeatAttnClassification\Data\” folder. Unprocessed raw data are stored for each participant in a .mat MATLAB data format within “FeatAttnClassification\Data\sourcedata\sub-*\eeg\”. These .mat files were converted to the BIDS compatible brain vision format and stored for each participant within “FeatAttnClassification\Data\sub-*\eeg\”. Raw data were recorded with a digital high pass filter at 1 Hz and low pass filter at 100 Hz. No further pre-processing has been applied to the data.
Behavioural data are stored for each participant under “FeatAttnClassification\Data\sourcedata\sub-*\behave\sub-*_task-FeatAttnDec_behav.mat”. These files contain all the variables created when the experimental task was run for each participant.

Please see the data descriptor [insert link] for a detailed description of the organisation of this data. 

## ExperimentalTask
This experimental task was designed to generate a dataset on which to train machine learning classifiers to discriminate between attended and unattended features. Participants were tasked with monitoring a field of randomly moving, flickering dots in order to identify short bursts of coherent motion in a cued colour (black or white). A cue presented before each 15 s trial indicated whether participants should monitor the black or the white dots. Bursts of coherent motion occurred when a subset of the dots moved in the same direction (up, down, left, or right) for 500 ms. During half the trials, black and white dots were presented concurrently, and participants were cued to monitor for bursts of coherent motion in only one of the two colours. During the remaining trials, only the cued coloured dots were presented, artificially mimicking complete attentional suppression of the uncued colour. Dots in different colours flickered at different frequencies (6 Hz, 7.5 Hz; colour and frequency were counterbalanced), thus evoking steady-state visual evoked potentials (SSVEPs). When only one colour was present, SSVEPs were only evoked at a single frequency counterbalanced across trials.
The MATLAB files used to present the experimental task are stored in “FeatAttnClassification\ExperimentalTask”. The main script used to run the experimental task is stored in “FeatAttnClassification\ExperimentalTask\main_RTAttnMethods.m”. This script relies on a number of functions, which are stored in “FeatAttnClassification\ExperimentalTask\Functions\”.

Please see the data descriptor [insert link] for a detailed description of the experimental task. 

## AnalysisScripts
We designed the current study to create a dataset that would allow the scientific community to design and benchmark different approaches to real-time feature-based attention classification. To validate the suitability of this dataset for this purpose, we compared the efficacy of 6 different algorithms for classifying the target of feature-based attention, using different combinations of training features. 
A number of different approaches were used to discriminate which feature participants were attending to using short latency single-trial EEG data from a low density electrode array positioned over the posterior scalp. These included a Z-score difference approach, Linear Discriminant Analysis, Multi-Layer Perceptron, K-Nearest Neighbours and Canonical Correlation Analysis. 

The scripts used to run these analyses are stored in “FeatAttnClassification\AnalysisScripts\”. See the MainAnalysisScript (MATLAB live document) for an interactive outline of our approach. This is available as a MATLAB live script, as well as as a PDF and HTML document. 

Please see the data descriptor [insert link] for a detailed description of the analysis approach.

## Results
The results folder contains the files output by individual subject analyses (see AnalysisScripts folder). These individual data files are used to run the “collateResults2.m” script, which creates group aggregate plots and calculates statistical tests. 
Please see the data descriptor [insert link] for a detailed description of these collated results and the contents of the Results Folder. 
Usage Notes

EEG data is stored in the brain vision format. If using MATLAB, we recommend using the Fieldtrip Toolbox (https://www.fieldtriptoolbox.org/). See “FeatAttnClassification\AnalysisScripts\MainAnalysisScript.mlx” or “FeatAttnClassification\AnalysisScripts\Main4.m” and specifically the “function” “FeatAttnClassification\AnalysisScripts\Functions\getEEGBIDS.m” for a specific example of how to load these data in MATLAB. To load the EEG data in python, we recommend the MNE-python package (https://mne.tools/stable/index.html)

Behavioural data are stored in “.mat” files. This native MATLAB format can easily be read into MATLAB using the load function. Alternatively, if using Python, the files are HDF5 formatted and can be read using h5py. 

The Experimental task was written to run on MATLAB (2017a) using Psychtoolbox-3 (http://psychtoolbox.org/)

The Analysis Scripts were written in MATLAB using the statistics and machine learning toolbox. Then functions required to run these scripts are included in “AnalysisScripts\Functions”. 

