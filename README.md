# Ordinal Regression for Difficulty Estimation of StepMania Levels

This repository contains code related to the paper ["Ordinal Regression for Difficulty Estimation of StepMania Levels" (which includes an appendix)](https://arxiv.org/pdf/2301.09485.pdf). The data extraction is partially based on ["Dance Dance Convolution"](https://github.com/chrisdonahue/ddc).

Accurately predicting the difficulty of StepMania charts is relevant for various interest groups. For instance, in tournament play, a significant amount of effort is placed on accurately assigning difficulties to charts because the difficulty commonly affects the advantage gained for a participant from passing, or better quadding, this song in the tournament. Another example would be the DDR community. The DDR community is interested in curating a repository of DDR-related content for casual play. However, DDR changed its difficulty scale somewhere throughout its development, leading to older and newer charts not agreeing on difficulties. This project could be used to re-rate old charts into the new scale or vice-versa.

This work also allows for comparing the difficulty perceptions of different packs. This is not obvious from the paper itself. However, using the regressor of the RED-SVM model, we can assign each pack its difficulty threshold, which can then be compared for different packs, visualizing potentially different perceptions. [Meatball](https://github.com/bjfranks/Meatball) contains an example with respect to this use-case.

### Contact

If you are interested in this work and have questions, feel free to contact me via email (franks@cs.uni-kl.de) or on Discord (Noodles#7386).

## Attribution

```
@article{franks2023ordinal,
  title={Ordinal Regression for Difficulty Estimation of StepMania Levels},
  author={Franks, Billy Joe and Dinkelmann, Benjamin and Fellenz, Sophie and Kloft, Marius},
  journal={arXiv preprint arXiv:2301.09485},
  year={2023}
}
```

## Requirements
* numpy
* scikit-learn
* pandas
* matplotlib
* pytorch

## Example Use

### Disclaimer

The code will create persistent artifacts on your disk, such as an internal array representation of the charts for easier and faster access, or model parameters after training.
The repository is easiest to use if you create the folders next to the code files.
Alternatively you can set a different root directory or choose each folder individually.
You will find the relevant parser arguments in each code file.

### Step by Step Use
1. **Create a directory for your packs or datasets and put your packs inside:** <br>
 Default directory: `data/raw/` relative to where you are launching the code from <br>
 Note: Generally, using packs directly works just fine. But you can also bundle multiple packs or any songs into one folder. They are then taken as one dataset and referred to by that folder's name
 
2. **Extract the internal data representation for your datasets:** <br>
 Default command: `python extract_features.py` extracts time series for our model <br>
 The extracted data will then be stored in `data/time_series` <br>
 Note: If you wish to extract data for evaluating the baseline, add `-extract_patt` to the command.
 
3. **Fit the model** <br>
 Default command: `python run_model.py -dataset _your_dataset_name_` <br>
 Will train a REDSVM style TransformerEncoder Model on one split of the designated dataset, evaluate it thereafter and store the predictions in a file. <br>
 CrossValidation, different loss functions, and hyperparameters can be set via additional arguments. <br>
 Note: Only a single set of model parameters per model and dataset name will be saved in this standard case. When using cross validation, the models for all runs are stored together in a separate folder.
 
 4. **Evaluate the model**
    1. On other datasets:<br>
    To get predictions on your other packs/datasets that are in line with the packs you trained on
    2. Based on human validated pairs: <br>
    Please take a look at `model_evaluations.py` to learn more about how packs for user experiments can be generated and evaluated for a given model
 
