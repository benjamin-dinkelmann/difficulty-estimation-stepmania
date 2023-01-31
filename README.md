# Ordinal Regression for Difficulty Estimation of StepMania Levels

This repository contains code related to the paper ["Ordinal Regression for Difficulty Estimation of StepMania Levels"](https://arxiv.org/pdf/2301.09485.pdf). The data extraction is patially based on ["Dance Dance Convolution"](https://github.com/chrisdonahue/ddc).

Accurately predicting the difficulty of StepMania charts is relevant for various interest groups. For instance, in tournament play a significant amount of effort is placed on accurately assigning difficulties to charts, because the difficulty commonly affects the advantage gained for a participant from passing, or better quading, this song in the tournament. Another example would be the DDR community. The DDR community is interested in curating a repository of DDR related content for casual play. However, DDR changed its difficulty scale somewhere throughout its development, leading to older and newer charts not agreeing on difficulties. This project could be used to re-rate old charts into the new scale or vice-versa.

This work also allows for comparing the difficulty perceptions of different packs. This is not obvious from the paper itself. However, using the regressor of the RED-SVM model, we can assign each pack its difficulty threshold, which can then be compared for different packs, visualizing potentially different perceptions.

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
