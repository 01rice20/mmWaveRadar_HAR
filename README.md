# Application of Cross-Frequency mmWave Radar in Human Activity Identification

Yi-Fan Lin*, Kai-Lung Hua

## Overview
![Visualize_2](https://github.com/user-attachments/assets/10ac0655-15be-48ff-a590-ef9d393dca12)

Human Activity Recognition (HAR) plays an important role in identifying and classifying diverse human actions or behaviors, facilitating applications such as security surveillance, sports analytics, healthcare, and human-computer interaction. 

However, datasets associated with human activity recognition often encounter challenges related to insufficient data and significant differences between sensors, leading to issues with data consistency.

In this study, we propose a novel multi-domain method with a dynamic module and weighted loss function. We preprocess radar raw data into distinct feature maps to maximize the richness of features within the limited data volume. Our proposed model, DSCM, contains learnable parameters to dynamically regulate the balance between attention and convolution, which can better adapt to the resolution differences caused by various mmWave radar sensor frequency bands.

Finally, with the combination of the Modified Dynamically Weighted Balanced (MDWB) loss function, it assigns higher weights to hard samples to improve prediction performance.

Experimental results indicate that our approach achieves an average accuracy of 98.01\%, precision of 98.02\%, recall of 97.59\% and F1-score of 97.72\% on the Ci4R Human Activity Dataset. Furthermore, we validate the effectiveness of our method through the proposed dataset. The model outperforms existing methods across all frequency bands, underscoring its superior generalization capability.

## Directory Structure
```
/.
├── README.md
├── data_preprocess/
│ ├── 10/
│ ├── 24/
│ ├── 60/
│ ├── 77/
│ ├── concate.py
│ ├── data24.py
│ ├── data24_2.py
│ ├── pair.py
│ ├── resize.py
│ ├── search.py
├── DSCM/ 
│ ├── model/
│ │ ├── attention.py 
│ │ ├── function.py 
│ │ ├── module.py 
│ ├── pretrain.py 
│ ├── train.py 
│ ├── test.py 
├── dataset/
├── weights/
└── requirements.txt
```
[dataset](https://www.icloud.com/iclouddrive/050gTB0jRPnOeIgnV_7LeZypg#dataset) and [weights](https://www.icloud.com/iclouddrive/072iJXqVRGm49cw6l0iw-6Eng#weights) can be downloaded in the following link.
## Enviroments
```
MY DEVICES:
    OS:  Ubuntu 20.04
    GPU: Nvidia Geforce RTX 2080 8GB x2
    PyTorch:
         CUDA 11.8
         PyTorch 2.0.1
    Python 3.10.11
```
## Requirements
The code requires python>=3.10.11, as well as pytorch>=2.0.1 and CUDA 11.8
```
git clone https://github.com/01rice20/mmWaveRadar_HAR.git
pip3 install -r requirements.txt
```
or download [docker image](https://www.icloud.com/iclouddrive/02diJoC8dvAJUera0qnjGc1YQ#radarHAR)
```
docker load < radarHAR.tar
```
## Getting Started
After downloading the dataset, first train the autoencoder model to store the pre-trained weights. Please note that different frequency bands need to be trained separately.

```
cd DSCM
python3 pretrain.py
```
The steps to fine-tune the model are as follow
```
python3 train.py --data 24
```
All weights are stored in weights folder.

## Our Baseline
S. Z. Gurbuz, M. M. Rahman, E. Kurtoglu, T. Macks, and F. Fioranelli, “Cross-frequency training with adversarial learning for radar micro-doppler signature classification (rising researcher),” in Radar Sensor Technology XXIV, vol. 11408, pp. 58–68, SPIE, 2020.
