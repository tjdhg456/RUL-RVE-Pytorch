# Pytorch Implementation of RUL-RVE
This is an unofficial PyTorch implementation of the paper [Variational encoding approach for interpretable assessment of remaining useful life estimation](https://www.sciencedirect.com/science/article/pii/S0951832022000321?via%3Dihub). This repo builds on the codebase of the official Tensorflow implementation [here](https://github.com/NahuelCostaCortez/Remaining-Useful-Life-Estimation-Variational).


# Requirements





# Implementation



# Results
Implementation | Dataset | lr | RMSE |
|:---:|:---:|:---:|:---:|
|Paper| FD001 | 0.001 | 13.42 |
|Our Implementation| FD001 | 0.005 |  11.05 |
|Paper| FD002 | 0.001 | 14.92 |
|Our Implementation| FD002 | 0.005 | 13.99 |
|Paper| FD003 | 0.001 | 12.51 |
|Our Implementation| FD003 | 0.001 | 12.08 |
|Paper| FD004 | 0.001 | 16.37 |
|Our Implementation| FD004 | 0.005 | 16.70 |

- We found the optimal LR for each dataset using grid-search (lr = choice(0.1, 0.01, 0.005, 0.001, 0.0001))


# References
1. Costa N, Sánchez L. Variational encoding approach for interpretable assessment of remaining useful life estimation. Reliab Eng Syst Saf. 2022;222:108353. doi:10.1016/J.RESS.2022.108353