# Can Fairness and Robustness Be Simultaneously Achieved Under Byzantine Attacks?
This hub stores the code for paper *Can Fairness and Robustness Be Simultaneously Achieved Under Byzantine Attacks?* 

## Install
1. Download the dependant packages (c.f. `install.sh`):
- python 3.8.10
- pytorch 1.9.0
- matplotlib 3.3.4

2. Download the dataset to the directory `./dataset` and create a directory named `./record`. The experiment outputs will be stored in `./record`.

## Runing
### Run AFL

`python AFL.py --aggregation trimmed-mean --attack sign_flipping --data-partition noniid --seed 1 `
> The arguments can be
> 
> `<aggregation>`: 
> - mean
> - trimmed-mean
> - median
> - geometric-median
> - faba
> - Krum
> - bulyan
>
> `<attack>`: 
> - none
> - gaussian
> - sign_flipping
> - disguise
>
> `<data-partition>`: 
> - iid
> - noniid

### Run other methods
To run other methods, such as $q$FFL, H-nobs, DRFL, one should simply run the corresponding `.py` file.
