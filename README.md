# Weighted Relevance and Redundancy Scoring

This repository contains a Python 3 implementation of the wRaR algorithm for feature selection on imbalanced dataset. The algorithm is based on RaR (see references), itself based on [this paper](http://ieeexplore.ieee.org/abstract/document/6228154/).  
The implementation is based around a heavily modified and extended version of this [HiCS implementation]((https://github.com/KDD-OpenSource/fexum-hics). It uses the [Gurobi Optimizer](http://www.gurobi.com) for the RaR Reasoning phase, you must acquire a license to be able to use it (academic licenses should be free).

## Install
Simply run 
```
python setup.py install
```

## How to use
Here is a quick example of running wRaR. It uses pandas dataframes 
```
import wrar
import pandas as pd

data = pd.read_csv('data.csv', header=None)
target = 'Class' # data[target] should be the target column

rar = wrar.rar.RaR(data)
rar.run(target, k=5, runs=200, split_iterations=20, compensate_imbalance=True)
```
After finishing, RaR prints a summary of the feature ranking. It is also available as attribute, i.e. `rar.feature_ranking`.

## Contributors (wRaR)
* [Daniel Thevessen](https://github.com/danthe96)

## Contributors (HiCS)
* [Marcus Pappik](https://github.com/marcuspappik)
* [Niklas Riekenbrauck](https://github.com/nikriek)
* [Daniel Thevessen](https://github.com/danthe96)
* [Alexander Meißner](https://github.com/Lichtso)
* [Axel Stebner](https://github.com/xasetl)
* [Louis Kirsch](https://github.com/timediv)
* [Julius Kunze](https://github.com/JuliusKunze)

## References

*  A. K. Shekar, T. Bocklisch, C. N. Straehle, P. I. Sánchez, and E. Müller, “Including multi-feature interactions and redundancy for feature ranking in mixed datasets,” in *Machine Learning and Knowledge Discovery in Databases - European Conference, ECML PKDD 2017, Macedonia, Skopje, September 18-22, 2017, Proceedings*, Lecture Notes in Computer Science, Springer, 2017.
* F. Keller, E. Muller, and K. Bohm, “Hics: High contrast subspaces for density-based outlier ranking,” in *Data Engineering (ICDE), 2012 IEEE 28th International Conference on*, pp. 1037–1048, IEEE, 2012.
