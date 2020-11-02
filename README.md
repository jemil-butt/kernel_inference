# kernel_inference
Python code for kernel inference - optimal estimation of covariance functions. 
This repository provides basic functions, tutorials, figures related to estimation of covariance functions.

Code and figures are meant as supplementaries the paper "Inference of instationary covariance functions for optimal estimation in 
spatial statistics" by Jemil Butt, Andreas Wieser, and Josef Teichmann. The repository consists of three separate folders whose contents differ functionally. 

The root folder contains the python functions necessary to execute kernel inference, the determination of a covariance function maximizing an objective function featuring losses derived from discrepancies to observed data and a prior regularization term. The file "KI.py" contains those maximization algorithms separated into two cases differing by their complexity, whereas the file "Support_funs.py" contains auxiliary numerical functions.

The two folders "Tutorial_examples" and "Figures" contain material for recreating figures and illustrating the process and results of kernel inference. In the folder "Tutorial_examples", 8 examples of increasing complexity can be found in form of executable python code. Starting from an inference problem featuring simple observations of a one-dimensional stochastic process, nonzero means, linear constraints and inhomogeneous observation types are successively added.
In the folder "Figures", executable python code may be found, that allows recreating the figures and tables presented in the aforementioned paper. Further code is provided, it allows creating figures for the 6 specific kernel_inference examples listed in the applications-section of the paper.

The folder "Data" contains data necessary to calculate some of the root-mean-square errors provided by the table detailing performance results.

The code is provided with the sole intent being useful in education and teaching and we hope, it will be found to be useful. Although we took care to provide clean and well-documented programs, no guarantees as with respect to its correctness can be given and we are aware of a number of numerical instabilities and fail-cases.
