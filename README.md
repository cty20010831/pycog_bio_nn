# Excitatory-inhibitory recurrent neural networks for Delayed Match to Sample (DMS) Task

This forked repository is used to train an excitatory-inhibitory RNN for DMS task (see this [paper](https://www.sciencedirect.com/science/article/pii/S0896627320300611) for task instruction).

## Forked Repository
This project builds on the (single recurrent layer) excitatory-inhibitory RNN developed by Song et al. (2016), which allows specification of proportion of excitatory and inhibitory neurons (following Dale's Principle) and of connection of neurons within the recurrent layer. 

### Citation

This code is the product of work carried out in the group of [Xiao-Jing Wang at New York University](http://www.cns.nyu.edu/wanglab/):

* Song, H. F.\*, Yang, G. R.\*, & Wang, X.-J. "Training Excitatory-Inhibitory Recurrent Neural Networks for Cognitive Tasks: A Simple and Flexible Framework." *PLoS Comp. Bio.* 12, e1004792 (2016). (\* = equal contribution) 

### Requirements

The code is written in Python 2.7 and requires

* [Theano 0.7](http://deeplearning.net/software/theano/)

Optional but recommended if you plan to run many trials with the trained networks outside of Theano:

* [Cython](http://cython.org/)

Optional but recommended for analysis and visualization of the networks (including examples from the paper):

* matplotlib

The code uses (but doesn't require) one function from the [NetworkX](https://networkx.github.io/) package to check if the recurrent weight matrix is connected (every unit is reachable by every other unit), which is useful if you plan to train very sparse connection matrices.

### Setting up virtual environment
The first step is to install [python 2.7](https://www.python.org/downloads/release/python-2718/) and then locate the Python 2.7 executable on your system:
```
which python2.7
```
Note: `/Library/Frameworks/Python.framework/Versions/2.7/bin/python2.7` is where my Python 2.7 executable is located, which will be used in later virtual environment creation. 

Then, install `virtualenv` using Python 2.7:
```
/Library/Frameworks/Python.framework/Versions/2.7/bin/python2.7 -m pip install virtualenv
```

Next, create the virtual environment (named **pycog_venv**) using the correct `virtualenv`:
```
/Library/Frameworks/Python.framework/Versions/2.7/bin/python2.7 -m virtualenv -p /Library/Frameworks/Python.framework/Versions/2.7/bin/python2.7 pycog_venv
```

Finally, activate the virtual environment:
```
source pycog_venv/bin/activate
```

### Installation

Because you will eventually want to modify the `pycog` source files, we recommend that you "install" by simply adding the `pycog` directory to your `$PYTHONPATH`, and building the Cython extension to (slightly) speed up Euler integration for testing the networks by typing

```
python setup.py build_ext --inplace
```

You can also perform a "standard" installation by going to the `pycog` directory and typing

```
python setup.py install
```

### Examples

Example task specifications, including those used to generate the figures in the paper, can be found in `examples/models`.

Training and testing networks involves some boring logistics, especially regarding file paths. You may find the script `examples/do.py` helpful as you start working with your own networks. For instance, to train a new network we can just type (from the `examples` directory)
```
cd examples
```

#### sinewave

For this particular example we've also directly included code for training and plotting the result, so you can simply type

```
python models/sinewave.py
```

For the normal workflow, 

- train the model:
```
python do.py models/sinewave train
```

- analyze the resting state model: 
```
python do.py models/sinewave restingstate
```

- plot cost history:
```
python do.py models/sinewave costs
```

- clean files in the temporary `work` directory:
```
python do.py models/sinewave clean
```

#### lee 
- train the model:
```
python do.py models/lee train
```

- analyze the resting state model: 
```
python do.py models/lee restingstate
```

- plot cost history:
```
python do.py models/lee costs
```

- run analysis on variants of the Lee sequence generation task
```
# Define SCRATCH as Environmental Variable for saving scratch files
export SCRATCH=/Users/samcong/Library/CloudStorage/OneDrive-TheUniversityofChicago/Summer_2024/pycog_bio_nn/exampl
es/scratch

# Run the analysis
python do.py models/lee run analysis/lee trials 10
```

- clean files in the temporary `work` directory:
```
python do.py models/lee clean
```


### Notes

* The default recurrent noise level (used for most of the tasks in our paper) is rather high. When training a new task start with a value of `var_rec` that is small, then increase the noise for more robust solutions.

* A list of parameters and their default values can be found in `defaults.py`

* The default time step is also relatively large, so always test with a smaller time step (say 0.05) and re-train with a smaller step size if the results change.

* By default, recurrent and output biases are set to zero. If you encounter difficulties with training, try including the biases by setting `train_brec = True` and/or `train_bout = True`.

* If you still have difficulties with training, try changing the value of `lambda_Omega`, the multiplier for the vanishing-gradient regularizer.

* It's common to see the following warning when running Theano:

  ```
  RuntimeWarning: numpy.ndarray size changed, may indicate binary incompatibility
  rval = __import__(module_name, {}, {}, [module_name])
  ```

  This is almost always innocuous and can be safely ignored.


## DMS Task
In order to utilize this excitatory-inhibitory RNN for DMS task, we need to create new `model` scripts to tailor the experimental design of DMS task. Meanwhile, we also need to modify `analysis` scripts to customize the type of analysis we want to perform upon the trained RNN model. 