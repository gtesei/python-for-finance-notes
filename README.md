# python-for-finance-notes

[![License](http://img.shields.io/:license-mit-blue.svg?style=flat)](http://badges.mit-license.org)
[![Badges](http://img.shields.io/:badges-4/4-ff6799.svg)](https://github.com/badges/badgerbadgerbadger)

# Requirements

This tutorial requires the following packages:

- Python version 3.4+ 
    - likely Python 2.7 would be fine, but *who knows*? :P
- `numpy` version 1.10 or later: http://www.numpy.org/
- `scipy` version 0.16 or later: http://www.scipy.org/
- `matplotlib` version 1.4 or later: http://matplotlib.org/
- `pandas` version 0.16 or later: http://pandas.pydata.org
- `scikit-learn` version 0.15 or later: http://scikit-learn.org
- `keras` version 1.0 or later: http://keras.io
- `theano` version 0.8 or later: http://deeplearning.net/software/theano/
- `ipython`/`jupyter` version 4.0 or later, with notebook support

(Optional but recommended):

- `pyyaml`
- `hdf5` and `h5py` (required if you use model saving/loading functions in keras)
- **NVIDIA cuDNN** if you have NVIDIA GPUs on your machines.
    [https://developer.nvidia.com/rdp/cudnn-download]()

The easiest way to get (most) these is to use an all-in-one installer such as [Anaconda](http://www.continuum.io/downloads) from Continuum. These are available for multiple architectures.

# Recreate the Conda Environment

#### A. Create the Environment

```
bash setup_osx.sh 
```

#### B. Activate the new `pffn` Environment

```
source activate pffn
```
