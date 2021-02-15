About
=====

The `xerus` library is a general purpose library for numerical calculations with higher order tensors, Tensor-Train Decompositions / Matrix Product States and other Tensor Networks.
The focus of development was the simple usability and adaptibility to any setting that requires higher order tensors or decompositions thereof. 

For tutorials and a documentation see <a href="http://libxerus.org">the documentation</a>.

The source code is licenced under the AGPL v3.0. For more details see the LICENSE file.


Features
========

+ Intuitive notation for expressions involving tensors of arbitrary degree: `A(i,j) = B(i,k,l) * C(k,j,l)`;
+ Lazy evaluation of tensor expressions with automatically optimized contraction orders.
+ Full implementation of all common functionality of the Tensor Train (or Matrix Product State) decompositions with either dense or sparse component tensors.
+ Implementations of common algorithms like the ALS, (multi-site-)DMRG, ASD, AFD, CG, and some less common ones e.g. to find the maximal entries in a Tensor Train.


Building the Xerus library
==========================

Building from source
--------------------

Copy the default configuration and modify it for your needs
> cp config.mk.default config.mk
> nano config.mk

Test whether everything works correctly with
> make test -j4

build (and optionally install) the library with
> make all -j4
> sudo make install

and you should be ready to use the library.
To install the python bindings run
> make python3
> pip install .

For more details see <a href="https://www.libxerus.org/building_xerus/">the "Building Xerus" page in the documentation</a>.

Building a conda package
------------------------

Modify `conda/meta.yaml`, `build.sh` and `run_test.sh` as needed and build the conda package.
For example, to create a conda package of the `SALSA` branch of `xerus` run the following commands.
> cd conda
> conda-build . -c conda-forge --python=3.7 --numpy=1.20
> conda install --use-local xerus_salsa


Issues
======

Should you have any problems with the library do not hesitate to contact us at contact[at]libxerus.org or describe your problem in the issuetracker.

