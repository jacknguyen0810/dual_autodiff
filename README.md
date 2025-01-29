# dual_autodiff

Author: [Phong-Anh Nguyen Trinh - pan31](https://github.com/phong-anh-nguyen-trinh)

Dual Number Automatic Differentiation Package

Computing 1 Module Coursework Project 2024 for Data Intensive Science at the University of Cambridge

## Installation
Once the repository has been cloned, run the following commands to install the package (preferably in a virtual environment):

- Navigate to the root directory
```
cd ~/pan31
```
- Install the Python package
```
pip install -e .
```
- Navigate to the dual_autodiff_x directory
```
cd dual_autodiff_x
```
- Build the Cython package
```
pip install -e .
```

## Usage
Two examples of usage can be found in demo folder. 

- **Automatic differentiation of a function of one variable.** 

This is found in dual_autodiff/demo/diff_demo.py.

This example can be run by simply calling the diff_demo.py file, or if in an IDE like VSCode, by running the ```Run``` button. 

- **Cython Timing Comparison**

This is found in dual_autodiff/demo/cython_timing.py.

This example can be run by simply calling the cython_timing.py file, or if in an IDE like VSCode, by running the ```Run``` button. This should result in a plot of the time taken for the basic operations using both Python and Cython. 

## Documentation

Documentation can be found in the docs, specifically in docs/build/html/index.html. 

These have been pre-built, but can easily be wiped using the following command:

```make clean``` 

or on WINDOWS:

```./make.bat clean```

and then run the following command to build the documentation:

```make html```

or on WINDOWS:

```./make.bat html```. 

TODO: Add link to READTHEDOCS

## Linux Wheels

The linux wheels can be found in the following folder:

```
dual_autodiff/dual_autodiff_x/wheelhouse
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request. 

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.
