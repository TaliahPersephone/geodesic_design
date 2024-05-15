# Unitary gate design with time-independent Hamiltonians

## GND package

GND implements the Geodesic uNitary Design as in the paper [Geodesic Algorithm for Unitary Gate Design with Time-Independent Hamiltonians](https://arxiv.org/abs/2401.05973). 

Target gates are defined as classes in the `configs.py`.
The `__str__` method returns the name of the folder saved under `data` and `__dir__` gives the unique variables in the file name.
The config also contains information such as the number of qubits. 

The Pauli word basis is given by `gnd.basis.construct_two_body_pauli_basis` or `gnd.basis.construct_full_pauli_basis`, for the provided number of qubits.

The optimizer is called from `gnd.optimize`, and requires a target unitary, basis, and some initial parameters.

The data from the optimizer is then stored in the `gnd.data.OptimizationData` class, which handles the saving and loading of data.
The data stored is the "steps", "parameters", "fidelities", and "step\_sizes".
The data class takes a list of optimizers and stores them as a CSV file in the correct folder defined by the config.
Data can also be loaded using the data class and the correct config.

## Installation and Usage

From the root project directory, install the package using:

```shell
make install
```

This installs the python package alongside a CLI interface to the active python bin directory.
The CLI interface is called via:

```shell
geodesic_unitary_design
```

Documentation for this is on the backlog.

## Benchmarks

As comparison, we provide two benchmarks:

1. the work by Innocenti (2020):
> Luca Innocenti, Leonardo Banchi, Alessandro Ferraro, Sougato Bose, Mauro Paternostro (2020). "*Supervised learning of time-independent Hamiltonians for gate design*". [*New J. Phys.* **22** 065001](https://doi.org/10.1088/1367-2630/ab8aaf) ([arXiv:1803.07119](https://arxiv.org/abs/1803.07119)).

They kindly provided code at https://github.com/lucainnocenti/quantum-gate-learning-1803.07119 that we adapted for our purposes.
See the folder `innocenti` for details.

2. a naive gradient descent method in the Lie algebra.
See the folder `jax_gd` for details.

## Development Setup

To set up the project for development, from the root project directory run: 

```shell
make install-dev
```

This will install the package alongside additional requirements for development. 
