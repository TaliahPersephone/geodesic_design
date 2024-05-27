# cli.py
""" This module provides a CLI interface for using gnd. """
import argparse
import sys
import logging
import numpy as np
from gnd import basis, optimize, data, unitaries


logger = logging.getLogger(__name__)


def parse_input():
    """ Parse arguments from stdin. """
    parser = argparse.ArgumentParser(description='Set parameters')
    parser.add_argument('-g', '--gate', default='cnot',
                        choices=['cnot', 'cz', 'cphase', 'cswap', 'qft', 'wkpx', 'wkpz'],
                        help='the target gate to compute.')
    parser.add_argument('-d', '--dqudits', type=int, default=2,
                        choices=[2, 3],
                        help='the dimension of qudits to use, currently only qubits and qutrits are supported.')
    parser.add_argument('-n', '--nqudits', type=int, default=3,
                        help='the number of qudits, the k-parity or number of control dits will be inferred.')
    parser.add_argument('-p', '--precision', type=float, default=0.999,
                        help='the precision required for early termination.')
    parser.add_argument('--data-name-precision', type=int, default=4,
                        help='the precision to use for the name of the data folder.')
    parser.add_argument('-s', '--steps', type=int, default=1000,
                        help='the number of steps to be computed before terminating.')
    parser.add_argument('-i', '--instance', type=int, default=3,
                        help='used to compute the seed for generating the initial anzats.')
    parser.add_argument('--commute', action='store_true',
                        help='require the anzats at each step to commute.')
    parser.add_argument('--rydberg', action='store_true',
                        help='require the anzats at each step to be constructable from a Rydberg system.')
    parser.add_argument('-r', '--rerun', action='store_true',
                        help='rerun optimization for parameters even if optimization data already exists.')
    parser.add_argument('--name',
                        help='override the gate name for the output folder.')
    parser.add_argument('-v', action='store_true',
                        help='log information while running.')

    return parser.parse_args()


def main():
    """ Run optimization and process output data. """
    args = parse_input()
    gate = args.gate
    dqudits = args.dqudits
    nqudits = args.nqudits
    seed = args.instance * 2 ** 8

    # Get unitary from gnd.unitaries
    unitary_method = getattr(unitaries, gate)
    unitary, name = unitary_method(nqudits, dqudits)

    # Replace name if provided
    name = args.name or name

    # Construct config
    config = {
        'dqudits': dqudits,
        'precision': args.precision,
        'max_steps': args.steps,
        'commute': args.commute,
        'rydberg': args.rydberg,
        'seed': seed,
        'name': name
    }

    # Set logging to info if running verbose
    if args.v:
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    print(f"Running qu{dqudits}it {name} with seed {seed}, " + 
          f"{'commuting ansatz ' if args.commute else 'no ansatz '}" +
          f"and {args.steps} steps")

    if dqudits == 2:
        full_basis = basis.construct_full_pauli_basis(nqudits)
        projection_basis = basis.construct_two_body_pauli_basis(nqudits)
    else:
        full_basis = basis.construct_full_gellmann_basis(nqudits)
        projection_basis = basis.construct_two_body_gellmann_basis(nqudits)

    np.random.seed(seed)

    dat = data.OptimizationDataHandler(load_data=False, float_precision=args.data_name_precision, **config)

    if dat.exists() and not args.rerun:
        logger.info("Data already exists, skipping...")
    else:
        logger.info("Running optimization...")
        opt = optimize.Optimizer(
            unitary,
            full_basis,
            projection_basis,
            precision=args.precision,
            max_steps=args.steps,
            commute=args.commute)

        dat = data.OptimizationDataHandler(
                optimizers=[opt],
                load_data=True,
                float_precision=args.data_name_precision,
                **config
            )

        dat.save_data()
