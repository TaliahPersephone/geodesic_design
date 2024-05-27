# data.py
""" This module contains a class for handling optimization data. """
import os
import pandas as pd
import matplotlib.pyplot as plt


class OptimizationDataHandler:
    """
    Data handling object for saving, loading, and plotting optimizer data

    Parameters
    ----------
    target_unitary : optimize.Optimizer
        Pre-initialised Optimizer object

    Attributes
    ----------
    config : dict
        dict of values to include in the filename that determines how to save and load optimisation data
    folder : str, default='data'
        Main folder to save the data
    extension : str, default='csv'
        File extension of the saved data
    """

    def __init__(self, optimizers=None, load_data=True, load_filepath=None,
                 folder='data', extension='csv', float_precision=4, **config):
        self.config = config
        self.folder = folder
        self.extension = extension
        self.samples = 0
        self.optimizers = []
        self.float_precision = float_precision
        if load_data:
            self._load_optimization_data(load_filepath)
        for optimizer in optimizers or []:
            self.add_optimizer(optimizer)

    def steps(self, sample):
        """ Return steps of requested sample. """
        self._is_optimizer_loaded()
        index = self._find_sample(sample)
        return self.optimizers[index]['steps']

    def parameters(self, sample):
        """ Return parameters of requested sample. """
        self._is_optimizer_loaded()
        index = self._find_sample(sample)
        return self.optimizers[index]['parameters']

    def fidelities(self, sample):
        """ Return fidelities of requested sample. """
        self._is_optimizer_loaded()
        index = self._find_sample(sample)
        return self.optimizers[index]['fidelities']

    def running_fidelities(self, sample):
        """ Return a running max fidelity at each step of requested sample. """
        self._is_optimizer_loaded()
        index = self._find_sample(sample)
        fs = self.optimizers[index]['fidelities']
        running_fidelities = [fs[0]]
        found_max = fs[0]
        for i, f in enumerate(fs[1:]):
            fr = f if f > found_max else found_max
            found_max = fr
            running_fidelities.append(fr)
        return running_fidelities

    def step_sizes(self, sample):
        """ Return step_sizes of requested sample. """
        self._is_optimizer_loaded()
        index = self._find_sample(sample)
        return self.optimizers[index]['step_sizes']

    def max_fidelity(self, sample):
        """ Return max fidelity achieved by requested sample. """
        self._is_optimizer_loaded()
        index = self._find_sample(sample)
        return max(self.optimizers[index]['fidelities'])

    def add_optimizer(self, optimizer):
        """ Append provided optimizer data to optimizers attribute. """
        self.samples += 1
        self.optimizers.append(self._construct_data_dict(optimizer))

    def save_data(self):
        """ Save data, append if parameter data already exists. """
        self._is_optimizer_loaded()
        filepath = self._generate_filepath()

        # Make directory if it doesn't exist
        directory = f"{os.getcwd()}/{os.path.dirname(filepath)}"
        os.makedirs(directory, exist_ok=True)

        dfs = {}
        for i, optimizer in enumerate(self.optimizers):
            dfs[i+1] = pd.DataFrame(optimizer)
        concatdfs = pd.concat(dfs)

        concatdfs.to_csv(filepath, index=False)
        return dfs

    def exists(self):
        """ Return whether optimization data for current parameters exists. """
        filepath = self._generate_filepath()
        return os.path.exists(filepath)

    def plot_parameters(self, basis, sample, title=False, figsize=None):
        """ Plot parameters of requested sample. """
        labels = ["".join(map(str, label)) for label in basis.labels]
        _, ax = plt.subplots(figsize=figsize or [14, 6])
        ax.bar(labels, self.parameters(sample)[-1])
        ax.grid()
        if title:
            plt.title(title)
        plt.xticks(rotation=90)
        plt.show()

    def plot_fidelities(self, title=False, figsize=None):
        """ Plot fidelities of requested sample. """
        _, ax = plt.subplots(figsize=figsize or [12, 6])
        for s in range(1, self.samples+1):
            ax.plot(self.steps(s), self.fidelities(s))
        if title:
            plt.title(title)
        plt.show()

    def plot_step_sizes(self, title=False, figsize=None):
        """ Plot step sizes of requested sample. """
        _, ax = plt.subplots(figsize=figsize or [12, 6])
        for s in range(1, self.samples+1):
            ax.plot(self.steps(s), self.step_sizes(s))
        if title:
            plt.title(title)
        plt.show()

    def _load_optimization_data(self, filepath=None):
        """ Load past data for current parameter configuration. """
        filepath = filepath or self._generate_filepath()

        # Return if the file does not exist
        if not os.path.isfile(filepath):
            return 0

        opt_data = pd.read_csv(filepath,
                               index_col=False,
                               )
        opt_data.dropna(axis=0, inplace=True)

        # Set samples attribute to total number of samples
        samples = opt_data['sample'].max()
        self.samples += samples

        # Load each run as a list
        for sample in range(1, samples+1):
            sample_data = opt_data[opt_data['sample'] == sample]
            self.optimizers.append(sample_data.to_dict('list'))

        return self.optimizers

    def _generate_filepath(self, filename='optimization_data', extension='csv'):
        """ Construct filename and folder from config parameters. """
        conf_folder = []

        for attr, value in self.config.items():
            # Assume 'name' is the only string
            if attr == 'name':
                continue

            # Only list the attr if a boolean is set to True
            if isinstance(value, bool):
                if value:
                    conf_folder += [attr]

            # Otherwise format numbers
            else:
                float_format = f".{self.float_precision}f" if isinstance(value, float) else ''
                minus_format = 'm' if value < 0 else ''
                conf_folder += [f"{attr}={minus_format}{abs(value):{float_format}}"]

        pathname = f"{self.folder}/{self.config['name']}/{'_'.join(conf_folder)}"
        filename = f"{filename}.{extension}"

        return f"{pathname}/{filename}"

    def _construct_data_dict(self, optimizer):
        """ Return dictionary from provided optimizer. """
        data_dict = {
            'sample': [self.samples] * (optimizer.steps[-1]+1),
            'steps': optimizer.steps,
            'parameters': [list(p) for p in optimizer.parameters],
            'fidelities': optimizer.fidelities,
            'step_sizes': optimizer.step_sizes,
        }
        return data_dict

    def _is_optimizer_loaded(self):
        """ Raise a ValueError if no optimizers are loaded. """
        if not self.optimizers:
            raise ValueError('Must use add_optimizer before you can check the parameters.')

    def _find_sample(self, sample):
        """ Return idx of requested sample. """
        for i, dic in enumerate(self.optimizers):
            if sample in dic['sample']:
                return i
        return -1
