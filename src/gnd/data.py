# data.py
""" This module contains a class for handling optimization data. """
import os
import pandas as pd
import numpy as np
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
    folder : str, default="data"
        Main folder to save the data
    extension : str, default="csv"
        File extension of the saved data
    """

    def __init__(self, optimizers=None, load_data=True, folder="data", extension="csv", **config):
        self.config = config
        self.folder = folder
        self.extension = extension
        self.samples = 0
        self.optimizers = []
        if load_data:
            self._load_optimization_data()
        for optimizer in optimizers or []:
            self.add_optimizer(optimizer)

    def steps(self, sample):
        self._is_optimizer_loaded()
        index = self._find_sample(sample)
        return self.optimizers[index]["steps"]

    def parameters(self, sample):
        self._is_optimizer_loaded()
        index = self._find_sample(sample)
        return self.optimizers[index]["parameters"]

    def fidelities(self, sample):
        self._is_optimizer_loaded()
        index = self._find_sample(sample)
        return self.optimizers[index]["fidelities"]

    def running_fidelities(self, sample):
        self._is_optimizer_loaded()
        index = self._find_sample(sample)
        fs = self.optimizers[index]["fidelities"]
        running_fidelities = [fs[0]]
        for i, f in enumerate(fs[1:]):
            fr = f if f > running_fidelities[i-1] else running_fidelities[i-1]
            running_fidelities.append(fr)
        return running_fidelities

    def step_sizes(self, sample):
        self._is_optimizer_loaded()
        index = self._find_sample(sample)
        return self.optimizers[index]["step_sizes"]

    def max_fidelity(self, sample):
        self._is_optimizer_loaded()
        index = self._find_sample(sample)
        return max(self.optimizers[index]["fidelities"])

    def add_optimizer(self, optimizer):
        self.samples += 1
        self.optimizers.append(self._construct_data_dict(optimizer))

    def save_data(self):
        self._is_optimizer_loaded()
        filepath = self._generate_filepath()
        dfs = {}
        for i, optimizer in enumerate(self.optimizers):
            dfs[i+1] = pd.DataFrame(optimizer)
        concatdfs = pd.concat(dfs)
        concatdfs.to_csv(filepath, index=False)
        return dfs

    def exists(self):
        filepath = self._generate_filepath()
        return os.path.exists(filepath)

    def plot_parameters(self, basis, sample, title=False, figsize=None):
        labels = ["".join(map(str, label)) for label in basis.labels]
        _, ax = plt.subplots(figsize=figsize or [14, 6])
        ax.bar(labels, self.parameters(sample)[-1])
        ax.grid()
        if title:
            plt.title(title)
        plt.xticks(rotation=90)
        plt.show()

    def plot_fidelities(self, title=False, figsize=None):
        _, ax = plt.subplots(figsize=figsize or [12, 6])
        for s in range(1, self.samples+1):
            ax.plot(self.steps(s), self.fidelities(s))
        if title:
            plt.title(title)
        plt.show()

    def plot_step_sizes(self, title=False, figsize=None):
        _, ax = plt.subplots(figsize=figsize or [12, 6])
        for s in range(1, self.samples+1):
            ax.plot(self.steps(s), self.step_sizes(s))
        if title:
            plt.title(title)
        plt.show()

    def _load_optimization_data(self):
        filepath = self._generate_filepath()
        if not os.path.isfile(filepath):
            return 0
        dfs = pd.read_csv(filepath,
                          index_col=False,
                          )
        dfs.dropna(axis=0, inplace=True)
        samples = dfs["sample"].max()
        self.samples += samples
        for sample in range(1, samples+1):
            df = dfs[dfs["sample"] == sample]
            self.optimizers.append(df.to_dict("list"))
        return self.optimizers

    def _generate_filepath(self, name="optimization_data", extension="csv", float_precision=4):
        conf_folder = ""
        for attr, value in self.config.items():
            if attr == 'name':
                continue
            a = value
            a = int(a) if np.isclose(int(a), a) else float(a)
            if isinstance(a, int):
                a_str = f"m{abs(a)}" if str(a)[0] == "-" else str(a)
                conf_folder += "_" + attr + "=" + a_str
            elif isinstance(a, float):
                a_str = f"m{abs(a):.{float_precision}f}" if str(a)[0] == "-" else f"{a:.{float_precision}f}"
                conf_folder += "_" + attr + "=" + a_str
            elif isinstance(a, bool):
                a_str = f"m{a}"
                conf_folder += "_" + attr + "=" + a_str
        conf_folder = conf_folder[1:]
        root_folder = f"{self.folder}/{self.config['name']}/{conf_folder}"
        directory = os.getcwd() + "/" + root_folder
        if not os.path.exists(directory):
            os.makedirs(directory)
        return root_folder + f"/{name}.{extension}"

    def _construct_data_dict(self, optimizer):
        data_dict = {
            "sample": [self.samples] * (optimizer.steps[-1]+1),
            "steps": optimizer.steps,
            "parameters": [list(p) for p in optimizer.parameters],
            "fidelities": optimizer.fidelities,
            "step_sizes": optimizer.step_sizes,
        }
        return data_dict

    def _is_optimizer_loaded(self):
        if not self.optimizers:
            raise ValueError("Must use add_optimizer before you can check the parameters.")

    def _find_sample(self, sample):
        for i, dic in enumerate(self.optimizers):
            if sample in dic["sample"]:
                return i
        return -1

