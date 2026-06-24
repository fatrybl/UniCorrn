import random


class RandomTaskDataLoader:
    def __init__(self, loader_dict):
        """
        Params
        ------
        loader_dict (dict): A dictionary mapping string keys to DataLoaders.
                            e.g., {'task1': loader1, 'task2': loader2, ...}
        """
        self.loaders = loader_dict
        self.keys = list(loader_dict.keys())
        self.iterators = {key: iter(loader) for key, loader in loader_dict.items()}
        self.exhausted_keys = set()

    def __iter__(self):
        return self

    def __len__(self):
        return sum(len(loader) for loader in self.loaders.values())

    def __next__(self):
        if len(self.exhausted_keys) == len(self.keys):
            # start a new epoch when loaders are exhausted
            self.iterators = {key: iter(self.loaders[key]) for key in self.keys}
            self.exhausted_keys.clear()

        while True:
            available_keys = [k for k in self.keys if k not in self.exhausted_keys]
            if len(available_keys) == 0:
                raise StopIteration

            key = random.choice(available_keys)
            try:
                batch = next(self.iterators[key])
                return key, batch
            except StopIteration:
                self.exhausted_keys.add(key)


class MultiTaskBatchDataLoader:
    def __init__(self, loader_dict, **kwargs):
        """
        Params
        ------
        loader_dict (dict): A dictionary mapping string keys to DataLoaders.
                            e.g., {'task1': loader1, 'task2': loader2, ...}
        """
        # Validate that all loaders have the same length
        lengths = [len(loader) for loader in loader_dict.values()]
        # assert all(l == lengths[0] for l in lengths), \
        #     f"All loaders must have the same length, but got lengths: {lengths}"

        self.num_batches = max(lengths)
        self.loaders = loader_dict
        self.keys = list(loader_dict.keys())
        self.loader_wts = {k: kwargs[f'{k}_wt'] for k in self.keys}
        self._create_iterators()

    def _create_iterators(self):
        self.iterators = {key: iter(loader) for key, loader in self.loaders.items()}
        self.batches_seen = 0

    def __iter__(self):
        self._create_iterators()  # Reset for a new epoch
        return self

    def __len__(self):
        return self.num_batches

    def __next__(self):
        if self.batches_seen >= self.num_batches:
            raise StopIteration

        batch_dict = {}
        for key in self.keys:
            try:
                batch = next(self.iterators[key])
                batch_dict[key] = batch
            except StopIteration:
                if self.loader_wts[key] != 0.0:
                    raise StopIteration

        self.batches_seen += 1
        return batch_dict
