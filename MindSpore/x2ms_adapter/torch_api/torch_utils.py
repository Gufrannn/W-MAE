#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
from ..core.context import x2ms_context
from ..utils.util_api import logger


def clip_grad_norm(parameters, max_norm, norm_type=2.0, error_if_nonfinite=False):
    x2ms_context.clip_grad_norm = max_norm
    return 0.0


def checkpoint(function, *args, **kwargs):
    return function(*args, **kwargs)


def pair(data):
    if isinstance(data, (tuple, list)):
        return data
    return data, data


def get_num_threads():
    return 1


def download_url_to_file(url, dst, hash_prefix=None, progress=True):
    logger.warning(f"MindSpore does not supported download {url} to {dst}, please download it by yourself.")


class SummaryWriter(object):
    def __init__(self, logdir=None, log_dir=None, comment='', purge_step=None, max_queue=10,
                 flush_secs=120, filename_suffix='', **kwargs):
        pass

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None):
        pass

    def add_scalars(self, main_tag, tag_scalar_dict, global_step=None, walltime=None):
        pass

    def add_graph(self, model, input_to_model=None, verbose=False):
        pass

    def add_histogram(self, tag, values, global_step=None, bins='tensorflow', walltime=None, max_bins=None):
        pass

    def add_image(self, tag, img_tensor, global_step=None, walltime=None, dataformats='CHW'):
        pass

    def add_images(self, tag, img_tensor, global_step=None, walltime=None, dataformats='NCHW'):
        pass

    def add_hparams(self, hparam_dict, metric_dict, hparam_domain_discrete=None, run_name=None):
        pass

    def close(self):
        pass


class SingleProcessDataLoaderIter:
    class DatasetFetcher:
        def __init__(self, loader):
            self.loader = iter(loader)

        def fetch(self, index):
            return next(self.loader)

    def __init__(self, loader):
        self._num_yielded = 0
        self.loader = loader
        self._pin_memory = False
        self._index = 0
        self._dataset_fetcher = SingleProcessDataLoaderIter.DatasetFetcher(self.loader)

    def __iter__(self):
        return self

    def __next__(self):
        data = self._next_data()
        self._num_yielded += 1
        return data

    def __len__(self):
        return len(self.loader)

    def _next_data(self):
        index = self._next_index()
        data = self._dataset_fetcher.fetch(index)
        return data

    def _next_index(self):
        start_index = self._index
        self._index += self.loader.batch_size
        return list(range(start_index, self._index))


class Stream:
    def __init__(self):
        pass
