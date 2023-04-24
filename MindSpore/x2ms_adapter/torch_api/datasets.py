#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.
import bisect
import os
from distutils.version import LooseVersion
from typing import Optional, Callable, List

import numpy
import numpy as np
import PIL.Image

import mindspore
import mindspore.dataset as ds

from mindspore.communication.management import get_rank, get_group_size, context
from mindspore.dataset import MappableDataset, BatchDataset
from ..utils.adapter_check import external_input_check
from ..core.context import x2ms_context
from ..utils.util_api import np_to_tensor
from ..third_party_adapter.numpy_api import TensorNumpy

if LooseVersion(mindspore.__version__) < LooseVersion('1.8.0'):
    import mindspore.dataset.vision.py_transforms as vision
    import mindspore.dataset.transforms.py_transforms as ms_transforms
else:
    import mindspore.dataset.vision as vision
    import mindspore.dataset.transforms as ms_transforms


def _dataset_len(self):
    self.dataset_size = None
    return self.get_dataset_size()


@property
def mindspore_dataset(self):
    return self.children[0]


@property
def dataset_classes(self):
    child_dataset = self
    while True:
        if isinstance(child_dataset, MappableDataset):
            break
        if not child_dataset.children:
            return []
        child_dataset = child_dataset.children[0]

    if isinstance(child_dataset, ds.Cifar10Dataset):
        return __read_meta(os.path.join(child_dataset.dataset_dir, "batches.meta.txt"))
    elif isinstance(child_dataset, ds.Cifar100Dataset):
        return __read_meta(os.path.join(child_dataset.dataset_dir, "fine_label_names.txt"))
    elif isinstance(child_dataset, ds.ImageFolderDataset):
        return os.listdir(child_dataset.dataset_dir)
    else:
        raise NotImplementedError("Cannot get classes from this dataset now.")


def __read_meta(meta_file_path):
    external_input_check(meta_file_path)
    with open(meta_file_path, 'r') as meta_file:
        content = meta_file.read(1024 * 1024)
    return list(class_content for class_content in content.splitlines() if len(class_content.strip()) != 0)


@property
def get_transform(self):
    return self.operations


@get_transform.setter
def set_transform(self, transform_to_set):
    child_dataset = self
    while True:
        if isinstance(child_dataset, MappableDataset) or not child_dataset.children:
            break
        child_dataset = child_dataset.children[0]

    if isinstance(child_dataset, ds.ImageFolderDataset):
        self.operations = [start_transform, vision.Decode(), vision.ToPIL()]
    else:
        self.operations = [start_transform, vision.ToPIL()]
    if isinstance(transform_to_set, list):
        self.operations.extend(transform_to_set)
    if isinstance(transform_to_set, ms_transforms.Compose):
        self.operations.extend(transform_to_set.transforms)
    self.operations.append(_ensure_numpy_array)
    self.operations.append(end_transform)
    if LooseVersion(mindspore.__version__) < LooseVersion('1.8.0'):
        self.operations = ms_transforms.Compose.reduce(self.operations)


mindspore.dataset.Dataset.__len__ = _dataset_len
mindspore.dataset.Dataset.dataset = mindspore_dataset
mindspore.dataset.Dataset.classes = dataset_classes
mindspore.dataset.Dataset.transform = get_transform
mindspore.dataset.Dataset.transform = set_transform


class RawDatasetWrapper:
    def __init__(self, dataset):
        self.dataset = dataset
        x2ms_context.thread_start_transform()
        sample = dataset[0]
        self.column_records = self._flatten_record(sample)
        flattened_sample = self._flatten_data(sample, self.column_records)
        self.dataset_return_type_list = [type(i) for i in flattened_sample]
        self.column_names = [str(i) for i in range(len(flattened_sample))]
        x2ms_context.thread_end_transform()

    def __getitem__(self, item):
        x2ms_context.thread_start_transform()
        item = item.item()
        output = self.dataset[item]
        output = self._flatten_data(output, self.column_records)
        output = tuple(self._to_numpy_array(value) for value in output)
        x2ms_context.thread_end_transform()
        return output

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def _to_numpy_array(data) -> numpy.ndarray:
        if isinstance(data, tuple) and len(data) == 1:
            data = data[0]
        if isinstance(data, np.ndarray):
            if data.dtype == np.int64:
                return data.astype(np.int32)
            if data.dtype == np.float64:
                return data.astype(np.float32)
            return data
        elif isinstance(data, mindspore.Tensor):
            if data.dtype == mindspore.int64:
                return data.astype(mindspore.int32).asnumpy()
            if data.dtype == mindspore.float64:
                return data.astype(mindspore.float32).asnumpy()
            return data.asnumpy()
        else:
            result = np.asarray(data)
            if result.dtype == np.int64:
                return result.astype(np.int32)
            if result.dtype == np.float64:
                return result.astype(np.float32)
            if result.dtype == object:
                return np.array(0, np.float32)
            return result

    def _flatten_data(self, original_data, structure):
        flattened_value = []
        if isinstance(structure, dict):
            for key in original_data.keys():
                tmp_value = self._flatten_data(original_data[key], structure[key])
                flattened_value.extend(tmp_value)
            return flattened_value
        elif isinstance(structure, (tuple, list)):
            for idx, item in enumerate(original_data):
                tmp_value = self._flatten_data(item, structure[idx])
                flattened_value.extend(tmp_value)
            return flattened_value
        else:
            return [original_data]

    def _flatten_record(self, structure):
        if isinstance(structure, dict):
            structure_records = {}
            for key in structure.keys():
                tmp_record = self._flatten_record(structure[key])
                structure_records[key] = tmp_record
            return structure_records
        elif isinstance(structure, tuple):
            structure_records = []
            for item in structure:
                tmp_value = self._flatten_record(item)
                structure_records.append(tmp_value)
            return structure_records
        elif isinstance(structure, list):
            structure_records = []
            if all(isinstance(item, (int, float)) for item in structure):
                return 1
            if all(isinstance(item, (list, tuple)) for item in structure):
                item_shape_list = [len(item) for item in structure]
                if min(item_shape_list) == max(item_shape_list):
                    # current list is one sample, should not be flattened
                    return 1
            for item in structure:
                tmp_value = self._flatten_record(item)
                structure_records.append(tmp_value)
            return structure_records
        else:
            return 1


class CollateFnDatasetWrapper(mindspore.dataset.GeneratorDataset):
    def __init__(self, dataset: RawDatasetWrapper, batch_size=1, drop_last=False):
        if context.get_auto_parallel_context("parallel_mode") == context.ParallelMode.DATA_PARALLEL:
            super(CollateFnDatasetWrapper, self).__init__(dataset, dataset.column_names, shard_id=get_rank(),
                                                          num_shards=get_group_size(), shuffle=False)
        else:
            super(CollateFnDatasetWrapper, self).__init__(dataset, dataset.column_names, shuffle=False)
        self._column_records = dataset.column_records
        self._dataset_return_type_list = dataset.dataset_return_type_list
        self._batch_size = batch_size
        self._dataset = dataset
        self.drop_last = drop_last
        self.dataset_size = self.get_dataset_size()

    def __iter__(self):
        iterator = IteratorWrapper(self.create_tuple_iterator(output_numpy=True),
                                   self._column_records, self._dataset_return_type_list)
        cnt = 0
        res = []
        for data in iter(iterator):
            res.append(data)
            cnt += 1
            if cnt == self._batch_size:
                yield res
                res = []
                cnt = 0
        if not self.drop_last and res:
            yield res

    def __len__(self):
        if not self.dataset_size:
            self.get_dataset_size()
        dataset_size = self.dataset_size // self._batch_size
        if not self.drop_last and (self.dataset_size % self._batch_size) != 0:
            dataset_size += 1
        return dataset_size


class BatchDatasetWrapper(mindspore.dataset.BatchDataset):
    def __init__(self, dataset: RawDatasetWrapper, batch_size=1):
        self._column_records = dataset.column_records
        self.dataset_return_type_list = dataset.dataset_return_type_list
        parallel_mode = context.get_auto_parallel_context("parallel_mode")
        if parallel_mode == context.ParallelMode.DATA_PARALLEL:
            super().__init__(mindspore.dataset.GeneratorDataset(dataset, dataset.column_names, shard_id=get_rank(),
                                                                num_shards=get_group_size(), shuffle=False),
                             batch_size=batch_size)
        else:
            super().__init__(mindspore.dataset.GeneratorDataset(dataset, dataset.column_names, shuffle=False),
                             batch_size=batch_size)

    def __iter__(self):
        return IteratorWrapper(self.create_tuple_iterator(output_numpy=True),
                               self._column_records, column_type=None)


class IteratorWrapper:
    def __init__(self, iterator, column_records: list, column_type=None):
        self.iterator = iterator
        self.column_records = column_records
        self.column_type = column_type

    def __iter__(self):
        return self

    def __next__(self):
        flattened_batch = next(self.iterator)
        if self.column_type is not None:
            flattened_batch = [self._tensor_type_transform(data, target)
                               for data, target in zip(flattened_batch, self.column_type)]
        new_batch, _ = self._reconstruct_data(self.column_records, flattened_batch, 0)
        if isinstance(new_batch, list):
            return tuple(new_batch)
        return new_batch

    @staticmethod
    def _tensor_type_transform(data, target_type):
        if target_type == int or target_type == float:
            return data.item()
        elif target_type == list:
            return data.tolist()
        elif target_type == str:
            if LooseVersion(mindspore.__version__) >= LooseVersion('1.9.0'):
                return str(data)
            else:
                return str(mindspore.dataset.text.to_str(data))
        elif target_type == TensorNumpy:
            return np_to_tensor(data)
        else:
            return data

    def _reconstruct_data(self, structure, flattened_data, index):
        if isinstance(structure, dict):
            keys_iter = structure.keys()
            tmp_dict_data = []
            tmp_index = index
            for key in keys_iter:
                tmp_data, tmp_index = self._reconstruct_data(structure[key], flattened_data, tmp_index)
                tmp_dict_data.append(tmp_data)
            return dict(zip(keys_iter, tmp_dict_data)), tmp_index
        elif isinstance(structure, tuple):
            tmp_tuple_data = []
            tmp_index = index
            for item in structure:
                if isinstance(item, (dict, list, tuple)):
                    tmp_data, tmp_index = self._reconstruct_data(item, flattened_data, tmp_index)
                    tmp_tuple_data.append(tmp_data)
                else:
                    tmp_tuple_data.append(flattened_data[tmp_index])
                    tmp_index += 1
            return tuple(tmp_tuple_data), tmp_index
        elif isinstance(structure, list):
            tmp_tuple_data = []
            tmp_index = index
            for item in structure:
                if isinstance(item, (dict, list, tuple)):
                    tmp_data, tmp_index = self._reconstruct_data(item, flattened_data, tmp_index)
                    tmp_tuple_data.append(tmp_data)
                else:
                    tmp_tuple_data.append(flattened_data[tmp_index])
                    tmp_index += 1
            return tmp_tuple_data, tmp_index
        else:
            return flattened_data[index], index + 1


def _is_cifar100(check_dataset):
    dataset_child = check_dataset
    while True:
        if isinstance(dataset_child, MappableDataset):
            return isinstance(dataset_child, ds.Cifar100Dataset)
        if not dataset_child.children:
            break
        dataset_child = dataset_child.children[0]
    return False


def _add_sampler(dataset, sampler):
    if sampler and not isinstance(sampler, DistributedSampler):
        old_sampler = dataset.sampler
        dataset.use_sampler(sampler)
        dataset.add_sampler(old_sampler)


def data_loader(dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None, num_workers=1,
                collate_fn=None, pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None,
                multiprocessing_context=None, generator=None):
    """
    batch_sampler is partially implemented. Only batch_size in batch_sampler is mapped.
    """
    if batch_sampler is not None:
        sampler_batch_size = getattr(batch_sampler, 'batch_size', 1)
        batch_size = max(batch_size, sampler_batch_size)

    no_collate_fn_flag = collate_fn is None

    if isinstance(dataset, mindspore.dataset.Dataset):
        ms_dataset_flag = True
        dataset = dataset.__safe_deepcopy__({})
        if no_collate_fn_flag and _is_cifar100(dataset):
            dataset = dataset.batch(batch_size, per_batch_map=lambda col_1, col_2, col_3, batch_info: (col_1, col_2),
                                    input_columns=['image', 'fine_label', 'coarse_label'],
                                    output_columns=['image', 'label'])
        elif no_collate_fn_flag:
            dataset = dataset.batch(batch_size)
    else:
        ms_dataset_flag = False
        dataset = RawDatasetWrapper(dataset)
        if no_collate_fn_flag:
            dataset = BatchDatasetWrapper(dataset, batch_size=batch_size)
        else:
            dataset = CollateFnDatasetWrapper(dataset, batch_size=batch_size, drop_last=drop_last)

    child_dataset = dataset
    while True:
        if isinstance(child_dataset, MappableDataset):
            child_dataset.shuffle_flag = shuffle
            _add_sampler(child_dataset, sampler)
            child_dataset.num_parallel_workers = num_workers
        if isinstance(child_dataset, BatchDataset):
            child_dataset.drop_remainder = drop_last
        if not child_dataset.children:
            break
        child_dataset = child_dataset.children[0]
    return dataset, ms_dataset_flag


class DataLoader:
    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None, num_workers=1,
                 collate_fn=None, pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None,
                 multiprocessing_context=None, generator=None):
        self.x2ms_dataset = dataset
        num_workers = 1 if num_workers == 0 else num_workers
        if isinstance(sampler, ds.Sampler):
            sampler = ds.IterSampler(sampler=sampler)
        self.batch_sampler, self.ms_dataset = data_loader(dataset, batch_size, shuffle, sampler, batch_sampler,
                                                          num_workers, collate_fn, pin_memory, drop_last, timeout,
                                                          worker_init_fn, multiprocessing_context, generator)
        self.batch_size = batch_size
        self.collate_fn = collate_fn
        self.num_workers = num_workers
        self.drop_last = drop_last

    def __len__(self):
        return len(self.batch_sampler)

    def __iter__(self):
        if self.collate_fn is None:
            for batch in iter(self.batch_sampler):
                yield self._iter_object_to_tensor((batch,), current_depth=-1, max_depth=1)[0]
        elif self.ms_dataset:
            cnt = 0
            res = []
            for data in iter(self.batch_sampler):
                if isinstance(data, list):
                    res.append(tuple(data))
                else:
                    res.append(data)
                cnt += 1
                if cnt == self.batch_size:
                    yield self.collate_fn(res)
                    res = []
                    cnt = 0
            if not self.drop_last and res:
                yield self.collate_fn(res)
        else:
            for batch in iter(self.batch_sampler):
                yield self.collate_fn(batch)

    @property
    def dataset(self):
        return self.x2ms_dataset

    def _iter_object_to_tensor(self, iter_object, current_depth=0, max_depth=1):
        new_batch = []
        for item in iter_object:
            if isinstance(item, (tuple, list)) and current_depth < max_depth:
                new_batch.append(self._iter_object_to_tensor(item, current_depth=current_depth + 1))
            elif isinstance(item, dict):
                new_batch.append({k: (np_to_tensor(v) if isinstance(v, np.ndarray) else v) for k, v in item.items()})
            else:
                new_batch.append(np_to_tensor(item) if isinstance(item, np.ndarray) else item)
        return new_batch


class Subset:
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


def subset_dataset(dataset, indices):
    child_dataset = dataset
    while True:
        if isinstance(child_dataset, mindspore.dataset.MappableDataset):
            _add_sampler(child_dataset, mindspore.dataset.samplers.SubsetSampler(indices))
        if not child_dataset.children:
            break
        child_dataset = child_dataset.children[0]
    return dataset


class TensorDataset:
    def __init__(self, *tensors):
        self.tensors = tuple(self._type_convert(tensor) for tensor in tensors)

    def __getitem__(self, idx):
        return tuple(tensor[idx] for tensor in self.tensors)

    def __len__(self):
        return self.tensors[0].shape[0]

    @staticmethod
    def _type_convert(data):
        if data.dtype == mindspore.float64:
            return data.astype(mindspore.float32).asnumpy()
        else:
            return data.asnumpy()


def random_split(dataset, lengths, generator=None):
    if isinstance(dataset, mindspore.dataset.Dataset):
        return dataset.split(lengths, randomize=True)

    if sum(lengths) != len(dataset):
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    indices = np.random.permutation(np.arange(sum(lengths))).tolist()
    split_datasets = []
    offset = 0
    for length in lengths:
        split_datasets.append(Subset(dataset, indices[offset: offset + length]))
        offset += length

    return tuple(split_datasets)


def _ensure_numpy_array(data):
    if isinstance(data, np.ndarray):
        return data
    elif isinstance(data, PIL.Image.Image):
        return np.asarray(data)
    elif isinstance(data, mindspore.Tensor):
        return data.asnumpy()
    else:
        raise NotImplementedError(f'Unsupported data type {type(data)}')


def uint_to_int(data):
    if data.dtype == np.uint32:
        return data.astype(np.int32)
    return data


class ImageFolder:
    def __new__(cls, root, transform=None, target_transform=None, loader=None, is_valid_file=None):
        external_input_check(root)
        parallel_mode = context.get_auto_parallel_context("parallel_mode")
        if parallel_mode == context.ParallelMode.DATA_PARALLEL:
            ms_dataset = ds.ImageFolderDataset(dataset_dir=root, shard_id=get_rank(), num_shards=get_group_size())
        else:
            ms_dataset = ds.ImageFolderDataset(dataset_dir=root)
        target_transform_to_add = [uint_to_int]
        ms_dataset = _map_transform(ms_dataset, target_transform, target_transform_to_add, 'label')
        transform_to_add = [vision.Decode(), vision.ToPIL()]
        ms_dataset = _map_transform(ms_dataset, transform, transform_to_add, 'image')
        return ms_dataset


class CocoDetection:
    def __init__(self, root, annFile, transform=None, target_transform=None, transforms=None):
        raise NotImplementedError


def _folder_pil_loader(path):
    external_input_check(path)
    with open(path, 'rb') as img_file:
        img = PIL.Image.open(img_file)
        return img.convert('RGB')


def folder_default_loader(path):
    return _folder_pil_loader(path)


def cifar10(root, train=True, transform=None, target_transform=None, download=False):
    external_input_check(root)
    parallel_mode = context.get_auto_parallel_context("parallel_mode")
    if parallel_mode == context.ParallelMode.DATA_PARALLEL:
        ms_dataset = ds.Cifar10Dataset(dataset_dir=root, usage='train' if train else 'test',
                                       shard_id=get_rank(), num_shards=get_group_size())
    else:
        ms_dataset = ds.Cifar10Dataset(dataset_dir=root, usage='train' if train else 'test')
    target_transform_to_add = [uint_to_int]
    ms_dataset = _map_transform(ms_dataset, target_transform, target_transform_to_add, 'label')
    transform_to_add = [vision.ToPIL()]
    ms_dataset = _map_transform(ms_dataset, transform, transform_to_add, 'image')
    return ms_dataset


def cifar100(root, train=True, transform=None, target_transform=None, download=False):
    external_input_check(root)
    parallel_mode = context.get_auto_parallel_context("parallel_mode")
    if parallel_mode == context.ParallelMode.DATA_PARALLEL:
        ms_dataset = ds.Cifar100Dataset(dataset_dir=root, usage='train' if train else 'test',
                                        shard_id=get_rank(), num_shards=get_group_size())
    else:
        ms_dataset = ds.Cifar100Dataset(dataset_dir=root, usage='train' if train else 'test')
    target_transform_to_add = [uint_to_int]
    ms_dataset = _map_transform(ms_dataset, target_transform, target_transform_to_add, 'fine_label')
    transform_to_add = [vision.ToPIL()]
    ms_dataset = _map_transform(ms_dataset, transform, transform_to_add, 'image')
    return ms_dataset


def mnist(root, train=True, transform=None, target_transform=None, download=False):
    external_input_check(root)
    parallel_mode = context.get_auto_parallel_context("parallel_mode")
    if parallel_mode == context.ParallelMode.DATA_PARALLEL:
        ms_dataset = ds.MnistDataset(dataset_dir=root, usage='train' if train else 'test',
                                     shard_id=get_rank(), num_shards=get_group_size())
    else:
        ms_dataset = ds.MnistDataset(dataset_dir=root, usage='train' if train else 'test')

    if transform:
        transform_to_add = [lambda data: PIL.Image.fromarray(data.squeeze(-1), mode='L')]
        ms_dataset = _map_transform(ms_dataset, transform, transform_to_add, 'image')

    return ms_dataset


def qmnist(root, train=True, transform=None, target_transform=None, download=False):
    external_input_check(root)
    parallel_mode = context.get_auto_parallel_context("parallel_mode")
    if parallel_mode == context.ParallelMode.DATA_PARALLEL:
        ms_dataset = ds.QMnistDataset(dataset_dir=root, usage='train' if train else 'test',
                                      shard_id=get_rank(), num_shards=get_group_size())
    else:
        ms_dataset = ds.QMnistDataset(dataset_dir=root, usage='train' if train else 'test')

    if transform:
        transform_to_add = [lambda data: PIL.Image.fromarray(data.squeeze(-1), mode='L')]
        ms_dataset = _map_transform(ms_dataset, transform, transform_to_add, 'image')

    return ms_dataset


def kmnist(root, train=True, transform=None, target_transform=None, download=False):
    external_input_check(root)
    parallel_mode = context.get_auto_parallel_context("parallel_mode")
    if parallel_mode == context.ParallelMode.DATA_PARALLEL:
        ms_dataset = ds.KMnistDataset(dataset_dir=root, usage='train' if train else 'test',
                                      shard_id=get_rank(), num_shards=get_group_size())
    else:
        ms_dataset = ds.KMnistDataset(dataset_dir=root, usage='train' if train else 'test')

    if transform:
        transform_to_add = [lambda data: PIL.Image.fromarray(data.squeeze(-1), mode='L')]
        ms_dataset = _map_transform(ms_dataset, transform, transform_to_add, 'image')

    return ms_dataset


def fashion_mnist(root, train=True, transform=None, target_transform=None, download=False):
    external_input_check(root)
    parallel_mode = context.get_auto_parallel_context("parallel_mode")
    if parallel_mode == context.ParallelMode.DATA_PARALLEL:
        ms_dataset = ds.FashionMnistDataset(dataset_dir=root, usage='train' if train else 'test',
                                            shard_id=get_rank(), num_shards=get_group_size())
    else:
        ms_dataset = ds.FashionMnistDataset(dataset_dir=root, usage='train' if train else 'test')

    if transform:
        transform_to_add = [lambda data: PIL.Image.fromarray(data.squeeze(-1), mode='L')]
        ms_dataset = _map_transform(ms_dataset, transform, transform_to_add, 'image')

    return ms_dataset


def start_transform(*data):
    x2ms_context.thread_start_transform()
    return data[0] if len(data) == 1 else data


def end_transform(*data):
    x2ms_context.thread_end_transform()
    return data[0] if len(data) == 1 else data


def _map_transform(ms_dataset, transform, transform_to_add, input_columns):
    if transform:
        if isinstance(transform, list):
            transform_to_add.extend(transform)
        if isinstance(transform, ms_transforms.Compose):
            transform_to_add.extend(transform.transforms)
    transform_to_add.append(_ensure_numpy_array)
    transform_to_add = [start_transform, *transform_to_add, end_transform]
    ms_dataset = ms_dataset.map(operations=transform_to_add, input_columns=input_columns)
    return ms_dataset


class Sampler(ds.Sampler):
    def __iter__(self):
        pass


class DistributedSampler(ds.DistributedSampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        parallel_mode = context.get_auto_parallel_context("parallel_mode")
        if parallel_mode == context.ParallelMode.DATA_PARALLEL:
            super().__init__(num_shards=get_group_size(), shard_id=get_rank(), shuffle=shuffle)
        else:
            super().__init__(num_shards=1, shard_id=0, shuffle=shuffle)


class RandomSampler(mindspore.dataset.RandomSampler):
    def __init__(self, data_source, replacement=False, num_samples=None, generator=None):
        super().__init__(replacement=replacement, num_samples=num_samples)


class SequentialSampler(mindspore.dataset.SequentialSampler):
    def __init__(self, data_source):
        super().__init__()


class SubsetRandomSampler(mindspore.dataset.SubsetRandomSampler):
    def __init__(self, indices, generator=None):
        super().__init__(indices)


class VisionDataset:
    _repr_indent = 4

    def __init__(
            self,
            root: str,
            transforms_function: Optional[Callable] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ) -> None:
        if isinstance(root, (str, bytes)):
            root = os.path.expanduser(root)
        self.root = root

        has_transforms = transforms_function is not None
        has_separate_transform = transform is not None or target_transform is not None
        if has_transforms and has_separate_transform:
            raise ValueError("Only transform_function or transform/target_transform can "
                             "be passed as argument")

        self.transform = transform
        self.target_transform = target_transform

        if has_separate_transform:
            transforms_function = StandardTransform(transform, target_transform)
        self.transforms = transforms_function

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index: int):
        raise NotImplementedError

    def __repr__(self):
        title = f"Dataset {self.__class__.__name__}"
        body = ["Number of datapoints: {}".format(self.__len__())]

        if self.root is not None:
            body.append(f"Root location: {self.root}")
        body.extend(self.extra_repr().splitlines())

        if getattr(self, "transforms"):
            body.append(repr(self.transforms))
        lines = [title] + [" " * self._repr_indent + line for line in body]
        return '\n'.join(lines)

    @staticmethod
    def extra_repr():
        return ""

    @staticmethod
    def _format_transform_repr(transform: Callable, head: str) -> List[str]:
        lines = transform.__repr__().splitlines()
        title = f"{head}{lines[0]}"
        body = ["{}{}".format(" " * len(head), line) for line in lines[1:]]
        return [title] + body


class StandardTransform:
    def __init__(self, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform

    def __call__(self, data, target):
        if self.transform is not None:
            data = self.transform(data)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return data, target

    def __repr__(self) -> str:
        body = [self.__class__.__name__]
        if self.transform is not None:
            body.extend(self._format_transform_repr(self.transform, "Transform: "))
        if self.target_transform is not None:
            body.extend(self._format_transform_repr(self.target_transform, "Target transform: "))

        return '\n'.join(body)

    @staticmethod
    def _format_transform_repr(transform: Callable, head: str) -> List[str]:
        lines = transform.__repr__().splitlines()
        title = f"{head}{lines[0]}"
        body = ["{}{}".format(" " * len(head), line) for line in lines[1:]]
        return [title] + body


class ConcatDataset:
    def __init__(self, datasets):
        if len(datasets) <= 0:
            raise ValueError('Input datasets should not be empty.')
        self.datasets = list(datasets)
        for one_dataset in self.datasets:
            if not (hasattr(one_dataset, '__len__') and hasattr(one_dataset, '__getitem__')):
                raise TypeError("The datasets should have implemented '__len__' and '__getitem__' "
                                "method to be mindspore dataset")
        self.cumulative_index = self.index_generator(self.datasets)

    def __len__(self):
        return self.cumulative_index[-1]

    def __getitem__(self, item):
        if abs(item) > len(self):
            raise ValueError("Index out of dataset length range.")
        if item < 0:
            item += len(self)

        dataset_index = bisect.bisect_right(self.cumulative_index, item) - 1
        sample_index = item - self.cumulative_index[dataset_index]

        return self.datasets[dataset_index][sample_index]

    @staticmethod
    def index_generator(dataset_list):
        index_list = [0]
        for i, one_dataset in enumerate(dataset_list):
            index_list.append(len(one_dataset) + index_list[i])
        return index_list
