#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.

import os
import platform

from .util_api import logger


MAX_WINDOWS_PATH_LENGTH = 200
MAX_LINUX_PATH_LENGTH = 4000
MAX_LINUX_FILE_NAME_LENGTH = 200
DATA_FILE_SIZE = 10 * 1024 ** 3


def __check_path_length_valid(path):
    path = os.path.realpath(path)
    if platform.system().lower() == 'windows':
        return len(path) <= MAX_WINDOWS_PATH_LENGTH
    else:
        return len(path) <= MAX_LINUX_PATH_LENGTH and len(os.path.basename(path)) <= MAX_LINUX_FILE_NAME_LENGTH


def __check_path_owner_consistent(path):
    if platform.system().lower() == 'windows':
        return True

    file_owner_uid = os.stat(path).st_uid
    return file_owner_uid == os.getuid()


def __external_path_check(path, state='Input'):
    real_input_path = os.path.realpath(path)
    if not os.path.exists(real_input_path):
        raise FileNotFoundError(f'{state} {real_input_path} does not exist!')

    if os.path.islink(os.path.abspath(path)):
        logger.error(f"{state} {real_input_path} doesn't support soft link.")

    if not os.access(real_input_path, os.R_OK):
        raise ValueError(f'{state} {real_input_path} is not readable!')

    if not __check_path_owner_consistent(real_input_path):
        logger.warning(
            f'The {state.lower()} path may be insecure because it does not belong to you.')

    if not __check_path_length_valid(real_input_path):
        logger.error(f'The real path or file name of {state.lower()} is too long.')

    if os.path.getsize(real_input_path) > DATA_FILE_SIZE:
        logger.warning("File is too large which exceeds 10G, take care of your machine memory when load it.")


def external_input_check(path):
    __external_path_check(path)


def external_output_check(path):
    __external_path_check(path, state='Output')
    real_output_path = os.path.realpath(path)
    if not os.access(real_output_path, os.W_OK):
        raise ValueError(f'Output {real_output_path} is not writeable!')
