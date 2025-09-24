# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Defines all error types globally"""


class BackendInitError(Exception):
    pass


class FormatError(Exception):
    pass


class NoSuchToolError(Exception):
    pass
