# Copyright (c) Meta Platforms, Inc. and affiliates.

import json
import logging
from functools import partial
from pathlib import Path

from cwm.data.dataset import Dataset
from cwm.text.datatypes import DictDatum, StrDatum

logger = logging.getLogger()


def _from_text_file_line(file_line: StrDatum, strict: bool = True) -> DictDatum:
    try:
        datum = DictDatum(json.loads(file_line.val), src=file_line.src)
    except Exception as e:
        if strict:
            raise e
        else:
            logger.warning(f"Failed json loading for {file_line}, returning None.")
            datum = DictDatum({}, src=file_line.src)
    return datum


@Dataset.register_src("jsonl")
def from_jsonl(
    path: Path | str,
    *,
    pattern: str = r".*chunk\.\d+.*\.jsonl(?:\.gz)?",
    encoding: str | None = None,
    decode_errors: str | None = None,
    strict_json_load: bool = True,
) -> Dataset[DictDatum]:
    return Dataset.from_text_file_lines(
        path,
        pattern=pattern,
        encoding=encoding,
        decode_errors=decode_errors,
    ).map(partial(_from_text_file_line, strict=strict_json_load))
