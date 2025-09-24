# Copyright (c) Meta Platforms, Inc. and affiliates.

import logging
import re
from pathlib import Path
from typing import Any, BinaryIO

from upath import UPath

from cwm.data.dataset import Dataset
from cwm.data.ops.chain import ChainPolicy
from cwm.text.datatypes import BaseTextDatum, StrDatum

logger = logging.getLogger()


class OneFileLines(Dataset[StrDatum]):
    def __init__(
        self,
        path: Path | str,
        *,
        encoding: str | None = None,
        decode_errors: str | None = None,
    ) -> None:
        self.path = UPath(path)
        self.file: BinaryIO | None = None
        self.encoding = encoding or "utf-8"
        self.decode_errors = decode_errors or "strict"

        if not self.path.is_file():
            raise FileNotFoundError(f"File not found at {path}")

        self.reset()

    def _close(self) -> None:
        if self.file is not None:
            self.file.close()
            self.file = None

    def reset(self) -> None:
        logger.warning(f"Resetting iterator OneFileLines({self.path})")
        self._close()
        self.pos = 0
        self.line_no = 0

    def __next__(self) -> StrDatum:
        if self.file is None:
            compression = "gzip" if self.path.suffix == ".gz" else None
            self.file = self.path.open("rb", compression=compression)
            self.file.seek(self.pos)

        if line_bytes := self.file.readline():
            prev_pos = self.pos
            self.pos += len(line_bytes)
            line_no, self.line_no = self.line_no, self.line_no + 1
            return StrDatum(
                line_bytes.decode(self.encoding, errors=self.decode_errors),
                src=BaseTextDatum.Source(path=self.path, line_no=line_no, pos=prev_pos),
            )
        else:
            raise StopIteration

    def state_dict(self) -> dict[str, Any]:
        return {"line_no": self.line_no, "pos": self.pos}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.line_no = state_dict["line_no"]
        self.pos = state_dict["pos"]
        self._close()


@Dataset.register_src("text_file_lines")
def from_text_file_lines(
    path: Path | str,
    *,
    pattern: str | None = None,
    encoding: str | None = None,
    decode_errors: str | None = None,
) -> Dataset[StrDatum]:
    upath = UPath(path)
    files_policy = ChainPolicy()

    if upath.is_file():
        return OneFileLines(upath, encoding=encoding, decode_errors=decode_errors)
    if upath.is_dir():
        if pattern:
            file_paths = sorted(
                [p for p in upath.iterdir() if re.fullmatch(pattern, str(p))]
            )
        else:
            file_paths = sorted(upath.glob("*"))
        dataset_impls = [
            OneFileLines(file_path, encoding=encoding, decode_errors=decode_errors)
            for file_path in file_paths
        ]
        return Dataset.mix(dataset_impls, files_policy)
    raise FileNotFoundError(str(upath))
