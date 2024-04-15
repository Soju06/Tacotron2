from abc import abstractmethod
import importlib.util
from os import PathLike
from pathlib import Path


class TextTokenizer:
    symbols: list[str]
    symbol_map: dict[str, int]
    pad: int
    eos: int

    def __init__(self, symbols: list[str]):
        assert len(symbols) == len(set(symbols)), "Symbols must be unique"

        ex_symbols = {"<eos>", "<pad>"}
        ex_symbols.update(symbols)

        self.symbols = list(ex_symbols)
        self.pad = 1
        self.eos = 0
        self.symbol_map = {s: i for i, s in enumerate(symbols)}

    def __len__(self):
        return len(self.symbols)

    def __getitem__(self, key: str | int) -> int | str:
        if isinstance(key, str):
            return self.symbol_map[key]
        return self.symbols[key]

    @abstractmethod
    def clean(self, text: str) -> list[str]:
        raise NotImplementedError

    def tokenize(self, text: str) -> list[int]:
        return [self.symbol_map.get(s, self.pad) for s in self.clean(text)] + [self.eos]

    @staticmethod
    def load(
        path: str | PathLike, name: str | None = None, *args, **kwargs
    ) -> "TextTokenizer":
        path = Path(path)

        if not path.exists():
            path = path.with_suffix(".py")
            path = Path(__file__).parent / path

        module_name = f"{'.'.join(__name__.split('.')[:-1])}.{path.stem}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        assert spec is not None, "Spec is None"
        module = importlib.util.module_from_spec(spec)  # type: ignore
        assert spec.loader is not None, "Loader is None"
        spec.loader.exec_module(module)

        for value_name, obj in module.__dict__.items():
            if (
                isinstance(obj, type)
                and issubclass(obj, TextTokenizer)
                and (not name or value_name == name)
                and obj is not TextTokenizer
            ):
                return obj(*args, **kwargs)

        raise ValueError(f"Could not find TextTokenizer in {path}")
