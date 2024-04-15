# pip install ttstokenizer
# https://github.com/neuml/ttstokenizer

# JUST A SIMPLE IMPLEMENTATION :D

from ttstokenizer import TTSTokenizer
from .tokenizer import TextTokenizer


class EnglishTokenizer(TextTokenizer):
    def __init__(self):
        super().__init__(
            [
                *"!'(),-.:;?/",
                *list("abcdefghijklmnopqrstuvwxyz"),
                *["<pad>", "<unk>", "<s>", "</s>"],
                *[
                    "AA0",
                    "AA1",
                    "AA2",
                    "AE0",
                    "AE1",
                    "AE2",
                    "AH0",
                    "AH1",
                    "AH2",
                    "AO0",
                    "AO1",
                    "AO2",
                    "AW0",
                    "AW1",
                    "AW2",
                    "AY0",
                    "AY1",
                    "AY2",
                    "B",
                    "CH",
                    "D",
                    "DH",
                    "EH0",
                    "EH1",
                    "EH2",
                    "ER0",
                    "ER1",
                    "ER2",
                    "EY0",
                    "EY1",
                    "EY2",
                    "F",
                    "G",
                    "HH",
                    "IH0",
                    "IH1",
                    "IH2",
                    "IY0",
                    "IY1",
                    "IY2",
                    "JH",
                    "K",
                    "L",
                    "M",
                    "N",
                    "NG",
                    "OW0",
                    "OW1",
                    "OW2",
                    "OY0",
                    "OY1",
                    "OY2",
                    "P",
                    "R",
                    "S",
                    "SH",
                    "T",
                    "TH",
                    "UH0",
                    "UH1",
                    "UH2",
                    "UW",
                    "UW0",
                    "UW1",
                    "UW2",
                    "V",
                    "W",
                    "Y",
                    "Z",
                    "ZH",
                ],
            ]
        )
        self.tokenizer = TTSTokenizer()

    def clean(self, text: str) -> list[str]:
        return self.tokenizer(text)  # type: ignore
