import re
from io import StringIO

from jamo import hangul_to_jamo

from .tokenizer import TextTokenizer

jamo_to_pron = str.maketrans(
    {
        "ㄱ": "기역",
        "ㄲ": "쌍기역",
        "ㄴ": "니은",
        "ㄷ": "디귿",
        "ㄸ": "쌍디귿",
        "ㄹ": "리을",
        "ㅁ": "미음",
        "ㅂ": "비읍",
        "ㅃ": "쌍비읍",
        "ㅅ": "시옷",
        "ㅆ": "쌍시옷",
        "ㅇ": "이응",
        "ㅈ": "지읒",
        "ㅉ": "쌍지읒",
        "ㅊ": "치읓",
        "ㅋ": "키읔",
        "ㅌ": "티읕",
        "ㅍ": "피읖",
        "ㅎ": "히읗",
        "ㅏ": "아",
        "ㅐ": "애",
        "ㅑ": "야",
        "ㅒ": "얘",
        "ㅓ": "어",
        "ㅔ": "에",
        "ㅕ": "여",
        "ㅖ": "예",
        "ㅗ": "오",
        "ㅘ": "와",
        "ㅙ": "왜",
        "ㅚ": "외",
        "ㅛ": "요",
        "ㅜ": "우",
        "ㅝ": "워",
        "ㅞ": "웨",
        "ㅟ": "위",
        "ㅠ": "유",
        "ㅡ": "으",
        "ㅢ": "의",
        "ㅣ": "이",
        "ㄳ": "기역시옷",
        "ㄵ": "니은지읒",
        "ㄶ": "니은히읗",
        "ㄺ": "리을기역",
        "ㄻ": "리을미음",
        "ㄼ": "리을비읍",
        "ㄽ": "리을시옷",
        "ㄾ": "리을티읕",
        "ㄿ": "리을피읖",
        "ㅀ": "리을히읗",
        "ㅄ": "비읍시옷",
    }
)

prefix_special_symbol = {
    "-": "마이너스 ",
    "+": "플러스 ",
}

suffix_special_symbol = {
    "%p": "퍼센트 포인트",
    "%": "퍼센트",
    "nm": "나노미터",
    "mm": "밀리미터",
    "cm": "센치미터",
    "m": "미터",
    "km": "킬로미터",
    "mg": "밀리그램",
    "g": "그램",
    "kg": "킬로그람",
    "t": "톤",
    "ml": "밀리리터",
    "L": "리터",
    "Hz": "헤르츠",
    "GHz": "기가헤르츠",
    "MHz": "메가헤르츠",
    "kHz": "킬로헤르츠",
    "dB": "데시벨",
    "V": "볼트",
    "A": "암페어",
    "W": "와트",
    "F": "패럿",
    "Ω": "옴",
    "Ωm": "옴 미터",
    "Sv": "시버트",
    "cal": "칼로리",
    "kcal": "킬로칼로리",
    "$": "달러",
    "￦": "원",
    "℃": "도 씨",
    "℉": "화씨",
}


def split_numbers(text):
    """
    문자와 숫자를 분리합니다.

    >>> split_numbers("삼성전자는 8만8,400.65원 SK하이닉스는 10만9,200.35원이다.")
        [(0, '삼성전자는 '),
        (1, '8'),
        (0, '만'),
        (1, '8,400.65'),
        (0, '원 SK하이닉스는 '),
        (1, '10'),
        (0, '만'),
        (1, '9,200.35'),
        (0, '원이다.')]

    >>> split_numbers("삼성전자는 저번 분기보다 4%p 증가하였고, 하이닉스는 -1%p 감소하였다.")
        [(0, '삼성전자는 저번 분기보다 '),
        (1, '4'),
        (3, '%p'),
        (0, ' 증가하였고, 하이닉스는 '),
        (2, '-'),
        (1, '1'),
        (3, '%p'),
        (0, ' 감소하였다.')]
    """
    # 0: 문자
    # 1: 숫자
    # 2: 접두사
    # 3: 접미사
    # 콤마나 점이 포함된 숫자 패턴
    pattern = r"\d[\d,\.]*\d|\d"
    nums = re.findall(pattern, text)
    # 숫자의 시작 위치를 찾습니다
    num_indices = [m.start() for m in re.finditer(pattern, text)]
    result = []
    start = 0

    for i, num in enumerate(nums):
        # 숫자가 시작되는 위치 전까지의 문자열
        end = num_indices[i]
        # 접두사가 있는지 확인합니다
        prefix = next(
            (p for p in prefix_special_symbol if text.endswith(p, start, end)), None
        )

        if prefix:  # 접두사가 있는 경우 접두사를 제거하여 문자를 저장합니다
            result.append((0, text[start : end - len(prefix)]))
            result.append((2, prefix))
        else:  # 접두사가 없는 경우
            result.append((0, text[start:end]))

        result.append((1, num))  # 숫자

        # 접미사가 있는지 확인합니다
        suffix = next(
            (
                p
                for p in suffix_special_symbol
                if text.startswith(p, num_indices[i] + len(num))
            ),
            None,
        )
        start = num_indices[i] + len(num)

        if suffix:  # 접미사가 있는 경우 접미사를 제거하여 문자를 저장합니다
            result.append((3, suffix))
            start += len(suffix)
    # 마지막 숫자 이후의 문자열
    if start < len(text):
        result.append((0, text[start:]))

    return result


digit_to_text = {
    "0": "영",
    "1": "일",
    "2": "이",
    "3": "삼",
    "4": "사",
    "5": "오",
    "6": "육",
    "7": "칠",
    "8": "팔",
    "9": "구",
    ".": "쩜 ",
}

base_units = ["", "십", "백", "천"]
units = ["", "십", "백", "천 "] + [
    f"{unit}{add_unit} "
    for add_unit in ["만", "억", "조", "경", "해", "자", "양", "구", "간", "정", "재", "극"]
    for unit in base_units
]
quiet_unit = 4


def digit2text(text: str):
    """
    숫자를 읽습니다.

    >>> digit2text('3,458,912.134')
        '삼백사십오만 팔천 구백십이쩜 일삼사'

    """
    has_decimal = "." in text
    result = []
    unit = 0

    for i, char in enumerate(reversed(text)):
        if char == ",":
            continue

        if char == ".":
            unit = 0
            has_decimal = False
            if i:  # 소수점 자리면 '쩜'을 읽음.
                result.insert(0, digit_to_text[char])
            continue
        # 소숫점인 경우 단수로 읽음.
        if has_decimal:
            result.insert(0, digit_to_text[char])
            continue
        # 단위가 바뀌는 경우 단위를 읽음.
        if unit % quiet_unit == 0:
            result.insert(0, units[unit])
        # 0은 읽지 않음.
        if char == "0":
            unit += 1
            continue
        # 일 ~ 천 단위의 일은 읽지 않음.
        digit = "" if char == "1" and unit % quiet_unit else digit_to_text[char]
        # 만 ~ 극 단위는 읽지 않음.
        if unit % quiet_unit != 0:
            # 일 ~ 천 단위에서 읽음.
            digit += units[unit % quiet_unit]

        result.insert(0, digit)
        unit += 1

    return "".join(result)


eng_to_kor = str.maketrans(
    {
        "a": "에이",
        "b": "비",
        "c": "씨",
        "d": "디",
        "e": "이",
        "f": "에프",
        "g": "지",
        "h": "에이치",
        "i": "아이",
        "j": "제이",
        "k": "케이",
        "l": "엘",
        "m": "엠",
        "n": "엔",
        "o": "오",
        "p": "피",
        "q": "큐",
        "r": "알",
        "s": "에스",
        "t": "티",
        "u": "유",
        "v": "브이",
        "w": "더블유",
        "x": "엑스",
        "y": "와이",
        "z": "지",
    }
)


def eng2kor(text):
    """
    영어를 한글로 읽습니다.

    >>> eng2kor('SK하이닉스')
        '에스케이 하이닉스'

    >>> eng2kor('Hello, World!')
        '에이치이엘엘오, 더블유오알엘디!'
    """
    return text.lower().translate(eng_to_kor)


def jamo2pron(text):
    """
    자모를 발음으로 읽습니다.

    >>> jamo2pron('ㄱ. 가')
        '기역. 가'

    >>> jamo2pron('ㄳ')
        '기역시옷'
    """
    return text.translate(jamo_to_pron)


class KoreanTokenizer(TextTokenizer):
    def __init__(self):
        super().__init__(
            [
                *"!'(),-.:;?/",
                *[chr(_) for _ in range(0x1100, 0x1113)],
                *[chr(_) for _ in range(0x1161, 0x1176)],
                *[chr(_) for _ in range(0x11A8, 0x11C3)],
            ]
        )

    def raw_clean(self, text: str) -> str:
        result = StringIO()

        for code, token in split_numbers(text):
            if code == 0:
                result.write(token)
            elif code == 1:
                result.write(digit2text(token))
            elif code == 2:
                result.write(prefix_special_symbol[token])
            elif code == 3:
                result.write(suffix_special_symbol[token])

        return jamo2pron(eng2kor(result.getvalue()))

    def clean(self, text: str) -> list[str]:
        return list(hangul_to_jamo(self.raw_clean(text)))
