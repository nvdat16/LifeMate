"""
Perception Agent: Audio transcription and generic Vietnamese text normalization.
Zero hardcoding: Uses mathematical numeral grammars and phonetic rules.
"""

import re
import unicodedata
from typing import Optional
from ...ports.asr_port import AbstractASREngine
from ...domain.entities import Sample

# Generic numeral mappings
VIETNAMESE_UNITS = {
    "không": 0, "một": 1, "mốt": 1, "hai": 2, "ba": 3, "bốn": 4, "tư": 4,
    "năm": 5, "lăm": 5, "sáu": 6, "bảy": 7, "tám": 8, "chín": 9
}

VIETNAMESE_TENS = {
    "mười": 10, "hai mươi": 20, "ba mươi": 30, "bốn mươi": 40, "năm mươi": 50,
    "sáu mươi": 60, "bảy mươi": 70, "tám mươi": 80, "chín mươi": 90
}

# Regex parsing compound Vietnamese numerals (10-99)
RE_COMPOUND_NUMBERS = re.compile(
    r"\b(mười|hai mươi|ba mươi|bốn mươi|năm mươi|sáu mươi|bảy mươi|tám mươi|chín mươi)(?:\s+(một|mốt|hai|ba|bốn|tư|năm|lăm|sáu|bảy|tám|chín))?\b",
    re.IGNORECASE
)

# Regex parsing single digits followed by measurement units or temporal words
RE_SINGLE_UNIT_NUMBERS = re.compile(
    r"\b(không|một|hai|ba|bốn|tư|năm|sáu|bảy|tám|chín)\s+(giờ|phút|giây|độ|tiếng|ngày|tháng|năm|cái|chiếc|đoạn|người)\b",
    re.IGNORECASE
)

# Currency shorthand regex (k/nghìn/ngàn, triệu/củ, tỷ)
RE_CURRENCY_THOUSANDS = re.compile(r"(\d+)\s*(k|nghìn|ngàn)\b", re.IGNORECASE)
RE_CURRENCY_MILLIONS = re.compile(r"(\d+)\s*(tr|triệu|củ)\b", re.IGNORECASE)
RE_CURRENCY_BILLIONS = re.compile(r"(\d+)\s*(tỷ)\b", re.IGNORECASE)

class PerceptionAgent:
    """Handles audio perception and generic linguistic normalization."""

    def __init__(self, asr_engine: AbstractASREngine, force_asr: bool = False):
        self.asr_engine = asr_engine
        self.force_asr = force_asr

    def perceive(self, sample: Sample) -> str:
        """
        Extracts transcript for sample. Uses ground truth audio_text if present (unless force_asr=True),
        otherwise delegates to the ASR engine.
        """
        if sample.audio_text and not self.force_asr:
            raw_text = sample.audio_text
        else:
            raw_text = self.asr_engine.transcribe(sample.audio)

        return self.normalize(raw_text, sample.context.current_time)

    def normalize(self, text: str, current_time_iso: str = "") -> str:
        """
        Generic normalization:
        - Unicode NFC normalization.
        - Compound and single digit numeral parsing.
        - Standard currency multipliers.
        """
        if not text:
            return ""

        # 1. Unicode NFC canonical normalization
        s = " ".join(unicodedata.normalize("NFC", text).split())

        # 2. Mathematical compound numeral conversion (10-99)
        def _repl_compound(m):
            ten = VIETNAMESE_TENS[m.group(1).lower()]
            unit = VIETNAMESE_UNITS[m.group(2).lower()] if m.group(2) else 0
            return str(ten + unit)

        s = RE_COMPOUND_NUMBERS.sub(_repl_compound, s)

        # 3. Single digit numeral with units conversion
        def _repl_single(m):
            digit = VIETNAMESE_UNITS[m.group(1).lower()]
            return f"{digit} {m.group(2)}"

        s = RE_SINGLE_UNIT_NUMBERS.sub(_repl_single, s)

        # 4. Currency multiplier shorthand
        s = RE_CURRENCY_THOUSANDS.sub(lambda m: str(int(m.group(1)) * 1000), s)
        s = RE_CURRENCY_MILLIONS.sub(lambda m: str(int(m.group(1)) * 1000000), s)
        s = RE_CURRENCY_BILLIONS.sub(lambda m: str(int(m.group(1)) * 1000000000), s)

        return s
