import re
from pylatexenc.latex2text import LatexNodes2Text

latex_converter = LatexNodes2Text()

def normalize(text):
    if not isinstance(text, str):
        return ""
    text = text.strip().lower()
    # Remove Bengali punctuation and standard punctuation
    text = re.sub(r"[।,?!\-—]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text

def extract_address_term(text):
    if not isinstance(text, str):
        return ""

    text = text.strip().lower()
    text = re.sub(r"[^\u0980-\u09FF\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    CANONICAL = ["আপনি", "তুমি", "তুই"]

    for token in text.split():
        for addr in CANONICAL:
            if token.startswith(addr):
                return addr
    return ""

def extract_option_number(text):
    if not isinstance(text, str):
        return ""

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    def strip_formatting(line):
        try:
            line = latex_converter.latex_to_text(line)
        except Exception:
            pass
        line = re.sub(r"[*_`~]", "", line)
        line = re.sub(r"^[>#\-]+\s*", "", line)
        line = re.sub(r"[ঃ:]", " ", line)
        return line.strip()

    if lines:
        last = strip_formatting(lines[-1])
        m = re.search(r"\b([1-4]|[১-৪])\b", last)
        if m:
            return m.group(1)

    if lines:
        first = strip_formatting(lines[0])
        m = re.search(r"\b([1-4]|[১-৪])\b", first)
        if m:
            return m.group(1)

    cleaned_text = strip_formatting(text)
    m = re.search(r"\b([1-4]|[১-৪])\b", cleaned_text)
    return m.group(1) if m else ""

def extract_choice_letter(text):
    if not isinstance(text, str):
        return ""
    m = re.search(r"\b([ABCD])\b", text.strip(), re.I)
    return m.group(1).upper() if m else ""