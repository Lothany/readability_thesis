import re
import unicodedata
from tagger import tag

pattern = re.compile(
    r'(?P<remove>[“”‘’\':;])|'           # Characters to remove
    r'(?P<fragment>[…,])|'               # Characters replaced with ' #'
    r'(?P<ellipsis>(?<!\.)\.\.(?!\.))|'  # '..' (not '...') replaced with ' $'
    r'(?P<sentence>[.?!])'
)

def replace_match(m):
    if m.group('remove'):
        return ''
    elif m.group('fragment'):
        return ' #'
    elif m.group('ellipsis'):
        return ' $'
    elif m.group('sentence'):
        return ' $'
    return m.group(0)  # fallback (shouldn’t happen)

def modify_punctuation(text):
    text = pattern.sub(replace_match, text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def lowercase(text):
    words = text.split()
    processed_words = []

    for word in words:
        match = re.match(r"(.+)\|(.+)", word)
        if match:
            text, tag = match.groups()
            if tag == "NNP":
                processed_words.append(f"{text}|{tag}")

            # Retain tags for end of sentence/fragment markers
            elif text in ("$", "#"):
                processed_words.append(f"{text}")
            else:
                processed_words.append(f"{text.lower()}|{tag}")
        else:
            processed_words.append(word)

    normalized_text = " ".join(processed_words)
    normalized_text = remove_accents(normalized_text)

    return normalized_text

def remove_accents(text: str) -> str:
    result = []
    for char in text:
        if char == "ñ" or char == "Ñ":
            result.append(char)
        else:
            # Normalize character and remove diacritics
            normalized = unicodedata.normalize('NFD', char)
            stripped = ''.join(
                c for c in normalized if unicodedata.category(c) != 'Mn')
            result.append(stripped)
    return ''.join(result)


def run_normalize(text):
    modified_punctuation = modify_punctuation(text)
    tagged = tag(modified_punctuation)
    normalized_path = lowercase(tagged)

    return normalized_path
