import pickle
from features import get_fragments
from word_embedding import WordEmbedding
from normalizer import run_normalize
import re 

txt = "Ang food na kinakain ni Martha ay masarap, kinuha ko ito sa refrigerator. Sana Hindi Maubos"
new_txt = run_normalize(txt) 
print(new_txt)

# Result:
# kinuha|VBOF ko|PRS ito|PRO sa|CCT refrigerator|FW $ ang | DTC food | FW na | CCP kinakain | VBTR ni | DTP Martha | NNP ay | LM masarap | JJD

word_embedder = WordEmbedding(new_txt)
print(word_embedder)

# Result
    # ang: {'traditional': CC: 3 SC: 1 IP: False, 'lexical': LX: other IF: False}
    # food: {'traditional': CC: 4 SC: 1 IP: False, 'lexical': LX: other IF: True}
    # ni: {'traditional': CC: 2 SC: 1 IP: False, 'lexical': LX: other IF: False}
    # ay: {'traditional': CC: 2 SC: 1 IP: False, 'lexical': LX: other IF: False}
    # kinakain: {'traditional': CC: 8 SC: 3 IP: True, 'lexical': LX: verb IF: False}
    # Martha: {'traditional': CC: 6 SC: 2 IP: True, 'lexical': LX: noun IF: False}
    # na: {'traditional': CC: 2 SC: 1 IP: False, 'lexical': LX: other IF: False}
    # ito: {'traditional': CC: 3 SC: 1 IP: False, 'lexical': LX: other IF: False}
    # kinuha: {'traditional': CC: 6 SC: 3 IP: True, 'lexical': LX: verb IF: False}
    # masarap: {'traditional': CC: 7 SC: 2 IP: True, 'lexical': LX: adj IF: False}
    # sa: {'traditional': CC: 2 SC: 1 IP: False, 'lexical': LX: other IF: False}
    # refrigerator: {'traditional': CC: 12 SC: 5 IP: True, 'lexical': LX: other IF: True}
    # ko: {'traditional': CC: 2 SC: 1 IP: False, 'lexical': LX: other IF: False}


def sentence_length(all_words):
    total_sentences = 0
    total_words = 0
    sentence = []

    for word in all_words:
        if word == "$":
            if sentence:
                total_sentences += 1
                total_words += len(sentence)
                sentence = []
        elif word != "#":
            sentence.append(word)

    if sentence:
        total_sentences += 1
        total_words += len(sentence)

    avg_sentence_length = total_words / total_sentences if total_sentences > 0 else 0
    return avg_sentence_length




all_words = new_txt.split()
sent_len = sentence_length(all_words)

print(all_words)

# Result:
# ['ang|DTC', 'food|FW', 'na|CCP', 'kinakain|VBTR', 'ni|DTP', 'Martha|NNP', 'ay|LM',
#     'masarap|JJD', '#', 'kinuha|VBOF', 'ko|PRS', 'ito|PRO', 'sa|CCT', 'refrigerator|FW', '$']

content = []
for word in all_words:
    match = re.match(r"(.+)\|(.+)", word)
    if match:
        parsed_word, _ = match.groups()  # Extract the word before "|"
        content.append(parsed_word)
    elif word in {"$", "#"}:  # Retain special markers if not tagged
        content.append(word)

print(content)

# Result:
# ['ang', 'food', 'na', 'kinakain', 'ni', 'Martha', 'ay', 'masarap', '#', 'kinuha', 'ko', 'ito', 'sa', 'refrigerator', '$']

get_fragments(content, sent_len, word_embedder.keyvalues())

# print(chunks)

# split_text = [fragment.strip() for fragment in re.split(r'[$#]', text) if fragment.strip()]


