import re
import foreignator
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

class LexicalMetadata:
    def __init__(self, token):
        self.lexeme = None
        self.is_foreign = None
        self.get_lexical_metadata(token)

    def __repr__(self):
        return (f"LX:{str(self.lexeme)} IF:{str(self.is_foreign)}")
    
    def __identify_lexeme(self, tag):
        if tag.startswith("NN"):
            self.lexeme = "noun"
        elif tag.startswith("VB"):
            self.lexeme = "verb"
        elif tag.startswith("JJ"):
            self.lexeme = "adj"
        elif tag.startswith("RB"):
            self.lexeme = "adv"
        elif tag.startswith("FW"):
            self.lexeme = "other"
            self.is_foreign = True
        else:
            self.lexeme = "other"

    def __identify_foreign(self, word):
        self.is_foreign = foreignator.identify(word)

    def get_lexical_metadata(self, token):
        if self.lexeme is None:
            self.__identify_lexeme(token[1])
        if self.is_foreign is None:
            self.__identify_foreign(token[0])

    def to_dict(self):
        return {
            "lex_pos": self.lexeme,
            "lex_foreign": self.is_foreign
        }

class TraditionalMetadata:
    def __init__(self, word):
        self.character_count = None
        self.syllable_count = None
        self.is_polysyllabic = None
        self.get_traditional_metadata(word)

    def __repr__(self):
        return (f"CC:{str(self.character_count)} SC:{str(self.syllable_count)} IP:{str(self.is_polysyllabic)}")

    def __count_characters(self, word):
        self.character_count = len(word)

    def __count_syllables(self, word):
        consonant_vowel = r'([b-df-hjklmnp-rtvwz][aeiou])'
        diphthongs = r'(ai|ei|oi|ui|au|ou)'
        consonant_clusters = r'(ng|br|tr|st)'
        pattern = re.compile(
            f'({consonant_clusters}|{diphthongs}|{consonant_vowel})')
        matches = pattern.findall(word.lower())
        self.syllable_count = 1 if len(matches) <= 0 else len(matches) 
        self.is_polysyllabic = self.syllable_count > 1

    def get_traditional_metadata(self, word):
        if self.character_count is None:
            self.__count_characters(word)
        if self.syllable_count is None:
            self.__count_syllables(word)
    
    def to_dict(self):
        return {
            "trad_char": self.character_count,
            "trad_syll": self.syllable_count,
            "trad_poly": self.is_polysyllabic
        }
class WordEmbedding:
    def __init__(self, file_path_or_text, max_workers=12):
        self.embeddings = {}
        self._cache = set()
        self._lock = Lock()
        self.input = file_path_or_text
        self.max_workers = max_workers
        self._embed()

    def __repr__(self):
        printable = "\n"
        for key, value in self.embeddings.items():
            printable += f"{key}: {value}\n"
        return printable
    
    def _embed_helper(self, tokens):
        word = tokens[0]

        with self._lock:
            if word in self._cache:
                return
            self._cache.add(word)

        traditional = TraditionalMetadata(word)
        lexical = LexicalMetadata(tokens)

        with self._lock:
            self.embeddings[word] = {
                "traditional": traditional,
                "lexical": lexical
            }

    def _get_tokens(self):
        all_tokens = []
        # if Path(self.input).exists():
        #     with open(self.input, "r", encoding="utf-8") as file:
        #         for line in file:
        #             for word in line.split():
        #                 if word not in ["$", "#"]:
        #                     tokens = word.split("|")
        #                     if len(tokens) >= 2:
        #                         all_tokens.append(tokens)
        # else:
        for word in str(self.input).split():
            if word not in ["$", "#"]:
                tokens = word.split("|")
                if len(tokens) >= 2:
                    all_tokens.append(tokens)
        return all_tokens

    def _embed(self):
        tokens_list = self._get_tokens()
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self._embed_helper, tokens)
                    for tokens in tokens_list]
            for future in as_completed(futures):
                future.result()

    def toJSON(self, filepath='tables/word_embeddings.json'):
        import json

        serializable = {
            word: {
                **meta["lexical"].to_dict(), 
                **meta["traditional"].to_dict()
                }
            for word, meta in self.embeddings.items()
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable, f, ensure_ascii=False, indent=4)
        print(f"\n✅ JSON saved to {filepath}")

    def keyvalues(self):
        return self.embeddings