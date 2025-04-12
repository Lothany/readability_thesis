import re
import foreignator
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

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
        else:
            self.lexeme = "other"

    def __identify_foreign(self, word):
        self.is_foreign = foreignator.identify(word)
            # self.is_foreign = True

    def get_lexical_metadata(self, token):
        print(f"Getting the Lexical Metadata of \"{token[0]}\"")
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
        self.syllable_count = len(matches)
        self.is_polysyllabic = self.syllable_count > 3

    def get_traditional_metadata(self, word):
        print(f"Getting the Traditional Metadata of {word}")
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
    def __init__(self, file_path_or_text: str) -> None:
        self.embeddings =  {}
        self.input = file_path_or_text
        self.input = self.__embed()

    # def __eq__(self, other):
    #     if not isinstance(other, WordEmbedding):
    #         return False
    #     return self.word == other.word

    def __repr__(self):
        return (str(self.embeddings))

    def __embed_helper(self, tokens):
        word_traditional_metadata = TraditionalMetadata(tokens[0])
        word_lexical_metadata = LexicalMetadata(tokens)
        if tokens[0] not in self.embeddings:
            self.embeddings[tokens[0]] = {"traditional": word_traditional_metadata, "lexical": word_lexical_metadata}
        else:
            self.embeddings[tokens[0]] = {"traditional": word_traditional_metadata, "lexical": word_lexical_metadata}

    # def __embed_sanitize_text(self, text):
    #     sanitized = re.sub(
    #         r"[^\w\s-]", " ", text)
    #     return " ".join(sanitized.split()).lower()
    
    def __embed(self):
        if Path(self.input).exists():
            with open(self.input, "r", encoding="utf-8") as file:
                for line in file:
                    words = line.split(" ")
                    for word in words:
                        if word in ["$", "#"]:
                            pass
                        else:
                            tokens = word.split("|")
                            self.__embed_helper(tokens)
        else:
            words = str(self.input).split(" ")
            for word in words:
                tokens = word.split("|")
                self.__embed_helper(tokens)

    def toJSON(self, filepath='tables/word_embeddings.json'):
        import json

        def serialize_word(word_metadata):
            word, metadata = word_metadata
            return word, {
                **metadata["lexical"].to_dict(),
                **metadata["traditional"].to_dict()
            }

        with ThreadPoolExecutor() as executor:
            results = executor.map(serialize_word, self.embeddings.items())

        serializable_data = dict(results)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, ensure_ascii=False, indent=4)
