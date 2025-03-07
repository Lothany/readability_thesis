import os
import re

class LexicalMetadata:
    def __init__(self):
        self.lexeme = None
        self.is_foreign = None

    def __identify_lexeme(self):
        

class TraditionalMetadata:
    def __init__(self, word):
        self.character_count = None
        self.syllable_count = None
        self.is_polysyllabic = None
        self.get_traditional_metadata(word)

    def __count_characters(self, word):
        self.character_count = len(word)
    
    def __count_syllables(self, word):
        consonant_vowel = r'([b-df-hjklmnp-rtvwz][aeiou])'
        diphthongs = r'(ai|ei|oi|ui|au|ou)'
        consonant_clusters = r'(ng|br|tr|st)'
        pattern = re.compile(f'({consonant_clusters}|{diphthongs}|{consonant_vowel})')
        matches = pattern.findall(word.lower())
        self.syllable_count = len(matches)
        self.is_polysyllabic = self.syllable_count > 1

    def get_traditional_metadata(self, word):
        if self.__count_characters is None: 
            self.__count_characters(word)
        if self.__count_syllables is None:
            self.__count_syllables(word)

class WordEmbedding:
    def __init__(self, file_path_or_text: str) -> None:
        self.embeddings = self.__embed(file_path_or_text)

    def __eq__(self, other):
        if not isinstance(other, WordEmbedding):
            return False
        return self.word == other.word

    def __repr__(self):
        return (str(self.word))

    def __embed_helper(self, word):
        word_traditional_metadata = TraditionalMetadata()
        if word not in self.embeddings:
            self.embeddings[word] = {"traditional": word_traditional_metadata}
        else:
            self.embeddings[word]["traditional"] = word_traditional_metadata
            # Add tradtional

    def __embed(self, file_path_or_text):
        if os.path.exists(file_path_or_text):
            with open(file_path_or_text, "r", encoding="utf-8") as file:
                for line in file:
                    words = [token.split('|')[0] for token in line.split()]
                    for word in words:
                        self.__embed_helper(word)
        else:
            words = str(file_path_or_text).split(" ")
            for word in words:
                self.__embed_helper(word)
