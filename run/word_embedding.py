import os
import re
import subprocess
import foreignator
from pathlib import Path

class LexicalMetadata:
    def __init__(self, word):
        self.lexeme = None
        self.is_foreign = None
        self.get_lexical_metadata(word)

    def __repr__(self):
        return (f"LX:{str(self.lexeme)} IF:{str(self.is_foreign)}")
    
    def __identify_lexeme(self, word):
        stanford_dir = "../stanford-postagger-full-2020-11-17/"
        stanford_copy = "../stanford-postagger-full-2020-11-17/hold-input.txt"
        with open(stanford_copy, "w") as file:
            file.write(word)

        # Execute shell command to run filipino tagger
        cmd = [
            "java", "-mx300m",
            "-classpath", "stanford-postagger.jar",
            "edu.stanford.nlp.tagger.maxent.MaxentTagger",
            "-model", "models/filipino-left5words-owlqn2-distsim-pref6-inf2.tagger",
            "-textFile", "hold-input.txt"
        ]

        # Save output to temporary text file in run/
        result = subprocess.run(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True, cwd=stanford_dir)
        output = result.stdout
        # Get part after "|" and remove extra spaces
        tag = output.split("|", 1)[1].strip()
        if "NN" in tag:
            self.lexeme = "noun"
        elif "VB" in tag:
            self.lexeme = "verb"
        elif "JJ" in tag:
            self.lexeme = "adj"
        elif "RB" in tag:
            self.lexeme = "adv"
        else:
            self.lexeme = "other"

    def __identify_foreign(self, word):
        if self.lexeme == "other":
            self.is_foreign = False
        else:
            self.is_foreign = foreignator.identify(word)
            # self.is_foreign = True

        self.is_foreign = True

    def get_lexical_metadata(self, word):
        print(f"Getting the Lexical Metadata of {word}")
        if self.lexeme is None:
            self.__identify_lexeme(word)
        if self.is_foreign is None:
            self.__identify_foreign(word)

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

    def __embed_helper(self, word):
        word_traditional_metadata = TraditionalMetadata(word)
        word_lexical_metadata = LexicalMetadata(word)
        if word not in self.embeddings:
            self.embeddings[word] = {"traditional": word_traditional_metadata, "lexical": word_lexical_metadata}
        else:
            self.embeddings[word] = {"traditional": word_traditional_metadata, "lexical": word_lexical_metadata}

    def __embed_sanitize_text(self, text):
        sanitized = re.sub(
            r"[^\w\s-]", " ", text)
        return " ".join(sanitized.split()).lower()
    
    def __embed(self):
        if Path(self.input).exists():
            with open(self.input, "r", encoding="utf-8") as file:
                for line in file:
                    sanitized_line = self.__embed_sanitize_text(line)
                    words = [token.split('|')[0] for token in sanitized_line.split()]
                    for word in words:
                        self.__embed_helper(word)
        else:
            words = str(self.input).split(" ")
            for word in words:
                self.__embed_helper(word)

    def toJSON(self):
        import json

        with open('word_embeddings.json', 'w', encoding='utf-8') as f:
            json.dump(self.embeddings, f, ensure_ascii=False, indent=4)
