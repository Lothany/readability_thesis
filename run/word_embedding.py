import os
import re
import subprocess


class LexicalMetadata:
    def __init__(self, word):
        self.lexeme = None
        self.is_foreign = None
        self.get_lexical_metadata(word)

    def __identify_lexeme(self, word):
        stanford_dir = "stanford-postagger-full-2020-11-17/"
        stanford_copy = "stanford-postagger-full-2020-11-17/hold-input.txt"
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

    # TODO: Add method to identify foreign words

    def get_lexical_metadata(self, word):
        if self.__identify_lexeme is None:
            self.__identify_lexeme(word)

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
        pattern = re.compile(
            f'({consonant_clusters}|{diphthongs}|{consonant_vowel})')
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
        word_traditional_metadata = TraditionalMetadata(word)
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
