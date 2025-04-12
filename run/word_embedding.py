import os
import re
import subprocess
import foreignator
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

class LexicalMetadata:
    def __init__(self, word):
        self.lexeme = None
        self.is_foreign = None
        self.get_lexical_metadata(word)

    def __repr__(self):
        return (f"LX:{str(self.lexeme)} IF:{str(self.is_foreign)}")
    
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

    def __identify_foreign(self, word):
        self.is_foreign = foreignator.identify(word)
            # self.is_foreign = True

    def get_lexical_metadata(self, word):
        if self.lexeme is None:
            self.__identify_lexeme(word)
        if self.is_foreign is None:
            self.__identify_foreign(word)
        print(f"\t{word} === {self}")

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
        if self.character_count is None:
            self.__count_characters(word)
        if self.syllable_count is None:
            self.__count_syllables(word)
        print(f"\t{word} === {self}")
    
    def to_dict(self):
        return {
            "trad_char": self.character_count,
            "trad_syll": self.syllable_count,
            "trad_poly": self.is_polysyllabic
        }


class WordEmbedding:
    def __init__(self, file_path_or_text, max_workers=8, batch_size=20):
        self.embeddings = {}
        self.input = file_path_or_text
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.words = self.__extract_words()
        self.total = len(self.words)
        self.__embed_all_words()

    def __sanitize(self, text):
        return " ".join(re.sub(r"[^\w\s-]", " ", text).split()).lower()

    def __extract_words(self):
        if Path(self.input).exists():
            with open(self.input, "r", encoding="utf-8") as f:
                text = self.__sanitize(f.read())
        else:
            text = self.__sanitize(str(self.input))
        return list(dict.fromkeys(text.split()))  # dedup & preserve order

    def __embed_word(self, word):
        try:
            return word, {
                "traditional": TraditionalMetadata(word),
                "lexical": LexicalMetadata(word)
            }
        except Exception as e:
            print(f"Failed to embed '{word}': {e}")
            return word, None

    def __embed_all_words(self):
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            batches = [self.words[i:i + self.batch_size]
                       for i in range(0, self.total, self.batch_size)]
            futures = []

            for batch in batches:
                for word in batch:
                    futures.append(executor.submit(self.__embed_word, word))

            for i, future in enumerate(as_completed(futures), 1):
                word, data = future.result()
                if data:
                    self.embeddings[word] = data
                self.__show_progress(i)

    def __show_progress(self, current):
        percent = (current / self.total) * 100
        bar_length = 30
        filled = int(bar_length * percent // 100)
        bar = "█" * filled + "-" * (bar_length - filled)
        print(
            f"Progress: |{bar}| {percent:5.1f}% ({current}/{self.total})", end=":\t")

    def toJSON(self, filepath='tables/word_embeddings.json'):
        import json

        serializable = {
            word: {**meta["lexical"].to_dict(), **
                   meta["traditional"].to_dict()}
            for word, meta in self.embeddings.items()
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable, f, ensure_ascii=False, indent=4)
        print(f"\n✅ JSON saved to {filepath}")
