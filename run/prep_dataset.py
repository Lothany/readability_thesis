import json
from pathlib import Path
from normalize import open_file

class DatasetEntry:
    def __init__(self, chunk, stride_len, stride_index, text_num, grade_lvl, syll_num):
        self.dictionary = self.load_dictionary(self)
        
        self.text_num = text_num
        self.grade_lvl = grade_lvl
        self.chunk = chunk
        self.stride_len = stride_len
        self.stride_index = stride_index
        self.noun_tr = self.__noun_token_ratio()
        # self.verb_tr = None
        # self.type_tr = None
        # self.lex_density = None
        # self.lex_foreign = None
        
        # To do (Traditional)
        # self.sent_len = None 
        self.word_len = self.__average_word_len()
        # self.word_num = None
        self.syll_num = self.__average_syll()
        self.poly_num = self.__polysyllabic_count()
    
    @staticmethod
    def load_dictionary(self):
        dictionary_json = "tables/dictionary.json"
        with open(dictionary_json, "r", encoding="utf-8") as file:
            return json.load(file)
            
    def __noun_token_ratio(self):
        noun_count = 0
        for word in self.chunk:
            if word in self.dictionary:
                lexeme = self.dictionary[word]["lex_pos"]
                if lexeme == "noun":
                    noun_count = noun_count + 1
            else:
                print(f"Warning: '{word}' not found in dictionary. Skipping.")

        num = noun_count / self.stride_len
        return float(f"{num:.4g}")
    
    def __average_syll(self):
        total_syllables = 0
        word_count = 0
        
        for word in self.chunk:
            if word in self.dictionary:
                syllables = self.dictionary[word]["trad_syll"]
                total_syllables += syllables
                word_count += 1
            else:
                print(f"Warning: '{word}' not found in dictionary. Skipping.")
        
        # Avoid division by zero if no words were found in dictionary
        if word_count == 0:
            return 0.0
            
        avg_syllables = total_syllables / word_count
        return float(f"{avg_syllables:.4g}")
        
    def __average_word_len(self):
        """Calculate the average word length (in characters) in the stride."""
        total_chars = 0
        word_count = len(self.chunk)
        
        if word_count == 0:
            return 0.0
            
        for word in self.chunk:
            total_chars += len(word)
            
        avg_word_len = total_chars / word_count
        return float(f"{avg_word_len:.4g}")
        
    def __polysyllabic_count(self):
        """Count the number of words with 3 or more syllables in the stride."""
        poly_count = 0
        
        for word in self.chunk:
            if word in self.dictionary:
                syllables = self.dictionary[word]["trad_poly"]
                if syllables:
                    poly_count += 1
            else:
                print(f"Warning: '{word}' not found in dictionary. Skipping word for polysyllabic count.")
        
        return poly_count

def stride_n(file_path):    
    file = Path(file_path)
    text_num = int(file.stem)
    grade_lvl = int(file.parent.name[1:])
    
    text = open_file(file_path)
    all_words = text.split()
    
    n_values = {2, 3, 5}
    for n in n_values:
        words = [word for word in all_words if word not in {"$", "#"}]
        n_gram = [words[i:i + n] for i in range(0, len(words) - n + 1)]

        features(n_gram, n, text_num, grade_lvl)

def features(n_gram, n, text_num, grade_lvl):
    for stride_index, chunk in enumerate(n_gram):
        entry = DatasetEntry(chunk, n, stride_index, text_num, grade_lvl, None)
        print(f"{entry.text_num} | {entry.grade_lvl} | {entry.stride_len} | {entry.stride_index} | {entry.chunk} | {entry.noun_tr} | Traditional - WL: {entry.word_len}, SC: {entry.syll_num}, PC: {entry.poly_num}")
        
def run_prep_dataset(file):
    stride_n(file)
