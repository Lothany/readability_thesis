import json
from pathlib import Path
from normalize import open_file

class DatasetEntry:
    def __init__(self, chunk, stride_len, stride_index, text_num, grade_lvl):
        self.dictionary = self.load_dictionary(self)
        self.text_num = text_num
        self.grade_lvl = grade_lvl
        self.chunk = chunk
        
        self.stride_len = stride_len
        self.stride_index = stride_index
        self.noun_tr, self.verb_tr = self.__lexical_var()
        # self.type_tr = None
        self.lex_density = self.__lexical_density()
        self.lex_foreign = self.__foreign()
        
        # self.sent_len = None
        # self.word_len = None
        # self.word_num = None
        # self.syll_num = None
        # self.poly_num = None
    
    @staticmethod
    def load_dictionary(self):
        dictionary_json = "tables/dictionary.json"
        with open(dictionary_json, "r", encoding="utf-8") as file:
            return json.load(file)
            
    def __lexical_var(self):
        noun_count = 0
        verb_count = 0
        for word in self.chunk:
            if word in self.dictionary:
                lexeme = self.dictionary[word]["lex_pos"]
                if lexeme == "noun":
                    noun_count = noun_count + 1
                elif lexeme == "verb":
                    verb_count = verb_count + 1
            else:
                print(f"Warning: '{word}' not found in dictionary. Skipping.")

        noun_tr = noun_count / self.stride_len
        noun_tr = float(f"{noun_tr:.4g}")
        
        verb_tr = verb_count / self.stride_len
        verb_tr = float(f"{verb_tr:.4g}")
        
        return noun_tr, verb_tr
    
    def __lexical_density(self):
        lexeme_count = 0
        for word in self.chunk:
            if word in self.dictionary:
                lexeme = self.dictionary[word]["lex_pos"]
                if lexeme != "other":
                    lexeme_count = lexeme_count + 1
            else:
                print(f"Warning: '{word}' not found in dictionary. Skipping.")
        
        lex_density = lexeme_count / self.stride_len
        lex_density = float(f"{lex_density:.4g}")
        
        return lex_density
    
    def __foreign(self):
        foreign_count = 0
        for word in self.chunk:
            if word in self.dictionary:
                foreign = self.dictionary[word]["lex_foreign"]
                if foreign:
                    foreign_count = foreign_count + 1
            else:
                print(f"Warning: '{word}' not found in dictionary. Skipping.")
        
        foreign_density = foreign_count / self.stride_len
        foreign_density = float(f"{foreign_density:.4g}")
        
        return foreign_density
            
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
        entry = DatasetEntry(chunk, n, stride_index, text_num, grade_lvl)
        print(f"{entry.text_num} | {entry.grade_lvl} | {entry.stride_len} | {entry.stride_index} | {entry.chunk} | {entry.noun_tr} | {entry.verb_tr} | {entry.lex_density} | {entry.lex_foreign}")
        
def run_prep_dataset(file):
    stride_n(file)
