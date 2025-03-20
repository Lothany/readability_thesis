import json
import os
import csv
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
        self.type_tr = self.__type_token_ratio()
        self.lex_density = self.__lexical_density()
        self.lex_foreign = self.__foreign()
        
        self.sent_len = None
        self.word_len = None
        self.word_num = None
        self.syll_num = None
        self.poly_num = None
    
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
    
    def __type_token_ratio(self):
        unique_words = set(self.chunk)
        
        type_tr = len(unique_words) / self.stride_len
        type_tr = float(f"{type_tr:.4g}")
        
        return type_tr
        
            
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
        
def stride_sentence(file_path):    
    file = Path(file_path)
    text_num = int(file.stem)
    grade_lvl = int(file.parent.name[1:])
    
    text = open_file(file_path)
    all_words = text.split()
    
    sentence = []
    stride_index = 0
    
    for word in all_words:
        if word == "$":
            if sentence:
                n = len(sentence)
                sentence_features(sentence, n, stride_index, text_num, grade_lvl)
                # print(f"{stride_index}: {sentence}")
                stride_index = stride_index + 1
                sentence = []
        elif word != "#":
            sentence.append(word)

    # If the last sentence is not followed by "$", process it
    if sentence:
        n = len(sentence)
        features(sentence, n, text_num, grade_lvl)
    

def features(n_gram, n, text_num, grade_lvl):
    for stride_index, chunk in enumerate(n_gram):
        entry = DatasetEntry(chunk, n, stride_index, text_num, grade_lvl)
        export_csv(entry)
        
def sentence_features(sentence, n, stride_index, text_num, grade_lvl):    
    entry = DatasetEntry(sentence, n, stride_index, text_num, grade_lvl)
    export_csv(entry)
    
        

def export_csv(entry):
    new_entry = {"text_num": entry.text_num,
                 "grade_level": entry.grade_lvl,
                 "stride_len": entry.stride_len,
                 "stride_index": entry.stride_index,
                 "noun_tr": entry.noun_tr,
                 "verb_tr": entry.verb_tr,
                 "type_tr": entry.type_tr,
                 "lex_density": entry.lex_density,
                 "lex_foreign": entry.lex_foreign,
                 "sent_len": entry.sent_len,
                 "word_len": entry.word_len,
                 "word_num": entry.word_num,
                 "syll_num": entry.syll_num,
                 "poly_num": entry.poly_num}
    dataset_path = "tables/dataset.csv"
    
    file_exists = os.path.exists(dataset_path)
    with open(dataset_path, "a", newline="", encoding="utf-8") as dataset:
        fieldnames = ["text_num","grade_level","stride_len","stride_index","noun_tr","verb_tr","type_tr","lex_density","lex_foreign","sent_len","word_len","word_num","syll_num","poly_num"]
        writer = csv.DictWriter(dataset, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(new_entry)
        print(f"{entry.text_num} | {entry.grade_lvl} | {entry.stride_len} | {entry.stride_index} | {entry.chunk} | {entry.noun_tr} | {entry.verb_tr} | {entry.lex_density} | {entry.lex_foreign}")

def run_prep_dataset(file):
    print(f"Collecting data from: {file}")
    stride_n(file)
    stride_sentence(file)
