import json
import os
import re
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
        self.word_len = self.__average_word_len()
        self.word_num = None
        self.syll_num = self.__average_syll()
        self.poly_num = self.__polysyllabic_count()
    
    @staticmethod
    def load_dictionary(self):
        dictionary_json = "tables/word_embeddings.json"
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

def clean_tags(file_path):
    file = Path(file_path)
    text_num = int(file.stem)
    grade_lvl = int(file.parent.name[1:])
    
    text = open_file(file_path)
    all_words = text.split()
    
    # Extract words without tags
    content = []
    for word in all_words:
        match = re.match(r"(.+)\|(.+)", word)
        if match:
            parsed_word, _ = match.groups()  # Extract the word before "|"
            content.append(parsed_word)
        elif word in {"$", "#"}:  # Retain special markers if not tagged
            content.append(word)
    
    return content, text_num, grade_lvl
    
def stride_n(file_path):    
    content, text_num, grade_lvl = clean_tags(file_path)
    
    # Remove # and $ symbols from the content
    filtered_content = [word for word in content if word not in {"#", "$"}]
    
    n_values = {1, 2, 3}
    for n in n_values:
        n_gram = [filtered_content[i:i + n] for i in range(0, len(filtered_content) - n + 1)]
        features(n_gram, n, text_num, grade_lvl)
        
# def stride_sentence(file_path):    
#     content, text_num, grade_lvl = clean_tags(file_path)
    
#     # print(content)
    
#     on_fragment = True
#     on_sentence = False
    
#     sentence = []
#     fragment = []
    
#     stride_index = 0
    
#     for word in content:
#         if word == "#":  # End of a sentence fragment
#             if fragment and on_fragment and not on_sentence:
#                 n = len(fragment)
#                 sentence_features(fragment, n, stride_index, text_num, grade_lvl)
#                 stride_index += 1
#                 fragment = []  # Reset the fragment after processing
#                 on_fragment = True
#                 on_sentence = False
        
#         elif word == "$":  # End of a full sentence
#             if fragment and on_fragment:
#                 n = len(fragment)
#                 sentence_features(fragment, n, stride_index, text_num, grade_lvl)
#                 stride_index += 1
#                 on_fragment = False
#                 on_sentence = True
#                 fragment = []
                
#             if sentence:
#                 n = len(sentence)
#                 sentence_features(sentence, n, stride_index, text_num, grade_lvl)
#                 stride_index += 1
#                 sentence = []
#                 on_fragment = False
#                 on_sentence = True
                
#         else:
#             sentence.append(word)
#             fragment.append(word)


#     # If the last sentence or fragment is not followed by "$", process it
#     if sentence:
#         n = len(sentence)
#         sentence_features(sentence, n, stride_index, text_num, grade_lvl)
    
def stride_sentence(file_path):    
    content, text_num, grade_lvl = clean_tags(file_path)

    chunks = []
    current_fragment = []
    current_sentence = []

    stride_index = 0

    for word in content:
        if word == "#":
            if current_fragment:
                # Process the fragment (no "#")
                n = len(current_fragment)
                sentence_features(current_fragment, n, stride_index, text_num, grade_lvl)
                stride_index += 1
                chunks.append(current_fragment[:])

                # Add to sentence context
                current_sentence.extend(current_fragment)
                current_fragment = []

        elif word == "$":
            if current_fragment:
                # Final fragment before sentence end
                n = len(current_fragment)
                sentence_features(current_fragment, n, stride_index, text_num, grade_lvl)
                stride_index += 1
                chunks.append(current_fragment[:])
                current_sentence.extend(current_fragment)
                current_fragment = []

            if current_sentence:
                # Process full sentence
                n = len(current_sentence)
                sentence_features(current_sentence, n, stride_index, text_num, grade_lvl)
                stride_index += 1
                chunks.append(current_sentence[:])
                current_sentence = []

        else:
            current_fragment.append(word)

    # Catch any leftover content
    if current_fragment:
        n = len(current_fragment)
        sentence_features(current_fragment, n, stride_index, text_num, grade_lvl)
        chunks.append(current_fragment)

    if current_sentence:
        n = len(current_sentence)
        sentence_features(current_sentence, n, stride_index, text_num, grade_lvl)
        chunks.append(current_sentence)

    # return chunks  # optional: return for testing



    

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
        print(f"{entry.text_num} | {entry.grade_lvl} | {entry.stride_len} | {entry.stride_index} | {entry.chunk} | NTR: {entry.noun_tr} | VTR: {entry.verb_tr} | LD: {entry.lex_density} | FD: {entry.lex_foreign} | WL: {entry.word_len}, SC: {entry.syll_num}, PC: {entry.poly_num}")
        # print(f"{entry.stride_len} {entry.chunk}")
        # print(f"NTR: {entry.noun_tr} | VTR: {entry.verb_tr} | LD: {entry.lex_density} | FD: {entry.lex_foreign} | WL: {entry.word_len}, SC: {entry.syll_num}, PC: {entry.poly_num}")
        # print(f"{entry.chunk}")

def run_prep_dataset(file):
    print(f"Collecting data from: {file}\n")
    stride_n(file)
    stride_sentence(file)
