import json
from preprocess import open_file

class DatasetEntry:
    def __init__(self, chunk, stride_len, stride_index):
        self.dictionary = self.load_dictionary(self)
        
        # self.text_num = None
        # self.grade_lvl = None
        self.chunk = chunk
        self.stride_len = stride_len
        self.stride_index = stride_index
        self.noun_tr = self.__noun_token_ratio()
        # self.verb_tr = None
        # self.type_tr = None
        # self.lex_density = None
        # self.lex_foreign = None
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
            

def stride_n(file_path, n):    
    text = open_file(file_path)
    all_words = text.split()

    words = [word for word in all_words if word not in {"$", "#"}]
    n_gram = [words[i:i + n] for i in range(0, len(words) - n + 1)]

    return n_gram, n

def features(n_gram, n):
    for stride_index, chunk in enumerate(n_gram):
        # print(chunk)
        entry = DatasetEntry(chunk, n, stride_index)
        print(f"{entry.stride_len} | {entry.stride_index} | {entry.chunk} | {entry.noun_tr}")
        
        # Get traditional properties (chunk, stride_index)
        
def main():
    n_gram, n = stride_n("txt/cleaned/gtest/short.txt", 3)
    features(n_gram, n)
    
main()
