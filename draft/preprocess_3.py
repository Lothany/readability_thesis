import os
import re
import string
import subprocess
import shutil
import hashlib

def open_file(file_path):
    if file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
    return content

# Creates new file directory for processed text files
def make_file(text, file_path, txt_type):
    grade = os.path.basename(os.path.dirname(file_path))
    file_name = os.path.basename(file_path)
    target_directory = os.path.join("txt", txt_type, grade, file_name)
    
    with open(target_directory, "w", encoding = "utf-8") as new_file:
        new_file.write(text)
    
    return target_directory    

# Takes tagged file and changes content to lowercase except for proper nouns
def clean_tag(file_path):
    text = open_file(file_path)
    words = text.split()
    processed_words = []

    for word in words:
        match = re.match(r"(.+)\|(.+)", word)
        if match:
            text, tag = match.groups()
            if tag == "NNP":
                processed_words.append(f"{text}")
            
            # Remove tags for end of sentence/fragment markers
            elif text in ("$", "#"):
                processed_words.append(text)
            else:
                processed_words.append(f"{text.lower()}")
        else:
            processed_words.append(word)

    cleaned_text =  " ".join(processed_words)
    print(cleaned_text)
    new_path = make_file(cleaned_text, file_path,"cleaned")
    

    return new_path
    

def normalize(file_path):
    text = open_file(file_path)
    
    # Remove quotation marks and commas
    text = text.replace("“", "").replace("”", "").replace(",", "")
    
    # Add # symbol for end of sentence fragment
    text = text.replace("…"," #")
    
    # Add $ symbol for end of full sentence
    text = text.replace("..", " $")
    text = text.replace(".", " $").replace("?", " $").replace("!", " $")
    
    # Remove extra spaces and indents
    text = re.sub(r"\s+", " ", text).strip()

    normalized_path = make_file(text, file_path, "normalized")
    
    return normalized_path

def tag(file_path):   
    destination_file = "stanford-postagger-full-2020-11-17/hold-input.txt"
    shutil.copyfile(file_path, destination_file)
    
    grade = os.path.basename(os.path.dirname(file_path))
    file_name = os.path.basename(file_path)
    tagged_path = os.path.join("txt", "tagged", grade, file_name)
    os.makedirs(os.path.dirname(tagged_path), exist_ok=True)
    
    # Directory containing the Stanford POS Tagger
    stanford_dir = "stanford-postagger-full-2020-11-17/"
    stanford_input = "hold-input.txt"
    
    cmd = [
        "java", "-mx1g",
        "-classpath", "stanford-postagger.jar",
        "edu.stanford.nlp.tagger.maxent.MaxentTagger",
        "-model", "models/filipino-left5words-owlqn2-distsim-pref6-inf2.tagger",
        "-textFile", stanford_input
    ]
    

    with open(tagged_path, "w") as output_file_path:
        subprocess.run(cmd, stdout=output_file_path, stderr=subprocess.PIP0E, text=True, cwd=stanford_dir)
    
    return tagged_path

def hash_word(word: str) -> str:
    return int(hashlib.sha256(word.encode()).hexdigest(), 16)


# Making LEXICAL Dictionary ------------------------------------------------------------
from word_embedding import WordEmbedding
import csv

def lexeme(tag):
    if "NN" in tag:
        return "noun"
    elif "VB" in tag:
        return "verb"
    elif "JJ" in tag:
        return "adj"
    elif "RB" in tag:
        return "adv"
    else:
        return "other"
    

def tag_word(word):
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
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=stanford_dir)
    output = result.stdout
    tag = output.split("|", 1)[1].strip()  # Get part after "|" and remove extra spaces
    lex_tag = lexeme(tag)
    return lex_tag
    
    
def lex_dict(embedding):
    dictionary_path = "tables/dictionary.csv"
    for word in embedding.keys():
        if word not in ("$", "#"):
            unique = set()
            with open(dictionary_path, "r", newline="", encoding="utf-8") as dictionary:
                reader = csv.DictReader(dictionary)
                for row in reader:
                    unique.add(row["Word"])
                    
            if word not in unique:
                lexeme = tag_word(word)
                new_word = {"Word": word, "Lex_pos": lexeme}
                
                with open(dictionary_path, "a", newline="", encoding="utf-8") as dictionary:
                    fieldnames = ["Word", "Lex_pos", "Lex_foreign", "Trad_char", "Trad_syll", "Trad_poly"]
                    writer = csv.DictWriter(dictionary, fieldnames=fieldnames)
                    writer.writerow(new_word)
        
    print("Hashed")

    

    

def test():
    file = "txt/utf/gtest/short.txt"
    print("Preprocessing: ", file, "\n")
    
    normalized = normalize(file)
    tagged = tag(normalized)    
    cleaned = clean_tag(tagged)
    print(f"Path: {cleaned}")
    # embedding = embed(cleaned)
    
    # lex_dict(embedding)
    
    # print(f"\nFinal file: ", cleaned)
