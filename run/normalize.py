import os
import re
import subprocess

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

def punctuation(file_path):
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

    # normalized_path = make_file(text, file_path, "normalized")
    
    return text

def tag(text):   
    stanford_dir = "stanford-postagger-full-2020-11-17/"
    stanford_copy = "stanford-postagger-full-2020-11-17/hold-input.txt"
    with open(stanford_copy, "w") as file:
        file.write(text)
    
    # Directory containing the Stanford POS Tagger    
    cmd = [
        "java", "-mx1g",
        "-classpath", "stanford-postagger.jar",
        "edu.stanford.nlp.tagger.maxent.MaxentTagger",
        "-model", "models/filipino-left5words-owlqn2-distsim-pref6-inf2.tagger",
        "-textFile", "hold-input.txt"
    ]
    

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=stanford_dir)
    tagged_text = result.stdout
    
    return tagged_text

# Takes tagged file and changes content to lowercase except for proper nouns
def lowercase(text, file_path):
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

    normalized_text =  " ".join(processed_words)
    print(normalized_text)
    new_path = make_file(normalized_text, file_path,"normalized")
    

    return new_path

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

def run_normalize(file):
    print("Normalizing: ", file, "\n")
    
    text = punctuation(file)
    tagged = tag(text)    
    normalized_path = lowercase(tagged, file)
    print(f"Normalized text exported in: {normalized_path}")
    
    return normalized_path
