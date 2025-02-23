import os
import re
import string
import subprocess
import shutil

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
                processed_words.append(f"{text}|{tag}")
            
            # Remove tags for end of sentence or fragment markers
            elif text in ("$", "#"):
                processed_words.append(text)
            else:
                processed_words.append(f"{text.lower()}|{tag}")
        else:
            processed_words.append(word)

    cleaned_text =  " ".join(processed_words)
    new_path = make_file(cleaned_text, file_path,"cleaned")
    

    return new_path
    

def normalize(file_path):
    text = open_file(file_path)
    
    # Remove quotation marks and commas
    text = text.replace("“", "").replace("”", "").replace(",", "")
    
    # Add # symbol for end of sentence fragment
    text = text.replace("…","#")
    
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
        subprocess.run(cmd, stdout=output_file_path, stderr=subprocess.PIPE, text=True, cwd=stanford_dir)
    
    return tagged_path
 
def test():
    file = "txt/utf/gtest/19.txt"
    grade = os.path.basename(os.path.dirname(file))
    
    normalized = normalize(file)
    tagged = tag(normalized)
    cleaned = clean_tag(tagged)
    
    print(cleaned)
    
        
test()