import os
import re
import subprocess
import unicodedata

def open_file(file_path):
    if file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
    return content

# Creates new file directory for processed text files
def make_file(text, file_path, txt_type):
    grade = os.path.basename(os.path.dirname(file_path))
    file_name = os.path.basename(file_path)
    
    target_directory = os.path.join("txt", txt_type, grade)
    
    os.makedirs(target_directory, exist_ok=True)
    
    full_file_path = os.path.join(target_directory, file_name)
    
    with open(full_file_path, "w", encoding="utf-8") as new_file:
        new_file.write(text)
    
    return full_file_path

def punctuation(file_path):
    text = open_file(file_path)
    
    # Remove quotation marks and commas
    text = text.replace("“", "").replace("”", "")
    
    # Add # symbol for end of sentence fragment
    text = text.replace("…"," #")
    text = text.replace(",", "#")
    
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
    
    # Write the text to the input file for tagging
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
    
    # Run the Stanford POS Tagger
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=stanford_dir)
    tagged_text = result.stdout
    
    # Retain "$" and "#" symbols without tagging
    processed_lines = []
    for word in tagged_text.split():
        if word in {"$", "#"}:
            processed_lines.append(word)  # Retain the symbol as is
        else:
            processed_lines.append(word)  # Add the tagged word
    
    # Join the processed words back into a single string
    return " ".join(processed_lines)

# Takes tagged file and changes content to lowercase except for proper nouns
def lowercase(text, file_path):
    words = text.split()
    processed_words = []

    for word in words:
        match = re.match(r"(.+)\|(.+)", word)
        if match:
            text, tag = match.groups()
            if tag == "NNP":
                processed_words.append(f"{text}|{tag}")
            
            # Retain tags for end of sentence/fragment markers
            elif text in ("$", "#"):
                processed_words.append(f"{text}")
            else:
                processed_words.append(f"{text.lower()}|{tag}")
        else:
            processed_words.append(word)

    normalized_text = " ".join(processed_words)
    normalized_text = remove_accents(normalized_text)
    
    new_path = make_file(normalized_text, file_path, "normalized")
    return new_path

def remove_accents(text: str) -> str:
    result = []
    for char in text:
        if char == "ñ" or char == "Ñ":
            result.append(char)
        else:
            # Normalize character and remove diacritics
            normalized = unicodedata.normalize('NFD', char)
            stripped = ''.join(c for c in normalized if unicodedata.category(c) != 'Mn')
            result.append(stripped)
    return ''.join(result)

def run_normalize(file):
    print("Normalizing: ", file)
    
    text = punctuation(file)
    tagged = tag(text)    
    normalized_path = lowercase(tagged, file)
    print(f"Normalized text exported in: {normalized_path} \n")
    
    # print(open_file(normalized_path))
    
    return normalized_path

# run_normalize("txt/utf/g0/555.txt")

