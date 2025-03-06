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

def normalize(text):
    # Remove quotation marks and commas
    text = text.replace("“", "").replace("”", "").replace(",", "")
    
    # Add # symbol for end of sentence fragment
    text = text.replace("…","#")
    
    # Add $ symbol for end of full sentence
    text = text.replace("..", " $")
    text = text.replace(".", " $").replace("?", " $").replace("!", " $")
    
    # Remove extra spaces and indents
    text = re.sub(r"\s+", " ", text).strip()

    stanford_copy(text)
    
    return text

def stanford_copy(text):
    directory = "stanford-postagger-full-2020-11-17/hold-input.txt"
    with open(directory, "w") as file:
        file.write(text)
    
def capitalize(text):
    words = text.split()
    new_text = []

    for word in words:
        if word not in ("$", "#"):
            if tag_word(word) != "NNP":
                word = word.lower()
        new_text.append(word)

    return " ".join(new_text)

def tag_text(text):
    # Copy text inside stanford directory
    stanford_copy = "stanford-postagger-full-2020-11-17/hold-input.txt"
    with open(stanford_copy, "w") as file:
        file.write(text)
        
    stanford_dir = "stanford-postagger-full-2020-11-17/"
    
    # Execute shell command to run filipino tagger
    cmd = [
            "java", "-mx1g",
            "-classpath", "stanford-postagger.jar",
            "edu.stanford.nlp.tagger.maxent.MaxentTagger",
            "-model", "models/filipino-left5words-owlqn2-distsim-pref6-inf2.tagger",
            "-textFile", "hold-input.txt"
        ]
    
    # Save output to temporary text file in run/
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=stanford_dir)
    tagged_txt = result.stdout
    
    return tagged_txt

def tag_file(file_path):   
    destination_file = "stanford-postagger-full-2020-11-17/hold-input.txt"
    shutil.copyfile(file_path, destination_file)

    # Directory containing the Stanford POS Tagger
    stanford_dir = "stanford-postagger-full-2020-11-17/"
    stanford_input = "hold-input.txt"

    cmd = [
        "java", "-mx300m",
        "-classpath", "stanford-postagger.jar",
        "edu.stanford.nlp.tagger.maxent.MaxentTagger",
        "-model", "models/filipino-left5words-owlqn2-distsim-pref6-inf2.tagger",
        "-textFile", stanford_input
    ]
    
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=stanford_dir)
    tagged_txt = result.stdout
    
    return tagged_txt

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
    return tag

def test():
    gtest = "txt/utf/gtest/19.txt"
    text = open_file(gtest)
    
    # text = "Ako si Ivy# si Judith ang aking ina$"
    
    normalized = normalize(text)
    # print(normalized)
    
    # for word in normalized.split():
    #     if word == "$":
    #         print ("$")
    #     elif word == "#":
    #         print ("#")
    #     else:
    #         print(tag_lex(word))
    
    # for word in normalized.split():
    #         print(tag_word(word))
    
    tagged_file = tag_text(normalized)
    print(open_file(tagged_file))
    
    # tag_pos(normalized)
    # tagged = open_file("hold_tagged.txt")
    # print(tagged)

def test2():
    gtest = "txt/utf/gtest/19.txt"
    
    text = open_file(gtest)
    normalized = normalize(text)
    tagged_file = tag_text(normalized)
    
    # tagged_file = tag_file(gtest)
    
    print(tagged_file)
    
    
    
test2()