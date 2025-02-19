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
    text = capitalize(text)
    return text

def capitalize(text):
    words = text.split()
    new_text = []

    for word in words:
        if word not in ("$", "#"):
            if tag_lex(word) != "NNP":
                word = word.lower()
        new_text.append(word)

    return " ".join(new_text)

def tag_lex(word):
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
    # gtest = "txt/utf/gtest/19.txt"
    # text = open_file(gtest)
    
    text = "Ako si Ivy# si Judith ang aking ina$"
    
    normalized = normalize(text)
    print(normalized)
    
    # for word in normalized.split():
    #     if word == "$":
    #         print ("$")
    #     elif word == "#":
    #         print ("#")
    #     else:
    #         print(tag_lex(word))
    
    for word in normalized.split():
            print(tag_lex(word))
    
    # tagged = tag_pos(normalized)
    # print(tagged)
    
    # tag_pos(normalized)
    # tagged = open_file("hold_tagged.txt")
    # print(tagged)
    
test()