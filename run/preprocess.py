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
    text = text.replace("..", "$")
    text = text.replace(".", "$").replace("?", "$").replace("!", "$")
    
    # Remove extra spaces and indents
    text = re.sub(r"\s+", " ", text).strip()
    
    return text

def tag_pos(text):
    # Copy text inside stanford directory
    stanford_copy = "../stanford-postagger-full-2020-11-17/hold-input.txt"
    with open(stanford_copy, "w") as file:
        file.write(text)
        
    stanford_dir = "../stanford-postagger-full-2020-11-17/"
    stanford_output = "hold_tagged.txt"
    
    # Execute shell command to run filipino tagger
    cmd = [
            "java", "-mx300m",
            "-classpath", "stanford-postagger.jar",
            "edu.stanford.nlp.tagger.maxent.MaxentTagger",
            "-model", "models/filipino-left5words-owlqn2-distsim-pref6-inf2.tagger",
            "-textFile", "hold-input.txt"
        ]
    
    # Save output to temporary text file in run/
    with open(stanford_output, "w") as output_file:
            subprocess.run(cmd, stdout=output_file, stderr=subprocess.PIPE, text=True, cwd=stanford_dir)
            
    tagged_text = open_file(stanford_output)
    return tagged_text

def test():
    # gtest = "../txt/utf/gtest/19.txt"
    # text = open_file(gtest)
    
    text = "Ako si Ivy, si Judith ang aking ina."
    
    normalized = normalize(text)
    # print(normalized)
    
    # tagged = tag_pos(normalized)
    # print(tagged)
    
    tag_pos(normalized)
    tagged = open_file("hold_tagged.txt")
    print(tagged)
    
test()