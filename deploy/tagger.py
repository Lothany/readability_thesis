import subprocess

def tag(text):
    stanford_dir = "../stanford-postagger-full-2020-11-17/"
    stanford_copy = "../stanford-postagger-full-2020-11-17/hold-input.txt"

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
    result = subprocess.run(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, cwd=stanford_dir)
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
