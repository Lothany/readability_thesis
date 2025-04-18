import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random
from sklearn.model_selection import train_test_split
from collections import defaultdict

base_dir = "txt/utf"
test_size=0.2
random_seed=42

# Group files by grade level
grade_level_files = defaultdict(list)
for root, _, files in os.walk(base_dir):
    for file in files:
        if file.endswith(".txt"):
            grade_level = os.path.basename(root)  # Parent directory as grade level
            grade_level_files[grade_level].append(os.path.join(root, file))

# Split files by grade level
train_files = []
test_files = []
for grade_level, files in grade_level_files.items():
    random.seed(random_seed)
    random.shuffle(files)
    train, test = train_test_split(files, test_size=test_size, random_state=random_seed)
    train_files.extend(train)
    test_files.extend(test)

# Data for histogram
data = {'Train': len(train_files), 'Test': len(test_files)}

# Create histogram
plt.bar(data.keys(), data.values(), color=['blue', 'orange'])
plt.title('Number of Files in Train and Test Sets by Grade Level')
plt.xlabel('Dataset')
plt.ylabel('Number of Files')
plt.show()


