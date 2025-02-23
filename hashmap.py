import re

count = 0
embedding = {}

class HashedWordEmbedding:
    def __init__(self, word, initialIndex):
        self.word = word
        self.indexes = [initialIndex]

    def __eq__(self, other):
        if not isinstance(other, HashedWordEmbedding):
            return False
        return self.word == other.word

    def __repr__(self):
        return str((str(self.word), self.indexes))
         
    def addIndex(self, index):
        self.indexes.append(index)

with open("1_extract.txt", "r", encoding="utf-8") as file:
    for line in file:
        words = re.findall(r'\b[a-zA-Z]+\b', line)
        for word in words:
            if word in embedding:
                hashedWord = embedding.get(word)
                hashedWord.addIndex(count)
            else:
                instance = HashedWordEmbedding(word, count)
                embedding[word] = instance
            count+=1

for item in embedding.items():
    print(item, "\n")

print(len(embedding.items()))