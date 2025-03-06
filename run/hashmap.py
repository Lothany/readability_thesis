import re

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

import re

def embed(file_path):
    count = 0
    embedding = {}

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            words = [token.split('|')[0] for token in line.split()]
            
            for word in words:
                if word in embedding:
                    hashedWord = embedding.get(word)
                    hashedWord.addIndex(count)
                else:
                    instance = HashedWordEmbedding(word, count)
                    embedding[word] = instance
                count += 1

    return embedding
