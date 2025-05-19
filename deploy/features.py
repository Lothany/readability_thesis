class Features:
    def __init__(self, chunk, stride_len, stride_index, sent_len, embeddings):
        self.dictionary = embeddings
        self.chunk = chunk
        
        self.stride_len = stride_len
        self.stride_index = stride_index
        self.noun_tr, self.verb_tr = self.__lexical_var()
        self.type_tr = self.__type_token_ratio()
        self.lex_density = self.__lexical_density()
        self.lex_foreign = self.__foreign()

        self.sent_len = sent_len 
        self.word_len = self.__average_word_len()
        self.word_num = None
        self.syll_num = self.__average_syll()
        self.poly_num = self.__polysyllabic_count()
    
    def __repr__(self):
        traditional_feature_set = f"\n\tsent_len: {self.sent_len} \n\tword_len: {self.word_len} \n\tsyll_num: {self.syll_num} \n\tpoly_num: {self.poly_num}"
        lexical_feature_set = f"\n\tnoun_tr: {self.noun_tr} \n\tverb_tr: {self.verb_tr} \n\ttype_tr: {self.type_tr} \n\tlex_density: {self.lex_density} \n\tlex_foreign: {self.lex_foreign}"
        return f"\nchunk {self.stride_index}: {self.chunk} with {self.stride_len} words = {traditional_feature_set} {lexical_feature_set}\n"

    def __lexical_var(self):
        noun_count = 0
        verb_count = 0
        for word in self.chunk:
            if word in self.dictionary:
                lexeme = self.dictionary[word]["lexical"].lexeme
                if lexeme == "noun":
                    noun_count = noun_count + 1
                elif lexeme == "verb":
                    verb_count = verb_count + 1

        noun_tr = noun_count / self.stride_len
        noun_tr = float(f"{noun_tr:.4g}")
        
        verb_tr = verb_count / self.stride_len
        verb_tr = float(f"{verb_tr:.4g}")
        
        return noun_tr, verb_tr
    
    def __lexical_density(self):
        lexeme_count = 0
        for word in self.chunk:
            if word in self.dictionary:
                lexeme = self.dictionary[word]["lexical"].lexeme
                if lexeme != "other":
                    lexeme_count = lexeme_count + 1
        
        lex_density = lexeme_count / self.stride_len
        lex_density = float(f"{lex_density:.4g}")
        
        return lex_density
    
    def __foreign(self):
        foreign_count = 0
        for word in self.chunk:
            if word in self.dictionary:
                foreign = self.dictionary[word]["lexical"].is_foreign
                if foreign:
                    foreign_count = foreign_count + 1
        
        foreign_density = foreign_count / self.stride_len
        foreign_density = float(f"{foreign_density:.4g}")
        
        return foreign_density
    
    def __type_token_ratio(self):
        unique_words = set(self.chunk)
        
        type_tr = len(unique_words) / self.stride_len
        type_tr = float(f"{type_tr:.4g}")
        
        return type_tr
    
    def __average_syll(self):
        total_syllables = 0
        word_count = 0
        
        for word in self.chunk:
            if word in self.dictionary:
                syllables = self.dictionary[word]["traditional"].syllable_count
                total_syllables += syllables
                word_count += 1
        
        # Avoid division by zero if no words were found in dictionary
        if word_count == 0:
            return 0.0
            
        avg_syllables = total_syllables / word_count
        return float(f"{avg_syllables:.4g}")
        
    def __average_word_len(self):
        total_chars = 0
        word_count = len(self.chunk)
        
        if word_count == 0:
            return 0.0
            
        for word in self.chunk:
            total_chars += len(word)
            
        avg_word_len = total_chars / word_count
        return float(f"{avg_word_len:.4g}")
        
    def __polysyllabic_count(self):
        poly_count = 0
        
        for word in self.chunk:
            if word in self.dictionary:
                syllables = self.dictionary[word]["traditional"].is_polysyllabic
                if syllables:
                    poly_count += 1
        
        return poly_count
    
    def get_features(self):
        return [self.sent_len, self.word_len, self.syll_num, self.poly_num, self.noun_tr, self.verb_tr, self.type_tr, self.lex_density, self.lex_foreign]

def sentence_features(sentence, stride_index, sent_len, embeddings):
    n = len(sentence)
    entry = Features(sentence, n, stride_index, sent_len, embeddings)
    return entry

def get_fragments(content, sent_len, embeddings):
    chunks = []

    # Chunk by sentences
    sentence = []
    for word in content:
        if word == "$":
            if sentence:
                sentence = []
        elif word != "#":
            sentence.append(word)

    # Chunk by fragments
    fragment = []
    prev_period = False
    stride_index = 0
    for word in content:
        if word == "#":
            if fragment:
                chunks.append(sentence_features(fragment, stride_index, sent_len, embeddings))
                prev_period = False
                stride_index += 1
                fragment = []
        elif word == "$":
            if fragment:
                if not prev_period:
                    chunks.append(sentence_features(fragment, stride_index, sent_len, embeddings))
                    prev_period = True
                    stride_index += 1
                    fragment = []
                else:
                    fragment = []
        else:
            fragment.append(word)

    if sentence:
        chunks.append(sentence_features(fragment, stride_index, sent_len, embeddings))

    return chunks