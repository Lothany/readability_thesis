from normalize import run_normalize
from prep_dataset import run_prep_dataset, split_documents
from word_embedding import WordEmbedding, LexicalMetadata, TraditionalMetadata

def run_file(file):
    normalized_path = run_normalize(file)
            
    # Generate metadata for words in text
    wb = WordEmbedding(normalized_path)
    wb.toJSON()
    
    # Extract features    
    run_prep_dataset(normalized_path)
    
run_file("txt/utf/g4/44.txt")