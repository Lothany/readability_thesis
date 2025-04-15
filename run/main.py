from normalize import run_normalize
from prep_dataset import run_prep_dataset
from word_embedding import WordEmbedding, LexicalMetadata, TraditionalMetadata

if __name__ == "__main__":
    file = "txt/utf/g0/19.txt"
    
    normalized_path = run_normalize(file)
    
    wb = WordEmbedding(normalized_path)
    wb.toJSON()
    
    run_prep_dataset(normalized_path)
    
