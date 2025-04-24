from normalize import run_normalize
from prep_dataset import run_prep_dataset, split_documents, get_documents
from word_embedding import WordEmbedding, LexicalMetadata, TraditionalMetadata

import os
from tqdm import tqdm

if __name__ == "__main__":   
    base_dir = "txt/utf" 
    
    # Split documents into training and testing sets
    train_files, test_files = split_documents(base_dir)
    # train_files = get_documents(base_dir)
    
    # Overwrite existing dataset csv file
    training_dataset_path = "tables/dataset.csv"
    if os.path.exists(training_dataset_path):
        os.remove(training_dataset_path)
    
    # Preprocess training files
    with tqdm(train_files, desc="Preprocessing Training Dataset", unit="file") as progress_bar:
        for file in progress_bar:
            # Log the current file being processed without interfering with the progress bar
            progress_bar.set_postfix(file=os.path.basename(file))
        
            # Normalize the text
            normalized_path = run_normalize(file)
            
            # Generate metadata for words in text
            wb = WordEmbedding(normalized_path)
            wb.toJSON()
            
            # Extract features    
            run_prep_dataset(normalized_path, training_dataset_path)
    
    # Preprocess testings files
    testing_dataset_path = "tables/dataset_testing.csv"
    if os.path.exists(testing_dataset_path):
        os.remove(testing_dataset_path)
        
    with tqdm(test_files, desc="Preprocessing Testing Dataset", unit="file") as progress_bar:
        for file in progress_bar:
            # Log the current file being processed without interfering with the progress bar
            progress_bar.set_postfix(file=os.path.basename(file))
        
            # Normalize the text
            normalized_path = run_normalize(file)
            
            # Generate metadata for words in text
            wb = WordEmbedding(normalized_path)
            wb.toJSON()
            
            # Extract features    
            run_prep_dataset(normalized_path, testing_dataset_path)
    
    
