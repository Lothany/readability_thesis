from normalize import run_normalize
from prep_dataset import run_prep_dataset

if __name__ == "__main__":
    file = "txt/utf/g0/420.txt"
    
    normalized_path = run_normalize(file)
    run_prep_dataset(normalized_path)
    
