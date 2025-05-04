from compare_performance import training_accuracy

def main():    
    # Call the function with the provided argument
    model = "models/svm_models/svm_1_B_100.pkl"
    score, model_id = training_accuracy(model)
    print(f"Model ID: {model_id}")
    print(f"Training accuracy: {score:.4f}")

if __name__ == "__main__":
    main()