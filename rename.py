import os

def rename_files_in_directory(directory_path):
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            try:
                # Extract the integer from the filename (excluding the extension)
                file_number = int(os.path.splitext(filename)[0])
                # Add 200 to the integer
                new_file_number = file_number - 40
                # Create the new filename
                new_filename = f"{new_file_number}.txt"
                # Rename the file
                os.rename(
                    os.path.join(directory_path, filename),
                    os.path.join(directory_path, new_filename)
                )
                print(f"Renamed: {filename} -> {new_filename}")
            except ValueError:
                print(f"Skipping: {filename} (not a valid integer filename)")

# Example usage
directory_path = "txt/new_utf/g5"  # Replace with your directory path
rename_files_in_directory(directory_path)