import os, glob
import argparse

def process_conll(filepath):
    with open(file_path, 'r') as input_file, open('temp.conll', 'w') as temp_file:
        # Iterate through each line in the input file
        for line in input_file:
            # Check if the line contains "-DOCSTART- -X- O"
            if "-DOCSTART- -X- O" not in line:
                # Replace " -X- _ " with a space
                modified_line = line.replace(" -X- _ ", " ")
                # Write the modified line to the temporary file
                temp_file.write(modified_line)
    
    # Rename the temporary file to overwrite the original input file
    os.rename('temp.conll', file_path)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, help='folder containing conll files to be processed')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    for file_path in glob.glob(os.path.join(args.folder, '*.conll')):
        process_conll(file_path)