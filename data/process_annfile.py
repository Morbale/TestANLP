import os, glob
import argparse
import json

cntdic = {'MethodName':0, 
          'HyperparameterName':0, 
          'HyperparameterValue':0, 
          'MetricName': 0, 
          'MetricValue': 0, 
          'TaskName':0, 
          'DatasetName':0,
          'O':0}

def process_conll(filepath, cnt):
    with open(filepath, 'r') as input_file, open('temp.conll', 'w') as temp_file:
        # Iterate through each line in the input file
        for line in input_file:
            # Check if the line contains "-DOCSTART- -X- O"
            if "-DOCSTART- -X- O" not in line:
                # Replace " -X- _ " with a space
                modified_line = line.replace(" -X- _ ", " ")
                # Write the modified line to the temporary file
                temp_file.write(modified_line)
                
                # move on to the next line if the current line is empty
                if cnt:
                    if not modified_line.strip():
                        continue
                    
                    tokens, named_entities = modified_line.strip().split()
                    if named_entities != 'O':
                        if named_entities[0] == 'B':
                            cntdic[named_entities[2:]] += 1
                    else:
                        cntdic[named_entities] += 1
    
    # Rename the temporary file to overwrite the original input file
    os.rename('temp.conll', filepath)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, help='folder containing conll files to be processed')
    parser.add_argument("--count", action='store_true', help='creates json of counts of each entity type')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    for file_path in glob.glob(os.path.join(args.folder, '*.conll')):
        process_conll(file_path, args.count)
    if args.count:
        with open(os.path.join(args.folder,'annotation_cnt.json'), 'w') as json_file:
            json.dump(cntdic, json_file)