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

def process_conll(folderpath, aggfile, cnt):
    with open(aggfile, 'w') as aggfile:
        for filename in glob.glob(os.path.join(folderpath, '*.conll')):
            with open(filename, 'r') as input_file:
                for line in input_file:
                    if "-DOCSTART- -X- O" not in line:
                        modified_line = line.replace(" -X- _ ", " ")
                        aggfile.write(modified_line)
                        
                        if cnt:
                            if not modified_line.strip():
                                continue
                            
                            tokens, named_entities = modified_line.strip().split()
                            if named_entities != 'O':
                                if named_entities[0] == 'B':
                                    cntdic[named_entities[2:]] += 1
                            else:
                                cntdic[named_entities] += 1

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, help='folder containing conll files to be processed')
    parser.add_argument("--aggfile", type=str, help='file path to aggregated conll file')
    parser.add_argument("--count", action='store_true', help='creates json of counts of each entity type')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    process_conll(args.folder, args.aggfile, args.count)
    if args.count:
        with open(os.path.join(args.folder,'annotation_cnt.json'), 'w') as json_file:
            json.dump(cntdic, json_file)