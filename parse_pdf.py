import argparse
import os
import scipdf
import json
import spacy

# def tokenize_text(input_text):
#     nlp = spacy.load("en_core_web_lg")
#     doc = nlp(input_text)
#     tokens = [token.text for token in doc]    
#     return tokens

# def tokenize_file(input_file_path, output_file_path):
#     print('start tokenizing')
#     with open(input_file_path, 'r', encoding='utf-8') as input_file, \
#          open(output_file_path, 'w', encoding='utf-8') as output_file:
#         for line in input_file:
#             line = line.strip()  # Remove leading/trailing whitespace
#             if line:
#                 tokens = tokenize_text(line)
#                 # Write the tokenized sentence to the output file
#                 output_file.write(" ".join(tokens) + "\n")

# def parse_pdfjson(directory_name, idx, pdfjson, nlp):
#     filename = f"{directory_name}/{idx}-orig.txt"
#     print(f"Start parsing file {filename}")
#     data = json.loads(pdfjson)
#     title = data['title']
#     abstract = data['abstract']
#     with open(filename, 'w', encoding='utf-8') as output_file:
#                 output_file.write(f"{title}\n")
#                 output_file.write(f"{abstract}\n")
#     for section in data['sections']:
#          heading = section['heading']
#          # get text from section['text'] which is list, and write to file on each line
#          text = "\n".join(section['text'])
#          with open(filename, 'a', encoding='utf-8') as output_file:
#                 if heading:
#                     output_file.write(f"{heading}\n")
#                 if text:
#                     output_file.write(f"{text}\n")
#     print(f"File {filename} parsed to text file, start tokenizing")
    # # outfile = f"{directory_name}/{idx}.txt"
    # outfile = f"{directory_name}/testifworks-tokenized.txt"
    # tokenize_file(filename, outfile)
    # os.remove(filename)
    # print(f"File {filename} tokenized to {outfile}")


def parse_pdfjson(directory_name, idx, pdfjson, nlp):
    lines = []
    data = json.loads(pdfjson)
    lines.append(data['title'])
    lines.append(data['abstract'])
    for section in data['sections']:
         lines.append(section['heading'])
         for line in section['text']:
            lines.append(line)
    
    outfile = f"{directory_name}/{idx}.txt"
    with open(outfile, 'w', encoding='utf-8') as output_file:
        for doc in nlp.pipe(lines, n_process=-1, batch_size=4000):
            tokens = [token.text for token in doc]
            output_file.write(" ".join(tokens) + "\n")
    


def url_to_dict(directory_name, idx, url, nlp):
    pdfdict = scipdf.parse_pdf_to_dict(url, as_list=True)
    print(f"URL {idx} parsed to dictionary")
    pdfjson = json.dumps(pdfdict)
    parse_pdfjson(directory_name, idx, pdfjson, nlp)


def process_urls(args):
    try:
        # create directory to store input files
        filepath = args.filepath
        directory_name = os.path.splitext(os.path.basename(filepath))[0]
        nlp = spacy.load("en_core_web_lg")
        if not os.path.exists(directory_name):
            os.makedirs(directory_name)
        # load urls from file
        with open(filepath, 'r') as f:
            urls = f.readlines()
            for idx, url in enumerate(urls):
                url = url.strip()
                if not url:
                    continue
                url_to_dict(directory_name, str(idx), url, nlp)
        print(f"Finished parsing {len(urls)} urls")
    except FileNotFoundError:
        print(f"File not found: {filepath}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", type=str, default = 'test_parsing.txt', help='.txt file containing urls of pdf to be parsed')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    process_urls(args)