import argparse
import os
import scipdf
import json
import spacy
import numpy as np
from bs4 import BeautifulSoup
import requests

def crawl_acl(output_file, venue, year, count, volume = None):
    page_url = "https://aclanthology.org/events/" + venue + "-" + str(year)
    if not volume:
        if venue == 'acl':
            conf_id = str(year)+venue+'-long'
        else:
            conf_id = str(year)+venue+'-main'
    else:
        conf_id = str(year)+volume
        
    response = requests.get(page_url)
    if response.status_code != 200:
        raise Exception(f"Check if the page exists: {page_url}")
    else:
        html = response.text

    soup = BeautifulSoup(html, 'html.parser')
    main_papers = soup.find('div', id = conf_id).find_all('p', class_ = "d-sm-flex")

    paper_list = []
    for paper_p in main_papers:
        pdf_url = paper_p.contents[0].contents[0]['href']
        # paper_span = paper_p.contents[-1]
        # assert paper_span.name == 'span'
        # paper_a = paper_span.strong.a
        # title = paper_a.get_text()
        # url = "https://aclanthology.org" + paper_a['href']
        paper_list.append(pdf_url)
    
    # select count number of papers randomly from paper_list
    paper_list = np.random.choice(paper_list, count, replace = False)
    
    # write txt file line by line in paper_lst
    with open(output_file, 'w') as f:
        for paper in paper_list:
            f.write(f"{paper}\n")



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