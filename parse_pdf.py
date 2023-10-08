import argparse
import os
import scipdf
import json
import spacy
import numpy as np
from bs4 import BeautifulSoup
import requests
from tqdm import tqdm
import pandas as pd

def crawl_acl(output_file, venue, year, count, volume=None):

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
    with open(output_file, 'a') as f:
        for paper in paper_list:
            f.write(f"{paper}\n")
    return output_file

def crawl_combination(args):
    outfile = args.urlpath
    if os.path.exists(outfile):
        raise FileExistsError(f"File already exists, choose different name: {outfile}")
    
    with open(args.comb_dict, 'r') as f:
        comb = json.load(f)
    
    for venue in comb.keys():
        for year in comb[venue].keys():
            crawl_acl(outfile, venue, year, comb[venue][year])
    print("Finished crawling combination")


nlp = spacy.load("en_core_web_lg")

def tokenize_text(input_text, pbar):
    doc = nlp(input_text)
    tokens = [token.text for token in doc]
    pbar.update(1)
    return ' '.join(tokens)


def parse_pdfjson(directory_name, idx, pdfjson):
    lines = []
    data = json.loads(pdfjson)
    lines.append(data['title'])
    lines.append(data['abstract'])
    for section in data['sections']:
         lines.append(section['heading'])
         for line in section['text']:
            lines.append(line)
    
    outfile = f"{directory_name}/{idx}.txt"
    df = pd.DataFrame({"text":lines})
    with tqdm(total=len(df), desc="Tokenizing", unit="texts") as pbar:
        df["tokens"] = df["text"].apply(lambda x: tokenize_text(x, pbar))
    with open(outfile, 'w', encoding='utf-8') as output_file:
        for row in df['tokens']:
            output_file.write(row + "\n")


def url_to_dict(directory_name, idx, url):
    pdfdict = scipdf.parse_pdf_to_dict(url, as_list=True)
    print(f"URL {idx} parsed to dictionary")
    pdfjson = json.dumps(pdfdict)
    parse_pdfjson(directory_name, idx, pdfjson)


def process_urls(args):
    try:
        # create directory to store input files
        filepath = args.urlpath
        directory_name = os.path.splitext(os.path.basename(filepath))[0]
        
        if not os.path.exists(directory_name):
            os.makedirs(directory_name)
        # load urls from file
        with open(filepath, 'r') as f:
            urls = f.readlines()
            for idx, url in enumerate(urls):
                url = url.strip()
                if not url:
                    continue
                url_to_dict(directory_name, str(idx), url)
        print(f"Finished parsing {len(urls)} urls")
    except FileNotFoundError:
        print(f"Crawl to create file first: {filepath}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--crawl", action='store_true', help='start from crawlingl else parse the txt file')
    parser.add_argument("--use_comb", action='store_true', help='crawl using combination dict')
    parser.add_argument("--comb_dict", type=str, default = 'comb_dict.json', help='dict containing combinations of venue and year')
    
    parser.add_argument("--urlpath", type=str, default = 'test_parsing.txt', help='.txt file containing urls of pdf to be parsed')
    
    parser.add_argument("--venue", type=str, choices=['acl','emnlp','naacl'], default = 'acl', help='venue of the conference')
    parser.add_argument("--year", type=int, choices = [2023, 2022, 2021], default = 2023, help='year of the conference')
    parser.add_argument("--count", type=int, default = 10, help='number of papers to be parsed')
    parser.add_argument("--volume", type=str, default = None, help='volume of the conference')
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    if args.crawl:
        if args.use_comb:
            crawl_combination(args)
        else:
            if args.volume:
                print(f'Crawling ACL Anthology, with {args.count} papers from {args.venue} {args.year}, volume {args.volume}')
                crawl_acl(args.urlpath, args.venue, args.year, args.count, args.volume)
            else:
                print(f'Crawling ACL Anthology, with {args.count} papers from {args.venue} {args.year}')
                crawl_acl(args.urlpath, args.venue, args.year, args.count)
            
        process_urls(args)
    else:
        print('Skipping crawling, parsing from txt file')
        process_urls(args)
