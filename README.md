# NLP from Scratch

## Installment
```bash
conda create --name nlpfromscratch python=3.8
conda activate nlpfromscratch
pip install -r requirements.txt
```

## Scraping
Uses [SciPDF](https://github.com/titipata/scipdf_parser) parser to parse the pdf. PDF crawler was built referring to [ACL-Anthology-Crawler](https://github.com/srhthu/ACL-Anthology-Crawler/tree/main). 

Open new terminal window and run following:
```bash
conda activate nlpfromscratch
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_lg
conda install openjdk=11    # download arm64 version depending on your machine
bash serve_grobid.sh
```
Go back to your original terminal window and run following command. 
```bash
python parse_pdf.py
```
Several arguments to specify:
- If you want to start by crawling: include `--crawl`, then specify:
    - `--urlpath`: path to save crawled url txt file
    - `--venue`: venue to crawl papers from. Refer to [ACL Anthology](https://aclanthology.org/) for possible options.
    - `--year`: year to crawl papers from. If the selected year is not available, it will raise an exception. Currently supports until 2021.
    - `--count`: number of papers to collect
    - `--volume`: defaults to the main conference of specified venue, if not specified. (i.e. [long papers](https://aclanthology.org/volumes/2023.acl-long/) for ACL)
- If you already have crawled url path containing list of pdf links, specify:
    - `--urlpath`: path to save crawled url txt file

Refer to `python parse_pdf.py --h` for more information.