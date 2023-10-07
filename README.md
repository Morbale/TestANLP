# NLP from Scratch

## Installment
```bash
conda create --name nlpfromscratch python=3.8
conda activate nlpfromscratch
pip install -r requirements.txt
```

## Scraping
Uses [SciPDF](https://github.com/titipata/scipdf_parser) parser to parse the pdf. 
Open new terminal window and run following:
```bash
conda activate nlpfromscratch
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_lg
conda install openjdk=11
bash serve_grobid.sh
```
Go back to your original terminal window and run following, specifying the path to the ```.txt``` file that contains list of the pdf urls to be parsed. Not specifying the `filepath` argument will use `test_parsing.txt` to run it by default.
```bash
python parse_pdf.py --filepath <path to urls .txt file>
```