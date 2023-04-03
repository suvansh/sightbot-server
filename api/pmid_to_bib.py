# from https://github.com/zhuchcn/pubmed-bib/blob/master/pubmed_bib.py
import click
import requests
import json
import re

# Get a reference from PubMed database
def getReferences(ids):
    ids_str = ','.join([str(id) for id in ids])
    url_format = 'https://api.ncbi.nlm.nih.gov/lit/ctxp/v1/pubmed/'
    # Parameters for query
    queryParams = {
        'format': 'csl',
        'id': ids_str
    }
    # get a reference
    response = requests.get(url_format, params = queryParams).json()
    return response

# Format the reference to BibTex format
def formatReference(reference, use_short):
    title = reference['title'] if 'title' in reference.keys() else ''
    # convert <sub> and <sup>> to latex
    title = re.sub("<sub>(.+)</sub>", "$_{\\1}$", title)
    title = re.sub("<sup>(.+)</sup>", "$^{\\1}$", title)

    authors = reference['author'] if 'author' in reference.keys() else ''
    authorList = []
    for author in authors:
        if ('family' in author.keys()) and ('given' in author.keys()):
            authorList.append(author['family'] + ', ' + author['given'])
        elif ('family' in author.keys()) and ('given' not in author.keys()):
            authorList.append(author['family'])
        elif ('family' not in author.keys()) and ('given' in author.keys()):
            authorList.append(author['given'])
        else:
            continue

    journal_long = reference.get('container-title') or ''
    journal_short = reference.get('container-title-short') or ''
    volume = reference.get('volume') or ''
    page = reference.get('page') or ''
    
    if 'issued' in reference.keys():
        year = reference['issued']['date-parts'][0][0]
    elif 'epub-date' in reference.keys():
        year = reference['epub-date']['date-parts'][0][0]    

    ref_id = authors[0]["family"].lower() \
            if "family" in authors[0].keys() else authors[0]
    ref_id += str(year) + title.split(' ')[0].lower()

    output = f'''@article{{{ ref_id },
    title={{{title}}},
    author={{{' and '.join(authorList)}}},
    {"journal-long" if use_short else "journal"}={{{journal_long}}},
    {"journal" if use_short else "journal-short"}={{{journal_short}}},
    volume={{{volume}}},
    pages={{{page}}},
    year={{{year}}},
    PMID={{{id}}}
}}
'''
    return output

def get_bibtex_from_pmids(pmids):
    references = getReferences(pmids)
    if (isinstance(references, list) and len(references) == 0) or (isinstance(references, dict) and 'status' in references.keys() and references['status'] == 'error'):
        return ''
    return '\n'.join([formatReference(reference, True) for reference in references])
