from flask import Flask, request, jsonify
from flask_restful import Api, Resource, reqparse
from flask_cors import CORS

from enum import Enum

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import OnlinePDFLoader, PagedPDFSplitter
from langchain.docstore.document import Document
from langchain import text_splitter
from langchain.vectorstores import Chroma
from langchain import embeddings
from langchain.llms import OpenAI
from langchain.chains import ChatVectorDBChain
from langchain.chains import LLMChain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains.chat_vector_db.prompts import CONDENSE_QUESTION_PROMPT

import xml.etree.ElementTree as ET

import sys
import requests

with open("/Users/suvansh/.openai.key") as f:
    openai_api_key = f.read().strip()

CHUNK_SIZE = 100
CHUNK_OVERLAP = 20
NUM_CHUNKS = 15


class DocType(Enum):
    FILE_PDF = 1
    ONLINE_PDF = 2
    TEXT = 3


def parse_pubmed_json(doc_json, pmid):
    documents = []
    pmcid = doc_json["documents"][0]["id"]
    passages = doc_json["documents"][0]["passages"]
    lead_author = doc_json["documents"][0]["passages"][0]["infons"]["name_0"].split(";")[0][8:]  # 8: to remove "Surname:"
    year = doc_json["date"][:4]  # get year
    for passage in passages:
        if (doc_type := passage["infons"]["type"].lower()) in ["ref", "front"]:
            continue  # skip references
        elif "table" in doc_type or "caption" in doc_type or "title" in doc_type:
            continue  # skip tables, captions, titles
        if (section_type := passage["infons"]["section_type"].lower()) == "auth_cont":
            continue
        citation = f"({lead_author} {year} - {pmid})"  # create citation; eg (Smith 2021 - 12345678)
        documents.append(Document(page_content=passage["text"],
                                  metadata={
                                    "pmcid": pmcid,
                                    "pmid": pmid,
                                    "offset": passage["offset"],
                                    "section_type": section_type,
                                    "type": doc_type,
                                    "source": citation}))
    return documents


def get_docs_from_file(file_: str, mode: DocType):
    """
    Get LangChain Document objects from a file,
    either a PDF (mode in [DocType.FILE_PDF, DocType.ONLINE_PDF])
    or a PubMed ID (mode == DocType.TEXT).
    """
    if mode == DocType.FILE_PDF:
        loader = PagedPDFSplitter(file_)
        docs = loader.load_and_split()
    elif mode == DocType.ONLINE_PDF:
        loader = OnlinePDFLoader(file_)
        docs = loader.load()
    elif mode == DocType.TEXT:
        # _file is pmid or pmcid
        req_url = f"https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/{file_}/unicode"
        try:
            doc_json = requests.get(req_url).json()
            docs = parse_pubmed_json(doc_json, file_)
        except:
            docs = None
            print(f"Error with {file_}")
    return docs


def split_docs(docs, splitter_type=text_splitter.TokenTextSplitter, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """
    Split a list of LangChain Document objects into chunks.
    """
    splitter = splitter_type(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs_split = splitter.split_documents(docs)
    return docs_split


def get_pubmed_results_old(query, num_results=30):
    """Get PubMed results"""
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&retmode=json&sort=relevance&retmax={num_results}&term=(pubmed%20pmc%20open%20access[filter])+{query}"
    response = requests.get(url)  # make API call
    pm_ids = response.json()['esearchresult']['idlist']  # get list of ids
    print(f"Found {len(pm_ids)} results for query '{query}'")
    return pm_ids


def get_abstracts_from_query(query, num_results=30):
    pmids = get_pubmed_results_old(query, num_results=num_results)
    docs = get_abstracts_from_pmids(pmids)
    return docs, pmids


def get_docs_from_query(query, mode="pubmed", num_results=30):
    """Get documents from a query"""
    if mode == "pubmed":
        pm_ids = get_pubmed_results_old(query, num_results=num_results)
        docs = []
        for pm_id in pm_ids:
            article_docs = get_docs_from_file(pm_id, DocType.TEXT)
            if article_docs:
                docs.extend(article_docs)
        return docs, pm_ids
    elif mode == "google":
        pass


def get_abstracts_from_pmids(pmids):
    def get_nexted_xml_text(element):
        """ Used for extracting all text from abstract, even in the presence of nested tags """
        if element.text is not None:
            text = element.text.strip()
        else:
            text = ''
        for child in element:
            child_text = get_nexted_xml_text(child)
            if child_text:
                text += ' ' + child_text
        return text

    pmids_str = ','.join(pmids)
    req_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={pmids_str}&rettype=abstract"
    response = requests.get(req_url)
    xml_root = ET.fromstring(response.content)
    articles = xml_root.findall("PubmedArticle")
    docs = []
    for pmid_, article in zip(pmids, articles):
        if not article.find("MedlineCitation").find("Article").find("Abstract"):
            print("No abstract found")
            continue
        try:
            pmid = article.find("MedlineCitation").find("PMID").text
            year = article.find("MedlineCitation").find("DateCompleted").find("Year").text
            author = article.find("MedlineCitation").find("Article").find("AuthorList").find("Author").find("LastName").text
            citation = f"({author} {year} - {pmid})"
            abstract_node = article.find("MedlineCitation").find("Article").find("Abstract").find("AbstractText")
            abstract = get_nexted_xml_text(abstract_node)
            docs.append(Document(page_content=abstract, metadata={"source": citation, "pmid": pmid}))
        except:
            print(f"Error parsing article {pmid_}")
    print(f"Parsed {len(docs)} documents from {len(articles)} abstracts.")
    return docs


""" Flask setup """
app = Flask(__name__)
api = Api(app)
#CORS(app, origins=['http://localhost:3000'])
CORS(app, resources={r"/api/*": {"origins": "*"}})
parser = reqparse.RequestParser()
parser.add_argument('messages')
parser.add_argument('question')
parser.add_argument('num_articles')
parser.add_argument('input')


class ChatGPSeeAPI(Resource):
    def post(self):
        args = parser.parse_args()
        # Invoke your text processing script here
        # processed_text = scripts.text_processor(args['text'])
        docs, pmids = get_abstracts_from_query(args['input'], num_results=args['num_articles'])
        docs_split = split_docs(docs)
        
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_api_key)
        question_generator = LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT)
        # Below, "with_sources" results in answer containing source references
        # "map_reduce" results in answer being a summary of the source references
        doc_chain = load_qa_with_sources_chain(llm, chain_type="stuff")
        vectorstore = Chroma.from_documents(docs_split, embeddings.HuggingFaceEmbeddings(), ids=[doc.metadata["source"] for doc in docs_split])
        print("Built Chroma vector store.")
        chain = ChatVectorDBChain(
            vectorstore=vectorstore,
            question_generator=question_generator,
            combine_docs_chain=doc_chain,
            return_source_documents=True,  # results in referenced documents themselves being returned
            top_k_docs_for_context=min(NUM_CHUNKS, len(docs_split))
        )
        print("Built Chain.")
        vectordbkwargs = {} # {"search_distance": 0.9}  # threshold for similarity search (setting this may reduce hallucinations)
        chat_history = [("You are a helpful chatbot. You are to explain abbreviations and symbols before using them. Please provide lengthy, detailed answers. If the documents provided are insufficient to answer the question, say so.",
                         "Understood. I am a helpful chatbot. I will explain abbreviations and symbols before using them and provide detailed answers. If the documents provided are insufficient to answer the question, I will say so.")]
        result = chain({"question": args.question, "chat_history": chat_history,
              "vectordbkwargs": vectordbkwargs})
        print("Received answer.")
        chat_history.append((args.question, result["answer"]))
        
        citations = [doc.metadata["source"] for doc in result["source_documents"]]
        response = {"answer": result["answer"], "citations": citations}
        print(result["answer"])
        return response, 200
    def get(self):
        response = {'data': "Hello, world!"}
        return response, 200

api.add_resource(ChatGPSeeAPI, '/api/chat')

#if __name__ == '__main__':
#    app.run(host='0.0.0.0', debug=True)

