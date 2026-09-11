import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma


load_dotenv()

PDF_PATH = "data/Machine_Learning.pdf"


# 1. Load PDF
loader = PyPDFLoader(PDF_PATH)
documents = loader.load()

print("Number of pages:", len(documents))


# 2. Split documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

print("Number of chunks:", len(chunks))


# 3. Create OpenAI embeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)


# 4. Store embeddings in Chroma
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="chroma_db"
)

print("Documents stored successfully!")
