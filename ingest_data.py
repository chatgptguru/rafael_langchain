from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI, VectorDBQA
from langchain.document_loaders import DirectoryLoader
from langchain.vectorstores import FAISS
import os
import pickle

with open('openaiapikey.txt', 'r') as infile:
    os.environ['OPENAI_API_KEY'] = infile.read()

loader = DirectoryLoader('./PaulGrahamEssaySmall/', glob='**/*.txt')
documents = loader.load()
vectorstore = FAISS.from_documents(documents, OpenAIEmbeddings())
with open("vectorstore.pkl", "wb") as f:
    pickle.dump(vectorstore, f)

