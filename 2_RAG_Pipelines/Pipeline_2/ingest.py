from langchain.document_loaders import PyPDFLoader, DirectoryLoader, PDFMinerLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain.embeddings import SentenceTransformerEmbeddings 
from langchain.vectorstores import Chroma 
import os 
from constants import CHROMA_SETTINGS
import torch

device = torch.device('cuda:0')

persist_directory = "embeddings_db_chroma"
# persist_directory = "dpr_embedding_db_chroma"
# persist_directory = "dpr_embedding_db_chroma_512_sized_chunks"

def main():
    for root, dirs, files in os.walk("docs"):
        for file in files:
            if file.endswith(".pdf"):
                print(file)
                loader = PyPDFLoader(os.path.join(root, file))
    documents = loader.load()
    print("splitting into chunks")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=400)
    texts = text_splitter.split_documents(documents)
    #create embeddings here
    print("Loading sentence transformers model")
    model_kwargs = {'device': 'cuda:0'}  # specify GPU device
    embeddings = SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2", model_kwargs=model_kwargs)
    # embeddings = SentenceTransformerEmbeddings(model_name="facebook-dpr-ctx_encoder-single-nq-base", model_kwargs=model_kwargs)
    
    #create vector store here
    print(f"Creating embeddings. May take some minutes...")
    db = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory, client_settings=CHROMA_SETTINGS)
    db.persist()
    db=None 

    print(f"Ingestion complete! You can now run privateGPT.py to query your documents")

if __name__ == "__main__":
    main()