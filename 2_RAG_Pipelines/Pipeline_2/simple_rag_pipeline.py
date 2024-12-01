import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline, AutoModelForCausalLM
import torch
from langchain.document_loaders import PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from constants import CHROMA_SETTINGS

# Set device to CPU (change to GPU if available)
device = torch.device('cuda:0')

# Initialize the model checkpoint
checkpoint = "google/gemma-2-2b-it"
print(f"Checkpoint path: {checkpoint}")

# Load tokenizer and base model
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
base_model = AutoModelForCausalLM.from_pretrained(
    checkpoint,
    device_map=device,
    torch_dtype=torch.float32
)

# Set the persistent directory for embeddings
persist_directory = "embeddings_db_chroma"


# Initialize the LLM pipeline
def llm_pipeline():
    pipe = pipeline(
        'text-generation',
        model=base_model,
        tokenizer=tokenizer,
        # max_length=4096,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.2,
        top_p=0.95,
        device_map=[0]
    )
    local_llm = HuggingFacePipeline(pipeline=pipe)
    return local_llm

# Function to create the RAG-based RetrievalQA system
def qa_llm():
    llm = llm_pipeline()
    model_kwargs = {'device': 'cuda:0'}  # specify GPU device
    embeddings = SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2", model_kwargs=model_kwargs)
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    # retriever = db.as_retriever()
    retriever = db.as_retriever(search_kwargs={"k":5})
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    return qa

# Function to process each query and get the answer
def process_answer(instruction):
    qa = qa_llm()
    generated_text = qa(instruction)
    answer = generated_text['result']
    return answer

# Function to read queries from the Excel file and save the responses
def process_excel_queries(file_path, output_file_path):
    # Load the Excel file
    df = pd.read_excel(file_path)
    
    # Assuming the first column contains queries
    #queries = df['question']  # Adjust this based on your actual column name
    queries = df['Query']

    # Prepare a list to store results
    results = []
    i = 0
    # Process each query and generate answers
    for query in queries:
        print(f"{i} Processing query: {query}")
        answer = process_answer({'query': query})
        print(f"Answer: {answer}")
        results.append(answer)
        i+=1

    # Add the responses to the dataframe and save to a new Excel file
    df['response'] = results
    df.to_excel(output_file_path, index=False)
    print(f"Responses saved to {output_file_path}")

if __name__ == "__main__":
    # First, we ingest the data and build the embeddings (skip this if embeddings already exist)
    # data_ingestion()

    # # Process textual queries and save responses to 'abstr_textual_TESTSET_RESPONSES.xlsx'
    # print("Processing textual queries from 'textual_TESTSET_FINAL.xlsx'...")
    # process_excel_queries("textual_TESTSET_FINAL.xlsx", "abstr_textual_TESTSET_RESPONSES.xlsx")

    # # Process tabular queries and save responses to 'abstr_tabular_TESTSET_RESPONSES.xlsx'
    # print("Processing tabular queries from 'tabular_TESTSET_FINAL.xlsx'...")
    # process_excel_queries("tabular_TESTSET_FINAL.xlsx", "abstr_tabular_TESTSET_RESPONSES.xlsx")
    # print("Processing textual queries from 'tabular_TESTSET_FINAL.xlsx'...")
    # process_excel_queries("tabular_TESTSET_FINAL.xlsx", "tabular_responses_gemma_better.xlsx")
    # print("Processing vague queries from 'final_vague_testset.xlsx'...")
    # process_excel_queries("final_vague_testset.xlsx", "vague_testset_langchain.xlsx")
    # print(process_answer({'query': "What are hard core courses?"}))
    # print(process_answer({'query': "What are soft core courses?"}))
    query_ip = input("Enter your query for the handbook: ")
    print(process_answer({'query': query_ip}))
