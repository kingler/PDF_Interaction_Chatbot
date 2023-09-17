# Import the necessary libraries and modules
import os
import hashlib
import glob
from langchain.document_loaders import PyPDFLoader
from langchain.chat_models import ChatOpenAI
# from langchain import OpenAI
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from modify import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

from rich import print
from rich.console import Console
from rich.table import Table

''' pip install the requirements.txt file to install the necessary libraries and modules. and upgrade the libraries by running the following command in the terminal:
pip install -r requirements.txt --upgrade'''

# define your openai api key in your environment variables as "OPENAI_API_KEY"
# or define it here
# openai.api_key = "YOUR_OPENAI_API_KEY"

# Create a Console instance for custom styling
console = Console()

# Define a class for handling chats with PDF documents and summarization
class Chat_With_PDFs_and_Summarize:
    # Initialize the class with model information
    def __init__(self, model_name="gpt-3.5-turbo", temperature=0):
        # Initialize ChatOpenAI for summarization and chat
        self.llm_summarize = ChatOpenAI(model_name=model_name, temperature=temperature)
        self.llm_chat = ChatOpenAI(model_name=model_name, temperature=temperature)
        
        # Initialize variables to store document, pages and index information
        self.loader = None
        self.pages = None
        self.docs = None
        self.db_index = None
        self.embeddings = OpenAIEmbeddings()
        self.persist_directory = "db_index"
        self.doc_hash = None

    # Load a PDF document and split it into pages
    def load_document(self, file_path, page_range=None):
        # Load the document using PyPDFLoader
        self.loader = PyPDFLoader(file_path)
        # Split the document into pages
        self.pages = self.loader.load_and_split()
        
        # If a page range is specified, only load those pages
        if page_range:
            new_docs = [Document(page_content=t.page_content) for t in self.pages[page_range[0]:page_range[1]]]
        else:
            new_docs = [Document(page_content=t.page_content) for t in self.pages]

        # Calculate a hash for the loaded documents
        new_hash = hashlib.md5(''.join([doc.page_content for doc in new_docs]).encode()).hexdigest()

        # Check whether the database index already exists
        if not os.path.exists(self.persist_directory):
            os.makedirs(self.persist_directory)

        if os.path.exists(os.path.join(self.persist_directory, 'doc_hash.txt')):
            with open(os.path.join(self.persist_directory, 'doc_hash.txt'), 'r') as f:
                stored_hash = f.read().strip()
            
            if new_hash == stored_hash:
                # Load an existing index from disk
                print("Loading the index from the disk...")
                self.db_index = Chroma(persist_directory=self.persist_directory, embedding_function=self.embeddings)
            else:
                self.docs = new_docs
                self.doc_hash = new_hash
                # Create a new index
                print("Creating a new index...")
                self.db_index = Chroma.from_documents(self.docs, self.embeddings, persist_directory=self.persist_directory)

                # Save the new hash in the index directory
                with open(os.path.join(self.persist_directory, 'doc_hash.txt'), 'w') as f:
                    f.write(self.doc_hash)
        else:
            self.docs = new_docs
            self.doc_hash = new_hash
            # Create a new index
            print("Creating a new index...")
            self.db_index = Chroma.from_documents(self.docs, self.embeddings, persist_directory=self.persist_directory)

            # Save the new hash in the index directory
            with open(os.path.join(self.persist_directory, 'doc_hash.txt'), 'w') as f:
                f.write(self.doc_hash)

    # Generate a summary of the loaded document
    def summarize(self, chain_type="map_reduce"):
        if not self.docs:
            raise ValueError("No document loaded. Please load a document first using `load_document` method.")
        
        # Load the summarization chain and run it on the loaded documents
        chain = load_summarize_chain(self.llm_summarize, chain_type=chain_type)
        return chain.run(self.docs)

    # Print test pages for reference
    def print_test_pages(self, page_indices):
        if not self.pages:
            raise ValueError("No document loaded. Please load a document first using `load_document` method.")
        
        for index in page_indices:
            print(f"Page {index}:")
            print(self.pages[index])
            print("\n")

    # Ask a question and get an answer from the model
    def ask_question(self, query):
        if not self.db_index:
            raise ValueError("No document index. Please load a document first using `load_document` method.")
        
        # Initialize the RetrievalQA object and run the query
        qa = RetrievalQA.from_chain_type(
            llm=self.llm_chat,
            chain_type="stuff",
            retriever=self.db_index.as_retriever()
        )
        
        # Get the response from the model based on the query
        response = qa.run(query)
        return response

# Main script starts here
if __name__ == "__main__":
    chat = Chat_With_PDFs_and_Summarize()

    # Search the 'documents' folder for PDF files
    documents_list = glob.glob("documents/*.pdf")

    # Prepare a Rich Table for displaying available PDF documents
    docs_table = Table(title="Available documents", show_header=True, header_style="bold magenta")
    docs_table.add_column("Index", justify="right", style="dim")
    docs_table.add_column("Document", justify="right", style="bright_yellow")

    for index, document in enumerate(documents_list):
        docs_table.add_row(str(index), document)

    console.print(docs_table)

    # Get user input for selecting the document to load
    document_index = int(console.input("Enter the index of the document you want to load: "))
    selected_document = documents_list[document_index]

    # Get user input for the range of pages to index
    page_range_option = console.input("Select pages to index: (A)ll pages or (C)ustom range: ").strip()

    if page_range_option.lower() == "c":
        start_page = int(console.input("Start page (0-indexed): "))
        end_page = int(console.input("End page: "))
        page_range = (start_page, end_page)
    elif page_range_option.lower() == "a":
        page_range = None

    # Load the selected document with the specified page range
    chat.load_document(selected_document, page_range=page_range)

    console.rule("Document loaded")

    # Get user input for generating a summary
    summary_option = console.input("Do you want to generate a summary? (Y)es or (N)o: ").strip()

    if summary_option.lower() == "y":
        summary = chat.summarize()
        console.print(f"\nSummary: {summary}", style="bold")
        console.print("\n")

    # Ask questions and get answers in a loop
    while True:
        query = console.input("Query: ")
        answer = chat.ask_question(query)
        console.print(f"Answer: {answer}", style="green")