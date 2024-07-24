from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma




def embedding(data):

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
    chunks = text_splitter.split_documents(data)
    embedding=OllamaEmbeddings(model="nomic-embed-text",show_progress=True)

    vector_db_new = Chroma.from_documents(
        documents=chunks, 
        embedding=OllamaEmbeddings(model="nomic-embed-text",show_progress=True),
        collection_name="local-rag",
        persist_directory="iam_pam"
    )

    vector_db_new.persist()
    return  vector_db_new 