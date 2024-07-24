# import chain
# import connect_confluance
# import Generate_embedding
# import retrival
# from langchain_community.document_loaders import UnstructuredHTMLLoader
# from langchain_community.chat_models import ChatOllama


# local_path='data.html'

# if local_path:
#     #load pdf data
#     # loader = UnstructuredPDFLoader(file_path=local_path)
#     #load urldata
#     # loader = WebBaseLoader("https://www.miniorange.com/about_us")
#     data=''''''
#     html_data=connect_confluance.gate_data_confluance()
#     for i in html_data:
#         with open('data.html', "a") as file:
#                 print(i)
#                 file.write(i)

#     loader = UnstructuredHTMLLoader(local_path)
#     data = loader.load()
# vector_db_new =Generate_embedding.embedding(data)
# local_model = "llama3"
# llm = ChatOllama(model=local_model)
# retrival=retrival.retrival(vector_db_new,llm)
# chains=chain.chain(retrival,llm)

# while True:
#     print(chains.invoke(input("Hi,How can I Help you!:-")))

import streamlit as st
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import WebBaseLoader

import chain
import connect_confluance
import Generate_embedding
import retrival

def main():
    st.title("Offline Confluence ChatBot with on premises database !")
    
    local_path = 'data.html'
    
    # Streamlit inputs for API, username, and password
    api = st.text_input('Enter the API URL:', 'http://localhost:8090/rest/api/space/TEST/content/page?expand=body.storage')
    username = st.text_input('Enter your Username:','chaitanya')
    password = st.text_input('Enter your Password:', type='password')
    # chains = None
    if st.button('Load Data'):
        pass
        # try:
        # Load HTML data from Confluence
        html_data = connect_confluance.gate_data_confluance(api, username, password)
        
        # Write HTML data to a local file
        with open(local_path, "w") as file:
            for i in html_data:
                file.write(i)
        
        # Load data using UnstructuredHTMLLoader
        loader = UnstructuredHTMLLoader(local_path)
    # loader = WebBaseLoader("https://en.wikipedia.org/wiki/Narendra_Modi")
        data = loader.load()
    
        # # Generate embeddings
        vector_db_new = Generate_embedding.embedding(data)
        st.success("Data loaded successfully!")
    vector_db_new1 = Chroma(persist_directory="iam_pam",embedding_function=OllamaEmbeddings(model="nomic-embed-text"),collection_name="local-rag")
    # vector_db_new2 = Chroma(persist_directory="iam_pam_db",embedding_function=OllamaEmbeddings(model="nomic-embed-text"),collection_name="local-rag")

    # Initialize the LLM
    local_model = "llama3"
    llm = ChatOllama(model=local_model)
    
    # Initialize retrieval
    retrival_instance1= retrival.retrival(vector_db_new1, llm)
    # retrival_instance2= retrival.retrival(vector_db_new2, llm)



    
    # Initialize the chain
    chains = chain.chain(retrival_instance1, llm)
    

    
    # except requests.exceptions.ConnectionError as e:
    #     st.error(f"Connection error: {e}. Please ensure the Confluence server is running and accessible.")
    
    # except Exception as e:
    #     st.error(f"An error occurred: {e}")

    # Streamlit input and output for chatbot interaction
    user_input = st.text_input("Hi, How can I help you?", "")
    
    if user_input:
        try:
            response = chains.invoke(user_input)
            st.write("Response:")
            st.write(response)
        except Exception as e:
            st.error(f"An error occurred while processing your request: {e}")

if __name__ == "__main__":
    main()
