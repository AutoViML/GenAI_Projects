##############################################################################
#############       This is not an Officially Supported Google Product! ######
##############################################################################
#Copyright 2025 Google LLC                                              ######
#                                                                       ######
#Licensed under the Apache License, Version 2.0 (the "License");        ######
#you may not use this file except in compliance with the License.       ######
#You may obtain a copy of the License at                                ######
#                                                                       ######
#    https://www.apache.org/licenses/LICENSE-2.0                        ######
#                                                                       ######
#Unless required by applicable law or agreed to in writing, software    ######
#distributed under the License is distributed on an "AS IS" BASIS,      ######
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.#####
#See the License for the specific language governing permissions and    ######
#limitations under the License.                                         ######
##############################################################################
###########           Vertex RAG Comparator with Judge Model            ######
###########             Developed by Ram Seshadri                       ######
###########             Last Updated:  Feb 2025                         ######
##############################################################################
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

from google import genai
from google.genai import types

def create_gemini_client():
    """Create client with automatic API selection and auth handling"""
    if os.environ.get("GOOGLE_VERTEXAI", "").lower() == "true":
        # Vertex AI configuration
        project = os.getenv("PROJECT_ID")
        location = os.getenv("LOCATION")
       
        if not project or not location:
            raise ValueError("VertexAI requires GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION environment variables")
       
        return genai.Client(
            vertexai=True,
            project=project,
            location=location,
            http_options=types.HttpOptions(api_version='v1')
        )
    else:
        # Gemini Developer API configuration
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Gemini API requires GOOGLE_API_KEY environment variable")
       
        return genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(api_version='v1alpha')
        )

def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text



def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain(context, question):

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    #model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    generation_config=genai.types.GenerationConfig(temperature=0.3, max_output_tokens=2048)

    # Initialize client with automatic auth detection
    client = create_gemini_client()
    
    # Start streaming chat session
    chat_session = client.chats.create(model='gemini-2.0-flash-001')

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"]).format(context=context, question=question)
    response = chat_session.send_message_stream(prompt)
    
    return response



def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain(docs, user_question)
    response = ''
    try:
        for chunk in chain:
            chunk_text = chunk.text
            st.write(chunk_text)
            response += chunk_text
    except Exception as e:
        print(f"\nError: {str(e)}")    

    st.write("Usage Metadata: token count = ", chunk.usage_metadata.total_token_count)
    return response




def main():
    st.set_page_config("Demo 0: Chat with PDF")
    st.header("Demo 0: Chat with PDF using Gemini💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        #pdf_docs = ['./recipes/asian_recipes1.pdf','./recipes/asian_recipes2.pdf','./recipes/asian_recipes3.pdf', './recipes/asian_recipes4.pdf']
        if st.button("Submit Docs & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")


if __name__ == "__main__":
    main()
