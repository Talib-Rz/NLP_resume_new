import os
import streamlit as st
import fitz  # PyMuPDF
import re
import nltk
import requests
from dotenv import load_dotenv
from nltk.corpus import stopwords

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


# Setup
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
load_dotenv()

st.set_page_config(page_title="Resume Search", layout="wide")
st.title("NLP Resume Search")

# Preprocess text
def preprocess(text):
    text = re.sub(r'\W+', ' ', text.lower())
    return ' '.join([word for word in text.split() if word not in stop_words])

# Extract text from PDF
def extract_text_from_pdf(uploaded_file):
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        return "\n".join([page.get_text() for page in doc])

filenames = []
texts = []

# Load embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Upload PDFs
uploaded_files = st.file_uploader("Upload Resumes (PDFs)", type="pdf", accept_multiple_files=True)

vector_store = None

if uploaded_files:
    with st.spinner("Processing and indexing resumes..."):
        documents = []

        for file in uploaded_files:
            text = extract_text_from_pdf(file)
            cleaned = preprocess(text)

            # Create LangChain Document with metadata
            doc = Document(page_content=cleaned, metadata={"filename": file.name})
            documents.append(doc)
            texts.append(cleaned)
            filenames.append(file.name)

        # Correct way to create FAISS vector store
        vector_store = FAISS.from_documents(documents, embedding_model)

        st.success(f"{len(uploaded_files)} resume(s) indexed successfully!")

# Query input
query = st.text_input("Enter your search query (e.g., 'Python developer with NLP experience')")

# Search and show results
if query and vector_store is not None:
    with st.spinner("Searching for the most relevant resumes..."):
        results = vector_store.similarity_search_with_score(query, k=len(uploaded_files))

        if results:
            st.subheader("Matching Resumes:")

            for i, (doc, score) in enumerate(results, 1):
                st.markdown(f"### {i}. `{doc.metadata['filename']}`")
                st.markdown(f"**Similarity Score:** `{score:.4f}`")
                st.markdown(f"**Resume Content Snippet:**\n\n```text\n{doc.page_content}...\n```")
                st.markdown("---")

        else:
            st.warning("No matching resumes found.")



# Chat with selected resume
st.subheader("Let's gossip with a Resume:")

if filenames:
    selected_resume = st.selectbox("Select a resume to chat with", filenames)
    selected_index = filenames.index(selected_resume)
    resume_text = texts[selected_index]

    user_question = st.chat_input("Ask a question about this resume")

    if user_question:
        with st.spinner("Thinking with LLaMA 3 via Groq..."):
            def ask_groq_llm(prompt):
                headers = {
                    "Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": "llama3-8b-8192",
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant that reads resumes and answers questions."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.3,
                    "max_tokens": 500
                }
                response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
                return response.json()["choices"][0]["message"]["content"]

            prompt = f"Here is the resume:\n\n{resume_text}\n\nQuestion: {user_question}\nAnswer:"
            answer = ask_groq_llm(prompt)
            st.markdown("**Answer:**")
            st.write(answer)

     
     
    # if user_question: 
    #     with st.spinner("thinking with LLama 3 via Groq..."):  
    
    #       llm = ChatGroq(model="llama-3.1-8b-instant",temperature=0.2)
     
    #     prompt = f"""
    #     You are a helpful assistant that reads resumes and answers questions.

    #     Here is the resume:

    #     {resume_text}

    #     Question: {user_question}
    #     Answer:
    #     """

    #     # Invoke the model
    #     answer = llm.invoke(prompt)

    #     # Display result in Streamlit
    #     st.markdown("Answer:")
    #     st.write(answer)
