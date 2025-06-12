import streamlit as st
import os
import time
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFDirectoryLoader

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("HR_CHATBOT_GROQ_KEY")

if not groq_api_key:
    st.error("❌ GROQ API key not set in .env (HR_CHATBOT_GROQ_KEY)")
    st.stop()

st.title("HR Chatbot (TF-IDF + Groq LLaMA3)")

# Initialize LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-70b-8192")

# Prompt Template
prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the context provided.
If the answer is not in the context, say "I don't have enough information."

<context>
{context}
</context>

Question: {input}
""")

# TF-IDF Embedding
def vector_embedding():
    st.info("Processing documents...")
    with st.spinner("Loading and embedding..."):
        try:
            hr_docs_path = r"C:/Users/Meghamary.vinu/policy_docs"  # Update this path as needed

            if not os.path.isdir(hr_docs_path):
                st.error(f"📁 Folder not found: {hr_docs_path}")
                return

            loader = PyPDFDirectoryLoader(hr_docs_path)
            docs = loader.load()

            if not docs:
                st.warning(f"⚠️ No documents found in: {hr_docs_path}")
                return

            st.write(f"✅ Loaded {len(docs)} documents.")
            for d in docs:
                st.write(f"📄 {d.metadata.get('source', 'Unknown Source')}")

            # Split into chunks
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            final_documents = splitter.split_documents(docs)

            # Extract only text for TF-IDF
            texts = [doc.page_content for doc in final_documents]

            vectorizer = TfidfVectorizer()
            X = vectorizer.fit_transform(texts)

            # Store full documents (not text only)
            st.session_state.documents = final_documents
            st.session_state.vectorizer = vectorizer
            st.session_state.tfidf_matrix = X

            st.success("✅ Documents embedded using TF-IDF!")

        except Exception as e:
            st.error(f"❌ Error while loading: {e}")
            st.session_state.documents = []
            st.session_state.vectorizer = None
            st.session_state.tfidf_matrix = None

# Button to trigger vector embedding
if st.button("📄 Load and Process HR Documents"):
    vector_embedding()

# Show status
if "tfidf_matrix" in st.session_state and st.session_state.tfidf_matrix is not None:
    st.success("✅ TF-IDF vector DB is ready!")
else:
    st.warning("⚠️ Please load and process documents first.")

# Input box
prompt1 = st.text_input("🔍 Ask a question from HR policy documents:")

# Handle user query
if prompt1:
    if "tfidf_matrix" in st.session_state and st.session_state.tfidf_matrix is not None:
        with st.spinner("🤖 Thinking..."):
            try:
                # Vectorize user query
                question_vec = st.session_state.vectorizer.transform([prompt1])
                scores = cosine_similarity(question_vec, st.session_state.tfidf_matrix)[0]
                top_idx = scores.argsort()[::-1][:3]  # Top 3 most relevant

                # Get top documents as Document objects (not just text)
                top_docs = [st.session_state.documents[i] for i in top_idx]

                # Use LLM with Document objects
                document_chain = create_stuff_documents_chain(llm, prompt)
                response = document_chain.invoke({"input": prompt1, "context": top_docs})

                # Display Answer - response is a string, not a dict
                st.write("**Answer:**")
                st.write(response)



            except Exception as e:
                st.error(f"❌ Error during answer generation: {e}")
                import traceback
                st.error(f"Full traceback: {traceback.format_exc()}")
    else:
        st.warning("⚠️ Please load documents first.")