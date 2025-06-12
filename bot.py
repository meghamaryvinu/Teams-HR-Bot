import streamlit as st
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFDirectoryLoader

# Get API key from Streamlit secrets or environment variables
try:
    groq_api_key = st.secrets["HR_CHATBOT_GROQ_KEY"]
except:
    try:
        from dotenv import load_dotenv
        load_dotenv()
        groq_api_key = os.getenv("HR_CHATBOT_GROQ_KEY")
    except:
        groq_api_key = None

if not groq_api_key:
    st.error("❌ GROQ API key not set. Please configure it in Streamlit Cloud secrets or .env file.")
    st.info("💡 In Streamlit Cloud, go to Settings → Secrets and add: HR_CHATBOT_GROQ_KEY = 'your_key_here'")
    st.stop()

st.title("Metayb HR Assistant")

# Initialize LLM
@st.cache_resource
def initialize_llm():
    return ChatGroq(groq_api_key=groq_api_key, model_name="llama3-70b-8192")

llm = initialize_llm()

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
@st.cache_data
def vector_embedding():
    """Cache the document processing to avoid reprocessing on every run"""
    try:
        # Use relative path for both local and cloud deployment
        hr_docs_path = "policy_docs"
        
        if not os.path.isdir(hr_docs_path):
            raise FileNotFoundError(f"Document folder not found. Expected: 'policy_docs' or '{hr_docs_path}'")

        loader = PyPDFDirectoryLoader(hr_docs_path)
        docs = loader.load()

        if not docs:
            raise ValueError(f"No documents found in: {hr_docs_path}")

        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_documents = splitter.split_documents(docs)

        # Extract only text for TF-IDF
        texts = [doc.page_content for doc in final_documents]

        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(texts)

        return {
            'documents': final_documents,
            'vectorizer': vectorizer,
            'tfidf_matrix': X,
            'doc_count': len(docs),
            'chunk_count': len(final_documents),
            'sources': [d.metadata.get('source', 'Unknown Source') for d in docs]
        }

    except Exception as e:
        st.error(f"❌ Error while loading documents: {e}")
        return None

# Auto-load documents on startup (silently)
if 'vector_data' not in st.session_state:
    # Load silently without showing spinner
    st.session_state.vector_data = vector_embedding()

# Only show error if documents failed to load
if not st.session_state.vector_data:
    st.error("❌ Failed to load documents. Please check your document folder and try refreshing the page.")
    st.info("💡 Make sure your PDF files are in the 'policy_docs' folder")
    st.stop()

# Input box
prompt1 = st.text_input("🔍 Ask a question about HR policies:", placeholder="e.g., What is the leave policy?")

# Handle user query
if prompt1:
    if st.session_state.vector_data:
        with st.spinner("🤖 Generating answer..."):
            try:
                data = st.session_state.vector_data
                
                # Vectorize user query
                question_vec = data['vectorizer'].transform([prompt1])
                scores = cosine_similarity(question_vec, data['tfidf_matrix'])[0]
                top_idx = scores.argsort()[::-1][:3]  # Top 3 most relevant

                # Check if we have relevant results
                if scores[top_idx[0]] < 0.1:  # Low similarity threshold
                    st.warning("⚠️ I couldn't find very relevant information for your question. The answer below is based on the most similar content I found.")

                # Get top documents as Document objects
                top_docs = [data['documents'][i] for i in top_idx]

                # Use LLM with Document objects
                document_chain = create_stuff_documents_chain(llm, prompt)
                response = document_chain.invoke({"input": prompt1, "context": top_docs})

                # Display Answer
                st.write("**Answer:**")
                st.write(response)



            except Exception as e:
                st.error(f"❌ Error generating answer: {e}")
                st.error("Please try rephrasing your question or contact support.")
    else:
        st.error("❌ Documents not loaded. Please refresh the page.")

# Footer
st.markdown("---")
st.markdown("💡 **Tip**: Ask specific questions about HR policies, procedures, benefits, or company guidelines for best results.")
