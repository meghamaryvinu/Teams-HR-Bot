import streamlit as st
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFDirectoryLoader

# Import MultiQueryRetriever and necessary components
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from typing import List, Any
from pydantic import Field

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

st.title("Metayb HR Assistant") # Removed the tech stack from the title

# Initialize LLM
@st.cache_resource
def initialize_llm():
    return ChatGroq(groq_api_key=groq_api_key, model_name="llama3-70b-8192")

llm = initialize_llm()

# Prompt Template - MODIFIED FOR BETTER CONFIDENCE AND INFERENCE
prompt = ChatPromptTemplate.from_template("""
You are an expert HR assistant. Your goal is to provide comprehensive and helpful answers based on the HR policy context provided.

**Instructions:**
1.  **Strictly use only the provided context.** Do not bring in outside knowledge.
2.  If the user asks for information that can be *calculated or derived* from the context (e.g., monthly from yearly data, total from components), please perform that calculation or derivation and present the answer confidently.
3.  If the exact answer is not explicitly stated, but related information is available, summarize or synthesize that related information clearly.
4.  If the information is genuinely not present or cannot be inferred/calculated from the context, then and only then state: "I don't have enough information on that specific detail in the provided policies."

<context>
{context}
</context>

Question: {input}
""")

# Custom TF-IDF Retriever Class
class TFIDFRetriever(BaseRetriever):
    """
    A custom LangChain-compatible retriever using TF-IDF for document similarity.
    """
    documents: List[Document] = Field(exclude=True)
    vectorizer: TfidfVectorizer = Field(exclude=True)
    tfidf_matrix: Any = Field(exclude=True)

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        if self.tfidf_matrix is not None and len(self.documents) > 0:
            try:
                question_vec = self.vectorizer.transform([query])
                scores = cosine_similarity(question_vec, self.tfidf_matrix)[0]
                top_idx = scores.argsort()[::-1][:3]  # Top 3 most relevant

                relevant_docs = []
                for i in top_idx:
                    relevant_docs.append(self.documents[i])
                return relevant_docs
            except Exception as e:
                print(f"Error during TFIDF retrieval: {e}")
                return []
        return []

# TF-IDF Embedding
@st.cache_data
def vector_embedding():
    """Cache the document processing to avoid reprocessing on every run"""
    try:
        hr_docs_path = "policy_docs"
        
        if not os.path.isdir(hr_docs_path):
            raise FileNotFoundError(f"Document folder not found. Expected: 'policy_docs' or '{hr_docs_path}'")

        loader = PyPDFDirectoryLoader(hr_docs_path)
        docs = loader.load()

        if not docs:
            raise ValueError(f"No documents found in: {hr_docs_path}")

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_documents = splitter.split_documents(docs)

        texts = [doc.page_content for doc in final_documents]
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(texts)

        custom_tfidf_retriever = TFIDFRetriever(
            documents=final_documents,
            vectorizer=vectorizer,
            tfidf_matrix=X
        )

        return {
            'custom_tfidf_retriever': custom_tfidf_retriever,
            'doc_count': len(docs),
            'chunk_count': len(final_documents),
            'sources': [d.metadata.get('source', 'Unknown Source') for d in docs]
        }

    except Exception as e:
        st.error(f"❌ Error while loading documents: {e}")
        return None

# Auto-load documents on startup (silently)
if 'vector_data' not in st.session_state:
    st.session_state.vector_data = vector_embedding()
    # Removed the success and info messages here
    # if st.session_state.vector_data:
    #     st.success(f"✅ Loaded {st.session_state.vector_data['doc_count']} documents, split into {st.session_state.vector_data['chunk_count']} chunks.")
    #     st.info("Sources: " + ", ".join(set(st.session_state.vector_data['sources'])))

if not st.session_state.vector_data:
    st.error("❌ Failed to load documents. Please check your document folder and try refreshing the page.")
    st.info("💡 Make sure your PDF files are in the 'policy_docs' folder")
    st.stop()


# Input box
prompt1 = st.text_input("🔍 Ask a question about HR policies:", placeholder="e.g., What is the leave policy?")

# Handle user query
if prompt1:
    if st.session_state.vector_data and st.session_state.vector_data['custom_tfidf_retriever']:
        with st.spinner("🤖 Generating answer..."):
            try:
                base_retriever = st.session_state.vector_data['custom_tfidf_retriever']

                query_generator_llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-8b-8192")
                
                multiquery_retriever = MultiQueryRetriever.from_llm(
                    retriever=base_retriever,
                    llm=query_generator_llm,
                    include_original=True
                )

                top_docs = multiquery_retriever.invoke(prompt1)

                document_chain = create_stuff_documents_chain(llm, prompt)
                response = document_chain.invoke({"input": prompt1, "context": top_docs})

                st.write("**Answer:**")
                st.write(response)

            except Exception as e:
                st.error(f"❌ Error generating answer: {e}")
                st.error("Please try rephrasing your question or contact support.")
                import traceback
                st.error(f"Full traceback: {traceback.format_exc()}")
    else:
        st.error("❌ Documents not loaded. Please refresh the page.")

# Footer
st.markdown("---")
st.markdown("💡 **Tip**: Ask specific questions about HR policies, procedures, benefits, or company guidelines for best results.")
# st.markdown("**(Powered by TF-IDF and MultiQueryRetrieval with Groq LLaMA3)**")
