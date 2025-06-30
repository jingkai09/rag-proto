import streamlit as st
import openai
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import docx
import io
import re
from typing import List, Dict, Tuple
import json

# Configure page
st.set_page_config(
    page_title="RAG Prototype",
    page_icon="🤖",
    layout="wide"
)

class SimpleRAG:
    def __init__(self):
        """Initialize the RAG system"""
        self.embedding_model = None
        self.documents = []
        self.embeddings = []
        self.chunks = []
        self.chunk_metadata = []
        
    @st.cache_resource
    def load_embedding_model(_self):
        """Load the sentence transformer model"""
        try:
            return SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            st.error(f"Error loading embedding model: {e}")
            return None
    
    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract text from PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            st.error(f"Error extracting text from PDF: {e}")
            return ""
    
    def extract_text_from_docx(self, docx_file) -> str:
        """Extract text from DOCX file"""
        try:
            doc = docx.Document(docx_file)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            st.error(f"Error extracting text from DOCX: {e}")
            return ""
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace and newlines
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?;:-]', '', text)
        return text.strip()
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if len(chunk.strip()) > 0:
                chunks.append(chunk)
        
        return chunks
    
    def process_documents(self, uploaded_files, chunk_size: int = 500) -> bool:
        """Process uploaded documents and create embeddings"""
        if not self.embedding_model:
            self.embedding_model = self.load_embedding_model()
            if not self.embedding_model:
                return False
        
        self.documents = []
        self.chunks = []
        self.chunk_metadata = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"Processing {uploaded_file.name}...")
            
            # Extract text based on file type
            if uploaded_file.type == "application/pdf":
                text = self.extract_text_from_pdf(uploaded_file)
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                text = self.extract_text_from_docx(uploaded_file)
            elif uploaded_file.type == "text/plain":
                text = str(uploaded_file.read(), "utf-8")
            else:
                st.warning(f"Unsupported file type: {uploaded_file.type}")
                continue
            
            if not text.strip():
                st.warning(f"No text extracted from {uploaded_file.name}")
                continue
            
            # Clean and chunk text
            cleaned_text = self.clean_text(text)
            file_chunks = self.chunk_text(cleaned_text, chunk_size)
            
            # Store chunks with metadata
            for chunk_idx, chunk in enumerate(file_chunks):
                self.chunks.append(chunk)
                self.chunk_metadata.append({
                    'file_name': uploaded_file.name,
                    'chunk_index': chunk_idx,
                    'file_index': idx
                })
            
            progress_bar.progress((idx + 1) / len(uploaded_files))
        
        # Generate embeddings
        if self.chunks:
            status_text.text("Generating embeddings...")
            self.embeddings = self.embedding_model.encode(self.chunks)
            status_text.text(f"✅ Processed {len(self.chunks)} chunks from {len(uploaded_files)} documents")
            return True
        else:
            status_text.text("❌ No text chunks created")
            return False
    
    def retrieve_relevant_chunks(self, query: str, top_k: int = 5) -> List[Tuple[str, float, Dict]]:
        """Retrieve most relevant chunks for a query"""
        if not self.chunks or not self.embedding_model:
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Get top-k most similar chunks
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append((
                self.chunks[idx],
                similarities[idx],
                self.chunk_metadata[idx]
            ))
        
        return results
    
    def generate_response(self, query: str, relevant_chunks: List[Tuple[str, float, Dict]], 
                         openai_api_key: str) -> str:
        """Generate response using OpenAI API"""
        if not openai_api_key:
            return "Please provide an OpenAI API key to generate responses."
        
        # Set up OpenAI client
        openai.api_key = openai_api_key
        
        # Prepare context from relevant chunks
        context = "\n\n".join([chunk[0] for chunk in relevant_chunks])
        
        # Create prompt
        prompt = f"""Based on the following context, please answer the question. If the answer is not in the context, say so.

Context:
{context}

Question: {query}

Answer:"""
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating response: {str(e)}"

# Initialize RAG system
@st.cache_resource
def get_rag_system():
    return SimpleRAG()

def main():
    st.title("🤖 Simple RAG Prototype")
    st.markdown("Upload documents and ask questions about their content!")
    
    # Initialize RAG system
    rag = get_rag_system()
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # OpenAI API Key input
        openai_api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Enter your OpenAI API key for response generation"
        )
        
        # Chunk size configuration
        chunk_size = st.slider(
            "Chunk Size (words)",
            min_value=100,
            max_value=1000,
            value=500,
            step=50,
            help="Size of text chunks for processing"
        )
        
        # Number of retrieved chunks
        top_k = st.slider(
            "Retrieved Chunks",
            min_value=1,
            max_value=10,
            value=5,
            help="Number of relevant chunks to retrieve"
        )
    
    # Main interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📄 Document Upload")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Choose files",
            accept_multiple_files=True,
            type=['pdf', 'docx', 'txt'],
            help="Upload PDF, DOCX, or TXT files"
        )
        
        if uploaded_files:
            st.write(f"📁 {len(uploaded_files)} file(s) uploaded")
            
            # Process documents button
            if st.button("🔄 Process Documents", type="primary"):
                with st.spinner("Processing documents..."):
                    success = rag.process_documents(uploaded_files, chunk_size)
                    if success:
                        st.success(f"✅ Successfully processed {len(rag.chunks)} chunks!")
                        
                        # Show document statistics
                        st.subheader("📊 Document Statistics")
                        stats_df = pd.DataFrame([
                            {"Metric": "Total Documents", "Value": len(uploaded_files)},
                            {"Metric": "Total Chunks", "Value": len(rag.chunks)},
                            {"Metric": "Average Chunk Length", "Value": f"{np.mean([len(chunk.split()) for chunk in rag.chunks]):.0f} words"}
                        ])
                        st.dataframe(stats_df, hide_index=True)
    
    with col2:
        st.header("❓ Ask Questions")
        
        if rag.chunks:
            # Query input
            query = st.text_input(
                "Enter your question:",
                placeholder="What is this document about?"
            )
            
            if query and st.button("🔍 Search & Answer", type="primary"):
                with st.spinner("Searching and generating answer..."):
                    # Retrieve relevant chunks
                    relevant_chunks = rag.retrieve_relevant_chunks(query, top_k)
                    
                    if relevant_chunks:
                        # Generate response
                        response = rag.generate_response(query, relevant_chunks, openai_api_key)
                        
                        # Display response
                        st.subheader("💬 Answer")
                        st.write(response)
                        
                        # Display relevant chunks
                        st.subheader("📋 Relevant Chunks")
                        for i, (chunk, similarity, metadata) in enumerate(relevant_chunks):
                            with st.expander(f"Chunk {i+1} (Similarity: {similarity:.3f}) - {metadata['file_name']}"):
                                st.write(chunk)
                    else:
                        st.warning("No relevant chunks found for your query.")
        else:
            st.info("👆 Please upload and process documents first to ask questions.")
    
    # Footer
    st.markdown("---")
    st.markdown("**Note:** This is a simple RAG prototype. For production use, consider using more robust vector databases and error handling.")

if __name__ == "__main__":
    main()
