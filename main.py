import streamlit as st
import openai
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import PyPDF2
import docx
import io
import re
from typing import List, Dict, Tuple, Optional
import json
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Configure page
st.set_page_config(
    page_title="Advanced RAG Prototype",
    page_icon="🤖",
    layout="wide"
)

class AdvancedRAG:
    def __init__(self):
        """Initialize the advanced RAG system"""
        self.embedding_model = None
        self.tfidf_vectorizer = None
        self.documents = []
        self.embeddings = []
        self.tfidf_matrix = None
        self.chunks = []
        self.chunk_metadata = []
        self.stop_words = set(stopwords.words('english'))
        
    @st.cache_resource
    def load_embedding_model(_self, model_name: str):
        """Load the sentence transformer model"""
        try:
            return SentenceTransformer(model_name)
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
    
    def extract_text_from_csv(self, csv_file, **params) -> str:
        """Extract text from CSV file"""
        try:
            # Read CSV with various encodings
            encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
            df = None
            
            for encoding in encodings:
                try:
                    csv_file.seek(0)  # Reset file pointer
                    df = pd.read_csv(csv_file, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                st.error("Could not read CSV file with supported encodings")
                return ""
            
            # Convert DataFrame to text
            text_format = params.get('csv_format', 'structured')
            
            if text_format == 'structured':
                # Include column headers and structure
                text_parts = []
                text_parts.append(f"CSV File Structure:\nColumns: {', '.join(df.columns.tolist())}\nTotal Rows: {len(df)}\n")
                
                # Add column descriptions if available
                for col in df.columns:
                    unique_vals = df[col].nunique()
                    text_parts.append(f"Column '{col}': {unique_vals} unique values")
                    
                    # Add sample values for categorical columns
                    if unique_vals <= 20 and df[col].dtype == 'object':
                        sample_vals = df[col].value_counts().head(5).index.tolist()
                        text_parts.append(f"  Sample values: {', '.join(map(str, sample_vals))}")
                
                text_parts.append("\nData Rows:")
                
                # Include actual data rows
                for idx, row in df.iterrows():
                    if idx >= params.get('max_csv_rows', 1000):  # Limit rows to prevent memory issues
                        text_parts.append(f"... (showing first {params.get('max_csv_rows', 1000)} rows)")
                        break
                    
                    row_text = " | ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
                    text_parts.append(f"Row {idx + 1}: {row_text}")
                
                return "\n".join(text_parts)
            
            elif text_format == 'concatenated':
                # Simple concatenation of all cell values
                text_parts = []
                for col in df.columns:
                    col_values = df[col].dropna().astype(str).tolist()
                    text_parts.extend(col_values)
                return " ".join(text_parts)
            
            elif text_format == 'summary':
                # Generate a summary of the CSV content
                text_parts = []
                text_parts.append(f"CSV Summary: {len(df)} rows, {len(df.columns)} columns")
                text_parts.append(f"Columns: {', '.join(df.columns.tolist())}")
                
                # Describe each column
                for col in df.columns:
                    col_desc = df[col].describe() if df[col].dtype in ['int64', 'float64'] else df[col].value_counts()
                    text_parts.append(f"\n{col} column summary:")
                    text_parts.append(str(col_desc))
                
                return "\n".join(text_parts)
            
        except Exception as e:
            st.error(f"Error extracting text from CSV: {e}")
            return ""
    
    def extract_text_from_excel(self, excel_file, **params) -> str:
        """Extract text from Excel file"""
        try:
            # Read all sheets or specific sheet
            sheet_name = params.get('excel_sheet', None)  # None means all sheets
            
            if sheet_name:
                df = pd.read_excel(excel_file, sheet_name=sheet_name)
                sheets_data = {sheet_name: df}
            else:
                sheets_data = pd.read_excel(excel_file, sheet_name=None)
            
            text_parts = []
            
            for sheet_name, df in sheets_data.items():
                text_parts.append(f"\n=== Sheet: {sheet_name} ===")
                
                # Apply same formatting as CSV
                text_format = params.get('excel_format', 'structured')
                
                if text_format == 'structured':
                    text_parts.append(f"Sheet Structure:\nColumns: {', '.join(df.columns.tolist())}\nTotal Rows: {len(df)}\n")
                    
                    # Add column descriptions
                    for col in df.columns:
                        unique_vals = df[col].nunique()
                        text_parts.append(f"Column '{col}': {unique_vals} unique values")
                        
                        if unique_vals <= 20 and df[col].dtype == 'object':
                            sample_vals = df[col].value_counts().head(5).index.tolist()
                            text_parts.append(f"  Sample values: {', '.join(map(str, sample_vals))}")
                    
                    text_parts.append("\nData Rows:")
                    
                    # Include actual data rows
                    for idx, row in df.iterrows():
                        if idx >= params.get('max_excel_rows', 1000):
                            text_parts.append(f"... (showing first {params.get('max_excel_rows', 1000)} rows)")
                            break
                        
                        row_text = " | ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
                        text_parts.append(f"Row {idx + 1}: {row_text}")
                
                elif text_format == 'concatenated':
                    for col in df.columns:
                        col_values = df[col].dropna().astype(str).tolist()
                        text_parts.extend(col_values)
                
                elif text_format == 'summary':
                    text_parts.append(f"Sheet Summary: {len(df)} rows, {len(df.columns)} columns")
                    text_parts.append(f"Columns: {', '.join(df.columns.tolist())}")
                    
                    for col in df.columns:
                        col_desc = df[col].describe() if df[col].dtype in ['int64', 'float64'] else df[col].value_counts()
                        text_parts.append(f"\n{col} column summary:")
                        text_parts.append(str(col_desc))
            
            return "\n".join(text_parts)
            
        except Exception as e:
            st.error(f"Error extracting text from Excel: {e}")
            return ""
    
    def clean_text(self, text: str, cleaning_level: str = "medium") -> str:
        """Clean and normalize text with different intensity levels"""
        if cleaning_level == "light":
            # Minimal cleaning
            text = re.sub(r'\s+', ' ', text)
            return text.strip()
        elif cleaning_level == "medium":
            # Standard cleaning
            text = re.sub(r'\s+', ' ', text)
            text = re.sub(r'[^\w\s.,!?;:()-]', '', text)
            return text.strip()
        elif cleaning_level == "aggressive":
            # Heavy cleaning
            text = re.sub(r'\s+', ' ', text)
            text = re.sub(r'[^\w\s]', ' ', text)
            text = re.sub(r'\d+', '', text)  # Remove numbers
            return text.strip().lower()
        return text
    
    def chunk_text(self, text: str, strategy: str, chunk_size: int = 500, 
                   overlap: int = 50, min_chunk_size: int = 50) -> List[str]:
        """Split text into chunks using different strategies"""
        if strategy == "word_based":
            return self._chunk_by_words(text, chunk_size, overlap, min_chunk_size)
        elif strategy == "sentence_based":
            return self._chunk_by_sentences(text, chunk_size)
        elif strategy == "paragraph_based":
            return self._chunk_by_paragraphs(text, chunk_size)
        elif strategy == "semantic_based":
            return self._chunk_semantically(text, chunk_size)
        else:
            return self._chunk_by_words(text, chunk_size, overlap, min_chunk_size)
    
    def _chunk_by_words(self, text: str, chunk_size: int, overlap: int, min_chunk_size: int) -> List[str]:
        """Word-based chunking with overlap"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if len(chunk.split()) >= min_chunk_size:
                chunks.append(chunk)
        
        return chunks
    
    def _chunk_by_sentences(self, text: str, max_sentences: int) -> List[str]:
        """Sentence-based chunking"""
        sentences = re.split(r'[.!?]+', text)
        chunks = []
        current_chunk = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                current_chunk.append(sentence)
                if len(current_chunk) >= max_sentences:
                    chunks.append('. '.join(current_chunk) + '.')
                    current_chunk = []
        
        if current_chunk:
            chunks.append('. '.join(current_chunk) + '.')
        
        return chunks
    
    def _chunk_by_paragraphs(self, text: str, max_length: int) -> List[str]:
        """Paragraph-based chunking"""
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if paragraph:
                if len(current_chunk + paragraph) <= max_length:
                    current_chunk += paragraph + "\n\n"
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = paragraph + "\n\n"
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _chunk_semantically(self, text: str, target_size: int) -> List[str]:
        """Simple semantic chunking based on topic shifts"""
        sentences = re.split(r'[.!?]+', text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                sentence_length = len(sentence.split())
                if current_length + sentence_length <= target_size:
                    current_chunk.append(sentence)
                    current_length += sentence_length
                else:
                    if current_chunk:
                        chunks.append('. '.join(current_chunk) + '.')
                    current_chunk = [sentence]
                    current_length = sentence_length
        
        if current_chunk:
            chunks.append('. '.join(current_chunk) + '.')
        
        return chunks
    
    def process_documents(self, uploaded_files, **params) -> bool:
        """Process uploaded documents with advanced parameters"""
        embedding_model_name = params.get('embedding_model', 'all-MiniLM-L6-v2')
        
        if not self.embedding_model or self.embedding_model != embedding_model_name:
            self.embedding_model = self.load_embedding_model(embedding_model_name)
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
            elif uploaded_file.type == "text/csv" or uploaded_file.name.endswith('.csv'):
                text = self.extract_text_from_csv(uploaded_file, **params)
            elif uploaded_file.type in ["application/vnd.ms-excel", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"] or uploaded_file.name.endswith(('.xlsx', '.xls')):
                text = self.extract_text_from_excel(uploaded_file, **params)
            else:
                st.warning(f"Unsupported file type: {uploaded_file.type}")
                continue
            
            if not text.strip():
                st.warning(f"No text extracted from {uploaded_file.name}")
                continue
            
            # Clean text
            cleaned_text = self.clean_text(text, params.get('cleaning_level', 'medium'))
            
            # Chunk text
            file_chunks = self.chunk_text(
                cleaned_text,
                params.get('chunking_strategy', 'word_based'),
                params.get('chunk_size', 500),
                params.get('overlap', 50),
                params.get('min_chunk_size', 50)
            )
            
            # Store chunks with metadata
            for chunk_idx, chunk in enumerate(file_chunks):
                self.chunks.append(chunk)
                self.chunk_metadata.append({
                    'file_name': uploaded_file.name,
                    'file_type': uploaded_file.type,
                    'chunk_index': chunk_idx,
                    'file_index': idx,
                    'word_count': len(chunk.split()),
                    'char_count': len(chunk)
                })
            
            progress_bar.progress((idx + 1) / len(uploaded_files))
        
        if self.chunks:
            status_text.text("Generating embeddings and TF-IDF vectors...")
            
            # Generate semantic embeddings
            self.embeddings = self.embedding_model.encode(self.chunks)
            
            # Generate TF-IDF vectors for keyword search
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=params.get('tfidf_max_features', 5000),
                stop_words='english' if params.get('remove_stopwords', True) else None,
                ngram_range=(1, params.get('ngram_range', 2))
            )
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.chunks)
            
            status_text.text(f"✅ Processed {len(self.chunks)} chunks from {len(uploaded_files)} documents")
            return True
        else:
            status_text.text("❌ No text chunks created")
            return False
    
    def semantic_search(self, query: str, top_k: int = 5) -> List[Tuple[str, float, Dict]]:
        """Perform semantic search using embeddings"""
        if not self.chunks or not self.embedding_model:
            return []
        
        query_embedding = self.embedding_model.encode([query])
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append((
                self.chunks[idx],
                float(similarities[idx]),
                self.chunk_metadata[idx]
            ))
        
        return results
    
    def keyword_search(self, query: str, top_k: int = 5) -> List[Tuple[str, float, Dict]]:
        """Perform keyword search using TF-IDF"""
        if not self.chunks or self.tfidf_matrix is None:
            return []
        
        query_vector = self.tfidf_vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.tfidf_matrix)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:  # Only include non-zero similarities
                results.append((
                    self.chunks[idx],
                    float(similarities[idx]),
                    self.chunk_metadata[idx]
                ))
        
        return results
    
    def hybrid_search(self, query: str, top_k: int = 5, semantic_weight: float = 0.7) -> List[Tuple[str, float, Dict]]:
        """Combine semantic and keyword search"""
        semantic_results = self.semantic_search(query, top_k * 2)
        keyword_results = self.keyword_search(query, top_k * 2)
        
        # Create score dictionary
        scores = {}
        
        # Add semantic scores
        for chunk, score, metadata in semantic_results:
            chunk_id = id(chunk)
            scores[chunk_id] = {
                'chunk': chunk,
                'metadata': metadata,
                'semantic_score': score,
                'keyword_score': 0.0
            }
        
        # Add keyword scores
        for chunk, score, metadata in keyword_results:
            chunk_id = id(chunk)
            if chunk_id in scores:
                scores[chunk_id]['keyword_score'] = score
            else:
                scores[chunk_id] = {
                    'chunk': chunk,
                    'metadata': metadata,
                    'semantic_score': 0.0,
                    'keyword_score': score
                }
        
        # Calculate hybrid scores
        hybrid_results = []
        for chunk_id, data in scores.items():
            hybrid_score = (semantic_weight * data['semantic_score'] + 
                          (1 - semantic_weight) * data['keyword_score'])
            hybrid_results.append((
                data['chunk'],
                hybrid_score,
                data['metadata']
            ))
        
        # Sort by hybrid score and return top-k
        hybrid_results.sort(key=lambda x: x[1], reverse=True)
        return hybrid_results[:top_k]
    
    def retrieve_relevant_chunks(self, query: str, search_type: str = "hybrid", 
                               top_k: int = 5, **kwargs) -> List[Tuple[str, float, Dict]]:
        """Main retrieval method with multiple search strategies"""
        if search_type == "semantic":
            return self.semantic_search(query, top_k)
        elif search_type == "keyword":
            return self.keyword_search(query, top_k)
        elif search_type == "hybrid":
            semantic_weight = kwargs.get('semantic_weight', 0.7)
            return self.hybrid_search(query, top_k, semantic_weight)
        else:
            return self.semantic_search(query, top_k)
    
    def generate_response(self, query: str, relevant_chunks: List[Tuple[str, float, Dict]], 
                         **params) -> str:
        """Generate response using OpenAI API with advanced parameters"""
        openai_api_key = params.get('openai_api_key')
        if not openai_api_key:
            return "Please provide an OpenAI API key to generate responses."
        
        openai.api_key = openai_api_key
        
        # Prepare context from relevant chunks
        context_parts = []
        for i, (chunk, score, metadata) in enumerate(relevant_chunks):
            source_info = f"[Source: {metadata['file_name']}]"
            context_parts.append(f"{source_info}\n{chunk}")
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Advanced prompt engineering
        system_prompt = params.get('system_prompt', 
            "You are a helpful assistant that answers questions based on provided context. "
            "Be accurate, cite sources when possible, and indicate if information is not available in the context.")
        
        user_prompt_template = params.get('prompt_template',
            """Based on the following context, please answer the question. 
If the answer is not in the context, say so clearly.

Context:
{context}

Question: {query}

Answer:""")
        
        user_prompt = user_prompt_template.format(context=context, query=query)
        
        try:
            response = openai.ChatCompletion.create(
                model=params.get('model', "gpt-3.5-turbo"),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=params.get('max_tokens', 500),
                temperature=params.get('temperature', 0.7),
                top_p=params.get('top_p', 1.0),
                frequency_penalty=params.get('frequency_penalty', 0.0),
                presence_penalty=params.get('presence_penalty', 0.0)
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating response: {str(e)}"

# Initialize RAG system
@st.cache_resource
def get_rag_system():
    return AdvancedRAG()

def main():
    st.title("🚀 Advanced RAG Prototype")
    st.markdown("Upload documents (PDF, DOCX, TXT, CSV, Excel) and customize every aspect of your RAG pipeline!")
    
    # Initialize RAG system
    rag = get_rag_system()
    
    # Advanced sidebar configuration
    with st.sidebar:
        st.header("⚙️ Advanced Configuration")
        
        # API Configuration
        st.subheader("🔑 API Settings")
        openai_api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Enter your OpenAI API key for response generation"
        )
        
        # Document Processing Settings
        st.subheader("📄 Document Processing")
        
        embedding_model = st.selectbox(
            "Embedding Model",
            ["all-MiniLM-L6-v2", "all-mpnet-base-v2", "multi-qa-MiniLM-L6-cos-v1", 
             "paraphrase-multilingual-MiniLM-L12-v2"],
            help="Choose the sentence transformer model"
        )
        
        cleaning_level = st.selectbox(
            "Text Cleaning Level",
            ["light", "medium", "aggressive"],
            index=1,
            help="How aggressively to clean the text"
        )
        
        chunking_strategy = st.selectbox(
            "Chunking Strategy",
            ["word_based", "sentence_based", "paragraph_based", "semantic_based"],
            help="Method for splitting documents into chunks"
        )
        
        chunk_size = st.slider(
            "Chunk Size",
            min_value=100,
            max_value=2000,
            value=500,
            step=50,
            help="Size of text chunks (words for word-based, sentences for sentence-based)"
        )
        
        overlap = st.slider(
            "Chunk Overlap",
            min_value=0,
            max_value=200,
            value=50,
            help="Overlap between consecutive chunks (word-based only)"
        )
        
        min_chunk_size = st.slider(
            "Minimum Chunk Size",
            min_value=10,
            max_value=200,
            value=50,
            help="Minimum size for a chunk to be included"
        )
        
        # CSV/Excel specific settings
        st.subheader("📊 CSV/Excel Settings")
        
        csv_format = st.selectbox(
            "CSV/Excel Text Format",
            ["structured", "concatenated", "summary"],
            help="How to convert tabular data to text"
        )
        
        max_csv_rows = st.slider(
            "Max CSV Rows",
            min_value=100,
            max_value=5000,
            value=1000,
            help="Maximum number of rows to process from CSV/Excel"
        )
        
        excel_sheet = st.text_input(
            "Excel Sheet Name (optional)",
            placeholder="Leave empty for all sheets",
            help="Specific sheet name to process, or leave empty for all sheets"
        )
        
        # TF-IDF Settings
        st.subheader("🔍 Keyword Search Settings")
        
        tfidf_max_features = st.slider(
            "TF-IDF Max Features",
            min_value=1000,
            max_value=20000,
            value=5000,
            step=1000,
            help="Maximum number of TF-IDF features"
        )
        
        ngram_range = st.slider(
            "N-gram Range (max)",
            min_value=1,
            max_value=3,
            value=2,
            help="Maximum n-gram size for TF-IDF"
        )
        
        remove_stopwords = st.checkbox(
            "Remove Stopwords",
            value=True,
            help="Remove common stopwords from TF-IDF"
        )
        
        # Search Settings
        st.subheader("🎯 Search Settings")
        
        search_type = st.selectbox(
            "Search Type",
            ["hybrid", "semantic", "keyword"],
            help="Type of search to perform"
        )
        
        if search_type == "hybrid":
            semantic_weight = st.slider(
                "Semantic Weight",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.1,
                help="Weight for semantic vs keyword search in hybrid mode"
            )
        else:
            semantic_weight = 0.7
        
        top_k = st.slider(
            "Retrieved Chunks",
            min_value=1,
            max_value=15,
            value=5,
            help="Number of relevant chunks to retrieve"
        )
        
        # LLM Settings
        st.subheader("🤖 LLM Settings")
        
        model = st.selectbox(
            "OpenAI Model",
            ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo-preview"],
            help="OpenAI model to use for response generation"
        )
        
        max_tokens = st.slider(
            "Max Tokens",
            min_value=100,
            max_value=2000,
            value=500,
            help="Maximum tokens in the response"
        )
        
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=0.7,
            step=0.1,
            help="Creativity/randomness of the response"
        )
        
        # Advanced LLM parameters
        with st.expander("Advanced LLM Parameters"):
            top_p = st.slider("Top P", 0.0, 1.0, 1.0, 0.1)
            frequency_penalty = st.slider("Frequency Penalty", -2.0, 2.0, 0.0, 0.1)
            presence_penalty = st.slider("Presence Penalty", -2.0, 2.0, 0.0, 0.1)
    
    # Main interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📄 Document Upload & Processing")
        
        # File uploader with expanded file types
        uploaded_files = st.file_uploader(
            "Choose files",
            accept_multiple_files=True,
            type=['pdf', 'docx', 'txt', 'csv', 'xlsx', 'xls'],
            help="Upload PDF, DOCX, TXT, CSV, or Excel files"
        )
        
        if uploaded_files:
            st.write(f"📁 {len(uploaded_files)} file(s) uploaded")
            
            # Show file types
            file_types = {}
            for file in uploaded_files:
                file_ext = file.name.split('.')[-1].upper()
                file_types[file_ext] = file_types.get(file_ext, 0) + 1
            
            st.write("**File Types:** " + ", ".join([f"{ext}: {count}" for ext, count in file_types.items()]))
            
            # Process documents button
            if st.button("🔄 Process Documents", type="primary"):
                with st.spinner("Processing documents with advanced settings..."):
                    processing_params = {
                        'embedding_model': embedding_model,
                        'cleaning_level': cleaning_level,
                        'chunking_strategy': chunking_strategy,
                        'chunk_size': chunk_size,
                        'overlap': overlap,
                        'min_chunk_size': min_chunk_size,
                        'tfidf_max_features': tfidf_max_features,
                        'ngram_range': ngram_range,
                        'remove_stopwords': remove_stopwords,
                        'csv_format': csv_format,
                        'excel_format': csv_format,  # Use same format for both
                        'max_csv_rows': max_csv_rows,
                        'max_excel_rows': max_csv_rows,  # Use same limit for both
                        'excel_sheet': excel_sheet if excel_sheet.strip() else None
                    }
                    
                    success = rag.process_documents(uploaded_files, **processing_params)
                    if success:
                        st.success(f"✅ Successfully processed {len(rag.chunks)} chunks!")
                        
                        # Show detailed document statistics
                        st.subheader("📊 Document Statistics")
                        
                        # Basic stats
                        total_words = sum(metadata['word_count'] for metadata in rag.chunk_metadata)
                        total_chars = sum(metadata['char_count'] for metadata in rag.chunk_metadata)
                        avg_chunk_words = np.mean([metadata['word_count'] for metadata in rag.chunk_metadata])
                        
                        # File type breakdown
                        chunks_by_type = {}
                        for metadata in rag.chunk_metadata:
                            file_ext = metadata['file_name'].split('.')[-1].upper()
                            chunks_by_type[file_ext] = chunks_by_type.get(file_ext, 0) + 1
                        
                        stats_df = pd.DataFrame([
                            {"Metric": "Total Documents", "Value": len(uploaded_files)},
                            {"Metric": "Total Chunks", "Value": len(rag.chunks)},
                            {"Metric": "Total Words", "Value": f"{total_words:,}"},
                            {"Metric": "Total Characters", "Value": f"{total_chars:,}"},
                            {"Metric": "Avg Chunk Size", "Value": f"{avg_chunk_words:.0f} words"},
                            {"Metric": "Embedding Model", "Value": embedding_model},
                            {"Metric": "Chunking Strategy", "Value": chunking_strategy}
                        ])
                        st.dataframe(stats_df, hide_index=True)
                        
                        # Chunks by file type
                        if chunks_by_type:
                            st.subheader("📈 Chunks by File Type")
                            type_df = pd.DataFrame(list(chunks_by_type.items()), columns=['File Type', 'Chunks'])
                            st.bar_chart(type_df.set_index('File Type'))
                        
                        # Chunk size distribution
                        chunk_sizes = [metadata['word_count'] for metadata in rag.chunk_metadata]
                        st.subheader("📈 Chunk Size Distribution")
                        st.histogram(chunk_sizes, bins=20)
                        
                        # Show file processing details
                        with st.expander("📋 File Processing Details"):
                            file_details = []
                            for metadata in rag.chunk_metadata:
                                file_details.append({
                                    'File': metadata['file_name'],
                                    'Type': metadata.get('file_type', 'Unknown'),
                                    'Chunk': metadata['chunk_index'],
                                    'Words': metadata['word_count'],
                                    'Characters': metadata['char_count']
                                })
                            
                            detail_df = pd.DataFrame(file_details)
                            st.dataframe(detail_df, hide_index=True)
    
    with col2:
        st.header("❓ Advanced Q&A")
        
        if rag.chunks:
            # Query input
            query = st.text_area(
                "Enter your question:",
                placeholder="What insights can you provide from this data?",
                height=100
            )
            
            # Custom prompt templates
            with st.expander("🎨 Custom Prompt Engineering"):
                system_prompt = st.text_area(
                    "System Prompt",
                    value="You are a helpful assistant that answers questions based on provided context. Be accurate, cite sources when possible, and indicate if information is not available in the context. When analyzing tabular data, provide specific insights and summaries.",
                    height=100
                )
                
                prompt_template = st.text_area(
                    "User Prompt Template",
                    value="""Based on the following context, please answer the question. 
If the answer is not in the context, say so clearly.
When analyzing data from CSV/Excel files, provide specific insights, trends, and summaries.

Context:
{context}

Question: {query}

Answer:""",
                    height=150,
                    help="Use {context} and {query} as placeholders"
                )
            
            if query and st.button("🔍 Search & Answer", type="primary"):
                with st.spinner("Searching and generating answer..."):
                    # Retrieve relevant chunks
                    search_params = {
                        'semantic_weight': semantic_weight
                    } if search_type == "hybrid" else {}
                    
                    relevant_chunks = rag.retrieve_relevant_chunks(
                        query, search_type, top_k, **search_params
                    )
                    
                    if relevant_chunks:
                        # Generate response
                        generation_params = {
                            'openai_api_key': openai_api_key,
                            'model': model,
                            'max_tokens': max_tokens,
                            'temperature': temperature,
                            'top_p': top_p,
                            'frequency_penalty': frequency_penalty,
                            'presence_penalty': presence_penalty,
                            'system_prompt': system_prompt,
                            'prompt_template': prompt_template
                        }
                        
                        response = rag.generate_response(query, relevant_chunks, **generation_params)
                        
                        # Display response
                        st.subheader("💬 Answer")
                        st.write(response)
                        
                        # Display search results with detailed metrics
                        st.subheader(f"📋 Retrieved Chunks ({search_type.title()} Search)")
                        
                        for i, (chunk, similarity, metadata) in enumerate(relevant_chunks):
                            file_type = metadata.get('file_type', 'Unknown')
                            file_type_icon = {
                                'application/pdf': '📕',
                                'application/vnd.openxmlformats-officedocument.wordprocessingml.document': '📘',
                                'text/plain': '📄',
                                'text/csv': '📊',
                                'application/vnd.ms-excel': '📈',
                                'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': '📈'
                            }.get(file_type, '📄')
                            
                            with st.expander(
                                f"{file_type_icon} Chunk {i+1} - {metadata['file_name']} "
                                f"(Score: {similarity:.3f}, Words: {metadata['word_count']})"
                            ):
                                st.write(chunk)
                                
                                # Show metadata
                                st.caption(f"File: {metadata['file_name']} | "
                                         f"Type: {file_type} | "
                                         f"Chunk: {metadata['chunk_index']} | "
                                         f"Words: {metadata['word_count']} | "
                                         f"Characters: {metadata['char_count']}")
                    else:
                        st.warning("No relevant chunks found for your query.")
                        
                        # Show search debugging info
                        st.info(f"Search Type: {search_type} | "
                               f"Total Chunks: {len(rag.chunks)} | "
                               f"Query Length: {len(query.split())} words")
        else:
            st.info("👆 Please upload and process documents first to ask questions.")
            
            # Show available search modes and supported formats
            st.subheader("🔍 Search Modes Available")
            st.write("**Semantic Search**: Uses AI embeddings to find contextually similar content")
            st.write("**Keyword Search**: Uses TF-IDF to find exact keyword matches")
            st.write("**Hybrid Search**: Combines both semantic and keyword approaches")
            
            st.subheader("📁 Supported File Formats")
            st.write("**📕 PDF**: Extracts text from PDF documents")
            st.write("**📘 DOCX**: Processes Word documents")
            st.write("**📄 TXT**: Plain text files")
            st.write("**📊 CSV**: Tabular data with customizable text conversion")
            st.write("**📈 Excel (XLSX/XLS)**: Spreadsheets with multi-sheet support")
            
            # CSV/Excel processing info
            with st.expander("ℹ️ CSV/Excel Processing Options"):
                st.write("**Structured Format**: Includes column headers, data types, and row-by-row data")
                st.write("**Concatenated Format**: Simple concatenation of all cell values")
                st.write("**Summary Format**: Statistical summaries and overviews of the data")
                st.write("**Sheet Selection**: For Excel files, process all sheets or specify a particular sheet")
    
    # Footer with advanced info
    st.markdown("---")
    st.markdown("**Advanced RAG Prototype** - Customizable semantic and keyword search with hybrid retrieval supporting multiple file formats")
    
    if rag.chunks:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Documents Processed", len(set(m['file_name'] for m in rag.chunk_metadata)))
        with col2:
            st.metric("Total Chunks", len(rag.chunks))
        with col3:
            st.metric("Search Mode", search_type.title())
        with col4:
            file_types_processed = len(set(m.get('file_type', 'Unknown') for m in rag.chunk_metadata))
            st.metric("File Types", file_types_processed)

if __name__ == "__main__":
    main()
