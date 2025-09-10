import gradio as gr
import requests
import PyPDF2
import os
import re
import time
import json
from io import BytesIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk
from functools import lru_cache

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

class VercelPDFQA:
    """Vercel-optimized PDF QA System using APIs instead of local models"""
    
    def __init__(self):
        self.hf_token = os.getenv('HUGGINGFACE_API_KEY', '')
        if not self.hf_token:
            print("Warning: HUGGINGFACE_API_KEY not found. Some features may not work.")
        
        self.headers = {"Authorization": f"Bearer {self.hf_token}"} if self.hf_token else {}
        
        # API endpoints (lightweight models)
        self.embedding_api = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"
        self.qa_api = "https://api-inference.huggingface.co/models/microsoft/DialoGPT-small"
        self.reranker_api = "https://api-inference.huggingface.co/models/cross-encoder/ms-marco-MiniLM-L-4-v2"
        
        # Configuration
        self.chunk_size = 300
        self.chunk_overlap = 75
        self.max_chunks = 20
        self.top_k = 5
    
    def extract_text_from_pdfs(self, pdf_files):
        """Extract text from uploaded PDF files"""
        if not pdf_files:
            return ""
        
        all_texts = []
        for pdf_file in pdf_files:
            try:
                # Handle different input types
                if hasattr(pdf_file, 'read'):
                    pdf_content = pdf_file.read()
                elif isinstance(pdf_file, str):
                    with open(pdf_file, 'rb') as f:
                        pdf_content = f.read()
                else:
                    pdf_content = pdf_file
                
                reader = PyPDF2.PdfReader(BytesIO(pdf_content))
                text_blocks = []
                
                for page_num, page in enumerate(reader.pages):
                    try:
                        text = page.extract_text()
                        if text.strip():
                            cleaned_text = self.clean_text(text)
                            if cleaned_text:
                                text_blocks.append(f"[Page {page_num+1}] {cleaned_text}")
                    except Exception as e:
                        print(f"Error extracting page {page_num+1}: {e}")
                        continue
                
                if text_blocks:
                    document_text = "\n\n".join(text_blocks)
                    all_texts.append(document_text)
                
            except Exception as e:
                print(f"Error processing PDF: {e}")
                continue
        
        return "\n\n--- DOCUMENT SEPARATOR ---\n\n".join(all_texts)
    
    def clean_text(self, text):
        """Clean and normalize text"""
        if not text:
            return ""
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove common artifacts
        text = re.sub(r'Page\s*\d+(\s*of\s*\d+)?', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\f', '', text)  # Form feed characters
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', '', text)  # Control characters
        
        return text.strip()
    
    def create_chunks(self, text):
        """Create semantic chunks from text"""
        try:
            from nltk import sent_tokenize
            sentences = sent_tokenize(text)
        except:
            # Fallback sentence splitting
            sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text.strip())
            sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # Check if adding this sentence would exceed chunk size
            if current_length + sentence_length > self.chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk).strip()
                
                if len(chunk_text) >= 40:  # Minimum chunk length
                    chunks.append({
                        "text": chunk_text,
                        "length": len(chunk_text),
                        "chunk_id": len(chunks),
                        "sentence_count": len(current_chunk)
                    })
                
                # Maintain overlap
                if self.chunk_overlap > 0 and len(current_chunk) > 1:
                    overlap_sentences = current_chunk[-min(2, len(current_chunk)-1):]
                    current_chunk = overlap_sentences
                    current_length = sum(len(s) for s in current_chunk)
                else:
                    current_chunk = []
                    current_length = 0
            
            current_chunk.append(sentence)
            current_length += sentence_length
        
        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk).strip()
            if len(chunk_text) >= 40:
                chunks.append({
                    "text": chunk_text,
                    "length": len(chunk_text),
                    "chunk_id": len(chunks),
                    "sentence_count": len(current_chunk)
                })
        
        return chunks
    
    def classify_query(self, query):
        """Classify query type for adaptive processing"""
        query_lower = query.lower().strip()
        
        if any(phrase in query_lower for phrase in ["what is", "define", "definition of", "meaning of"]):
            return "definition"
        elif any(phrase in query_lower for phrase in ["summarize", "summary", "key findings", "main points"]):
            return "summary"
        elif any(word in query_lower for word in ["when", "date", "time", "year", "period"]):
            return "temporal"
        elif any(phrase in query_lower for phrase in ["how to", "how do", "process", "method"]):
            return "procedural"
        elif any(word in query_lower for word in ["list", "types", "kinds", "categories", "examples"]):
            return "enumeration"
        elif any(word in query_lower for word in ["compare", "difference", "versus", "vs"]):
            return "comparison"
        else:
            return "general"
    
    def get_adaptive_weights(self, query_type):
        """Get adaptive weights based on query type"""
        weight_mapping = {
            "definition": {"semantic": 0.85, "lexical": 0.15},
            "summary": {"semantic": 0.90, "lexical": 0.10},
            "temporal": {"semantic": 0.25, "lexical": 0.75},
            "procedural": {"semantic": 0.75, "lexical": 0.25},
            "enumeration": {"semantic": 0.80, "lexical": 0.20},
            "comparison": {"semantic": 0.70, "lexical": 0.30},
            "general": {"semantic": 0.70, "lexical": 0.30}
        }
        return weight_mapping.get(query_type, {"semantic": 0.70, "lexical": 0.30})
    
    @lru_cache(maxsize=50)
    def call_hf_api(self, api_url, payload_str, retries=3):
        """Cached API calls with retry logic"""
        if not self.hf_token:
            return {}
        
        payload = json.loads(payload_str)
        
        for attempt in range(retries):
            try:
                response = requests.post(api_url, headers=self.headers, json=payload, timeout=30)
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 503:
                    time.sleep(2 ** attempt)
                    continue
                else:
                    print(f"API call failed with status {response.status_code}")
                    if attempt == retries - 1:
                        return {}
            except Exception as e:
                print(f"API call error (attempt {attempt + 1}): {e}")
                if attempt == retries - 1:
                    return {}
                time.sleep(1)
        
        return {}
    
    def retrieve_chunks(self, question, chunks):
        """Retrieve relevant chunks using hybrid search"""
        if not chunks:
            return []
        
        query_type = self.classify_query(question)
        weights = self.get_adaptive_weights(query_type)
        
        # Lexical search using TF-IDF
        texts = [chunk["text"] for chunk in chunks]
        lexical_scores = np.zeros(len(chunks))
        
        try:
            vectorizer = TfidfVectorizer(
                max_features=1000, 
                stop_words='english', 
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.95
            )
            tfidf_matrix = vectorizer.fit_transform(texts)
            question_vector = vectorizer.transform([question])
            lexical_scores = cosine_similarity(question_vector, tfidf_matrix)[0]
        except Exception as e:
            print(f"TF-IDF search failed: {e}")
        
        # Semantic search using API
        semantic_scores = np.zeros(len(chunks))
        
        if self.hf_token:
            try:
                # Limit chunks for API efficiency
                limited_texts = texts[:15]
                
                # Get embeddings for chunks
                chunk_payload = json.dumps({"inputs": limited_texts})
                chunk_embeddings = self.call_hf_api(self.embedding_api, chunk_payload)
                
                # Get query embedding
                query_payload = json.dumps({"inputs": [question]})
                query_embedding = self.call_hf_api(self.embedding_api, query_payload)
                
                if chunk_embeddings and query_embedding and len(chunk_embeddings) > 0:
                    query_emb = np.array(query_embedding[0] if isinstance(query_embedding[0], list) else query_embedding)
                    chunk_embs = np.array(chunk_embeddings)
                    
                    if len(query_emb.shape) == 1:
                        query_emb = query_emb.reshape(1, -1)
                    
                    if query_emb.shape[1] == chunk_embs.shape[1]:
                        similarities = cosine_similarity(query_emb, chunk_embs)[0]
                        semantic_scores[:len(similarities)] = similarities
            
            except Exception as e:
                print(f"Semantic search failed: {e}")
        
        # Combine scores with adaptive weights
        combined_scores = (
            weights["semantic"] * semantic_scores + 
            weights["lexical"] * lexical_scores
        )
        
        # Get top results
        top_indices = np.argsort(combined_scores)[::-1][:self.max_chunks]
        
        results = []
        for idx in top_indices:
            if idx < len(chunks):
                chunk = dict(chunks[idx])
                chunk.update({
                    "retrieval_score": float(combined_scores[idx]),
                    "lexical_score": float(lexical_scores[idx]),
                    "semantic_score": float(semantic_scores[idx]),
                    "query_type": query_type
                })
                results.append(chunk)
        
        return results
    
    def rerank_chunks(self, question, chunks):
        """Rerank chunks using cross-encoder API"""
        if not chunks or not self.hf_token:
            return chunks[:self.top_k]
        
        try:
            # Prepare pairs for reranking (limit for API)
            pairs = [[question, chunk["text"]] for chunk in chunks[:8]]
            payload = json.dumps({"inputs": pairs})
            
            rerank_scores = self.call_hf_api(self.reranker_api, payload)
            
            if rerank_scores and len(rerank_scores) > 0:
                for i, chunk in enumerate(chunks[:len(rerank_scores)]):
                    chunk["rerank_score"] = float(rerank_scores[i])
                
                # Sort by rerank score
                chunks = sorted(chunks, key=lambda x: x.get("rerank_score", 0.0), reverse=True)
        
        except Exception as e:
            print(f"Reranking failed: {e}")
        
        return chunks[:self.top_k]
    
    def generate_answer(self, question, context_chunks):
        """Generate answer using context"""
        if not context_chunks:
            return "I don't have enough information to answer this question.", 0.1
        
        # Prepare context
        context_texts = [chunk["text"] for chunk in context_chunks[:3]]
        context = " ".join(context_texts)
        
        # Limit context length
        if len(context) > 1500:
            context = context[:1500] + "..."
        
        query_type = context_chunks[0].get("query_type", "general")
        
        # Try API-based generation first
        if self.hf_token:
            answer, confidence = self.generate_with_api(question, context, query_type)
            if answer:
                return answer, confidence
        
        # Fallback to extractive method
        return self.extractive_fallback(question, context_chunks)
    
    def generate_with_api(self, question, context, query_type):
        """Generate answer using HuggingFace API"""
        # Create query-type specific prompts
        prompt_templates = {
            "definition": f"Define: {question}\n\nContext: {context}\n\nDefinition:",
            "summary": f"Summarize: {question}\n\nContext: {context}\n\nSummary:",
            "temporal": f"When/dates: {question}\n\nContext: {context}\n\nDates:",
            "enumeration": f"List: {question}\n\nContext: {context}\n\nList:",
            "procedural": f"How to: {question}\n\nContext: {context}\n\nSteps:",
            "comparison": f"Compare: {question}\n\nContext: {context}\n\nComparison:",
            "general": f"Answer: {question}\n\nContext: {context}\n\nAnswer:"
        }
        
        prompt = prompt_templates.get(query_type, prompt_templates["general"])
        
        try:
            payload = json.dumps({
                "inputs": prompt,
                "parameters": {
                    "max_length": 150,
                    "temperature": 0.3,
                    "do_sample": True
                }
            })
            
            result = self.call_hf_api(self.qa_api, payload)
            
            if result and isinstance(result, list) and len(result) > 0:
                answer = result[0].get("generated_text", "").strip()
                
                # Clean up answer
                if answer.startswith(prompt):
                    answer = answer[len(prompt):].strip()
                
                # Remove redundant phrases
                answer = re.sub(r'The documents? (discuss|mention|describe|state) that\s*', '', answer, flags=re.IGNORECASE)
                answer = re.sub(r'According to the (context|text|document),?\s*', '', answer, flags=re.IGNORECASE)
                answer = re.sub(r'Based on the (provided |given )?(context|information),?\s*', '', answer, flags=re.IGNORECASE)
                
                if answer and len(answer.split()) > 3:
                    confidence = min(0.85, 0.5 + 0.3 * len(answer.split()) / 20)
                    return answer.strip(), confidence
        
        except Exception as e:
            print(f"Answer generation failed: {e}")
        
        return None, 0.0
    
    def extractive_fallback(self, question, context_chunks):
        """Extractive fallback method"""
        question_words = set(re.findall(r'\w+', question.lower()))
        best_sentences = []
        
        for chunk in context_chunks[:3]:
            try:
                from nltk import sent_tokenize
                sentences = sent_tokenize(chunk["text"])
            except:
                sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', chunk["text"])
            
            for sentence in sentences:
                if len(sentence.strip()) < 10:
                    continue
                    
                sentence_words = set(re.findall(r'\w+', sentence.lower()))
                overlap = len(question_words & sentence_words) / max(len(question_words), 1)
                
                if overlap > 0.1:
                    best_sentences.append((sentence.strip(), overlap))
        
        if best_sentences:
            best_sentences.sort(key=lambda x: x[1], reverse=True)
            
            # Select best sentences
            selected_sentences = [s[0] for s in best_sentences[:2]]
            answer = ". ".join(selected_sentences)
            
            # Ensure proper punctuation
            if answer and not answer.endswith(('.', '!', '?')):
                answer += '.'
            
            confidence = min(0.7, sum(s[1] for s in best_sentences[:2]) / 2)
            return answer, confidence
        
        return "I couldn't find a specific answer in the documents.", 0.2
    
    def process_pdfs_and_answer(self, pdf_files, question, progress=gr.Progress()):
        """Main processing function"""
        if not pdf_files or not question.strip():
            return "Please upload at least one PDF file and ask a question.", ""
        
        try:
            progress(0.1, desc="Extracting text from PDFs...")
            
            # Extract text
            combined_text = self.extract_text_from_pdfs(pdf_files)
            if not combined_text:
                return "No text could be extracted from the PDFs.", ""
            
            progress(0.3, desc="Creating semantic chunks...")
            
            # Create chunks
            chunks = self.create_chunks(combined_text)
            if not chunks:
                return "Could not create text chunks from the documents.", ""
            
            progress(0.5, desc="Retrieving relevant content...")
            
            # Retrieve relevant chunks
            retrieved_chunks = self.retrieve_chunks(question, chunks)
            
            progress(0.7, desc="Reranking for relevance...")
            
            # Rerank chunks
            reranked_chunks = self.rerank_chunks(question, retrieved_chunks)
            
            progress(0.9, desc="Generating answer...")
            
            # Generate answer
            answer, confidence = self.generate_answer(question, reranked_chunks)
            
            # Create detailed statistics
            stats = f"""**System Performance:**
- Documents processed: 1
- Total chunks created: {len(chunks)}
- Chunks retrieved: {len(retrieved_chunks)}
- Chunks used for answer: {len(reranked_chunks)}
- Query type: {reranked_chunks[0].get('query_type', 'general') if reranked_chunks else 'unknown'}
- Answer confidence: {confidence:.1%}
- Average chunk length: {np.mean([chunk['length'] for chunk in chunks]):.0f} chars
- Total text length: {len(combined_text):,} chars"""
            
            progress(1.0, desc="Complete!")
            return answer, stats
            
        except Exception as e:
            error_msg = f"Error processing request: {str(e)}"
            print(error_msg)
            return error_msg, ""

# Initialize the QA system
qa_system = VercelPDFQA()

# Create Gradio interface
def create_interface():
    with gr.Blocks(title="Advanced PDF QA System", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🤖 Advanced PDF Question Answering System
        ### Powered by AI with Intelligent Document Analysis
        
        Upload your PDF documents and ask intelligent questions. The system uses:
        - **🧠 Smart chunking** with semantic boundaries
        - **🔍 Hybrid search** combining keyword and semantic matching
        - **📊 Query classification** for optimized processing
        - **⚡ Advanced reranking** for better relevance
        - **🎯 Context-aware** answer generation
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                pdf_inputs = gr.Files(
                    label="📄 Upload PDF Documents",
                    file_types=[".pdf"],
                    file_count="multiple"
                )
                
                question_input = gr.Textbox(
                    label="❓ Ask Your Question",
                    placeholder="What would you like to know about the documents?",
                    lines=3
                )
                
                submit_btn = gr.Button("🔍 Get Intelligent Answer", variant="primary", size="lg")
                
                gr.Markdown("""
                ### 💡 Example Questions:
                - **Definition**: "What is machine learning?"
                - **Summary**: "Summarize the key findings"
                - **Process**: "How does this method work?"
                - **List**: "What are the different types mentioned?"
                - **Timeline**: "When did these events occur?"
                """)
            
            with gr.Column(scale=1):
                answer_output = gr.Markdown(
                    label="💡 Intelligent Answer",
                    value="👆 Upload your PDF documents and ask a question to get started!",
                    elem_id="answer-output"
                )
                
                stats_output = gr.Textbox(
                    label="📊 Processing Statistics",
                    lines=8,
                    max_lines=12,
                    interactive=False
                )
        
        # Handle submissions
        submit_btn.click(
            fn=qa_system.process_pdfs_and_answer,
            inputs=[pdf_inputs, question_input],
            outputs=[answer_output, stats_output]
        )
        
        question_input.submit(
            fn=qa_system.process_pdfs_and_answer,
            inputs=[pdf_inputs, question_input],
            outputs=[answer_output, stats_output]
        )
    
    return demo

# Create and launch the demo
demo = create_interface()

if __name__ == "__main__":
    # Vercel deployment configuration
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        share=False,
        show_error=True
    )
