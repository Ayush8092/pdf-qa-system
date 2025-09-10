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
from docx import Document
from fpdf import FPDF

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
                            cleaned = self.clean_text(text)
                            if cleaned:
                                text_blocks.append(f"[Page {page_num+1}] {cleaned}")
                    except:
                        continue
                if text_blocks:
                    all_texts.append("\n\n".join(text_blocks))
            except:
                continue
        return "\n\n--- DOCUMENT SEPARATOR ---\n\n".join(all_texts)

    def clean_text(self, text):
        """Clean and normalize text"""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'Page\s*\d+(\s*of\s*\d+)?', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\f', '', text)
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', '', text)
        return text.strip()

    def create_chunks(self, text):
        """Create semantic chunks from text"""
        try:
            from nltk import sent_tokenize
            sentences = sent_tokenize(text)
        except:
            sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text.strip())
            sentences = [s.strip() for s in sentences if s.strip()]
        chunks = []; curr=[]; length=0
        for s in sentences:
            l = len(s)
            if length + l > self.chunk_size and curr:
                chunk = ' '.join(curr).strip()
                if len(chunk) >= 40:
                    chunks.append({"text": chunk})
                if self.chunk_overlap>0 and len(curr)>1:
                    curr = curr[-1:]
                    length = len(curr[0])
                else:
                    curr, length = [], 0
            curr.append(s); length += l
        if curr:
            chunk = ' '.join(curr).strip()
            if len(chunk)>=40:
                chunks.append({"text": chunk})
        return chunks

    def classify_query(self, query):
        q=query.lower()
        if any(p in q for p in ["what is","define","meaning"]): return "definition"
        if any(p in q for p in ["summarize","summary","key findings"]): return "summary"
        if any(w in q for w in ["when","date","year"]): return "temporal"
        if any(p in q for p in ["how to","process","method"]): return "procedural"
        if any(w in q for w in ["list","types","examples"]): return "enumeration"
        if any(w in q for w in ["compare","difference","versus","vs"]): return "comparison"
        return "general"

    @lru_cache(maxsize=50)
    def call_hf_api(self, url, payload_str):
        if not self.hf_token: return {}
        for _ in range(3):
            try:
                resp = requests.post(url, headers=self.headers, json=json.loads(payload_str), timeout=30)
                if resp.status_code==200: return resp.json()
            except:
                time.sleep(1)
        return {}

    def retrieve_chunks(self, question, chunks):
        texts=[c["text"] for c in chunks]
        lexical=np.zeros(len(texts))
        try:
            v=TfidfVectorizer(max_features=1000,stop_words='english',ngram_range=(1,2))
            M=v.fit_transform(texts)
            qv=v.transform([question])
            lexical=cosine_similarity(qv,M)[0]
        except: pass
        semantic=np.zeros(len(texts))
        if self.hf_token:
            try:
                limited=texts[:15]
                embp=json.dumps({"inputs":limited})
                embc=self.call_hf_api(self.embedding_api,embp)
                qp=json.dumps({"inputs":[question]})
                embq=self.call_hf_api(self.embedding_api,qp)
                if embc and embq:
                    qt=np.array(embq[0] if isinstance(embq[0],list) else embq)
                    ct=np.array(embc)
                    if qt.ndim==1: qt=qt.reshape(1,-1)
                    if qt.shape[1]==ct.shape[1]:
                        semantic[:len(ct)]=cosine_similarity(qt,ct)[0]
            except: pass
        w=self.get_adaptive_weights(self.classify_query(question))
        combined=w["semantic"]*semantic + w["lexical"]*lexical
        idx=np.argsort(combined)[::-1][:self.max_chunks]
        return [ {**chunks[i],"score":combined[i]} for i in idx ]

    def generate_answer(self, question, context_chunks):
        if not context_chunks: return "Insufficient info.",0.1
        ctx=" ".join(c["text"] for c in context_chunks[:3])
        if len(ctx)>1500: ctx=ctx[:1500]+"..."
        prompt=f"Answer: {question}\n\nContext: {ctx}\n\nAnswer:"
        payload=json.dumps({"inputs":prompt,"parameters":{"max_length":150,"temperature":0.3,"do_sample":True}})
        res=self.call_hf_api(self.qa_api,payload)
        ans=res[0].get("generated_text","").strip() if isinstance(res,list) and res else ""
        # clean
        ans=re.sub(rf"^{re.escape(prompt)}","",ans).strip()
        return ans or "No answer found.",0.5

    def process_and_answer(self, pdfs, question):
        text=self.extract_text_from_pdfs(pdfs)
        if not text: return "No text extracted.",""
        chunks=self.create_chunks(text)
        retrieved=self.retrieve_chunks(question,chunks)
        answer,_=self.generate_answer(question,retrieved)
        return answer

    def download_docx(self, answer):
        doc=Document()
        doc.add_paragraph(answer)
        path="/tmp/answer.docx"
        doc.save(path)
        return path

    def download_pdf(self, answer):
        pdf=FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        for line in answer.split('\n'):
            pdf.multi_cell(0,10,line)
        path="/tmp/answer.pdf"
        pdf.output(path)
        return path

# Initialize
qa=VercelPDFQA()

with gr.Blocks() as demo:
    gr.Markdown("""
    # 🤖 PDF Question Answering
    ### Upload PDFs and ask questions online

    - **Semantic chunking with overlap**
    - **Hybrid search combining keyword & semantic matching**
    - **Query classification & adaptive processing**
    """)
    pdfs=gr.Files(label="Upload PDFs",file_types=[".pdf"])
    q=gr.Textbox(label="Your Question",lines=2)
    out=gr.Textbox(label="Answer",interactive=False,lines=5)
    btn=gr.Button("Get Answer")
    doc_btn=gr.Button("Download as DOCX")
    pdf_btn=gr.Button("Download as PDF")

    btn.click(fn=lambda pdfs,q: qa.process_and_answer(pdfs,q),inputs=[pdfs,q],outputs=out)
    doc_btn.click(fn=lambda ans: qa.download_docx(ans),inputs=out,outputs=gr.File(label="Download DOCX"))
    pdf_btn.click(fn=lambda ans: qa.download_pdf(ans),inputs=out,outputs=gr.File(label="Download PDF"))

if __name__=="__main__":
    demo.launch(server_name="0.0.0.0",server_port=int(os.environ.get("PORT",7860)),share=False)
