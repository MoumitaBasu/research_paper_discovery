import os
# CRITICAL: Set cache directories at the absolute top before any other imports
os.environ["HF_HOME"] = os.path.abspath("./.hf_cache")
os.environ["TRANSFORMERS_CACHE"] = os.path.abspath("./.hf_cache")
os.environ["SENTENCE_TRANSFORMERS_HOME"] = os.path.abspath("./.hf_cache")

import streamlit as st
import requests
import pandas as pd
import arxiv
import time
import fitz
import re
import cloudscraper
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from langchain_text_splitters import RecursiveCharacterTextSplitter
from curl_cffi import requests as curl_requests
from slugify import slugify
import faiss
import numpy as np
import pymupdf4llm
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi
import google.generativeai as genai

if not os.path.exists(os.environ["HF_HOME"]):
    os.makedirs(os.environ["HF_HOME"], exist_ok=True)

st.set_page_config(page_title="Research Paper Discovery", layout="wide", page_icon="📚")

# Create a local storage folder for downloaded papers
DOWNLOAD_DIR = "downloaded_papers"
if not os.path.exists(DOWNLOAD_DIR):
    os.makedirs(DOWNLOAD_DIR)

st.title("📚 Research Paper Discovery")
st.markdown("Find the latest and most relevant research papers across various domains.")

# Complete list of fields of study
CATEGORIES = [
    "All (Default)",
    "Agricultural and Food Sciences",
    "Art",
    "Biology",
    "Business",
    "Chemistry",
    "Computer Science",
    "Economics",
    "Education",
    "Engineering",
    "Environmental Science",
    "Geography",
    "History",
    "Interdisciplinary",
    "Law",
    "Linguistics",
    "Materials Science",
    "Mathematics",
    "Medicine",
    "Philosophy",
    "Physics",
    "Political Science",
    "Psychology",
    "Sociology"
]

# Sidebar for inputs
with st.sidebar:
    st.header("Search Parameters")
    topic = st.text_input("Enter a Topic", placeholder="e.g., Generative AI workplace")
    category = st.selectbox("Category", CATEGORIES)
    sort_by = st.selectbox(
        "Sort By", 
        ["Relevance", "Deep Semantic Relevance (AI)", "Latest (Year)", "Most Cited", "Most Viewed/Influential"]
    )
    only_full_text = st.checkbox("Only Show Full-Text Papers", value=True)
    num_results = st.slider("Results per API", min_value=5, max_value=500, value=20)
    
    st.divider()
    st.header("⚙️ AI Reasoning Engine")
    use_gemini = st.toggle("Use Gemini (Cloud High-Pro)", value=False)
    if use_gemini:
        api_key = st.text_input("Gemini API Key", type="password", help="Get it at aistudio.google.com")
        if api_key:
            genai.configure(api_key=api_key)
            st.session_state['gemini_active'] = True
        else:
            st.warning("Enter API Key to enable Gemini.")
            st.session_state['gemini_active'] = False
    else:
        st.session_state['gemini_active'] = False
        
    search_button = st.button("Search Papers", type="primary", use_container_width=True)

@st.cache_resource
def load_generative_model():
    """Load Flan-T5-Base using the direct API (seq2seq - no prompt echo)"""
    import time
    from transformers import T5ForConditionalGeneration, T5Tokenizer
    model_id = "google/flan-t5-base"
    for attempt in range(3):
        try:
            tokenizer = T5Tokenizer.from_pretrained(model_id)
            model = T5ForConditionalGeneration.from_pretrained(model_id)
            return tokenizer, model
        except Exception as e:
            st.warning(f"Model load attempt {attempt+1} failed: {e}")
            if attempt < 2:
                time.sleep(5)
            else:
                st.error(f"❌ Flan-T5 failed to load: {e}")
                return None, None

def generate_answer(prompt, max_new_tokens=400, model_type="local"):
    """Orchestrate between Local T5 and Cloud Gemini."""
    if st.session_state.get('gemini_active'):
        try:
            # Using the '-latest' alias to ensure compatibility with all API versions
            model = genai.GenerativeModel("gemini-3-flash-preview")
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            st.error(f"Gemini API Error: {e}")
            return f"Gemini failure: {e}"

    # Local Flan-T5 Fallback
    result = load_generative_model()
    if result is None or result[0] is None:
        return None
    tokenizer, model = result
    import torch
    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                min_new_tokens=60,
                num_beams=4,
                length_penalty=1.2,
                no_repeat_ngram_size=3,
                early_stopping=False
            )
        return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    except Exception as e:
        return f"Local Generation error: {str(e)}"

@st.cache_resource
def load_reranker():
    import time
    # Cross-Encoders are the gold standard for RAG accuracy in 2026
    model_id = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    for attempt in range(3):
        try:
            return CrossEncoder(model_id)
        except Exception as e:
            if attempt < 2:
                time.sleep(3)
                continue
            else:
                return None

@st.cache_resource
def load_embedding_model():
    from sentence_transformers import SentenceTransformer
    import time
    for attempt in range(3):
        try:
            # Upgrade to BGE-Small (Top-tier for Research RAG)
            return SentenceTransformer("BAAI/bge-small-en-v1.5")
        except Exception as e:
            if attempt < 2:
                time.sleep(3)
                continue
            else:
                st.error("Failed to load AI model.")
                raise e

def semantic_rerank(query, papers_list):
    if not papers_list:
        return []
    try:
        model = load_embedding_model()
        
        # Performance: Batch encoding is 10x faster than loops
        query_vec = model.encode([query], normalize_embeddings=True)[0]
        
        paper_texts = [f"{p.get('Title', '')}. {p.get('Abstract', '')}" for p in papers_list]
        doc_vecs = model.encode(paper_texts, normalize_embeddings=True)
        
        for i, p in enumerate(papers_list):
            sim = np.dot(query_vec, doc_vecs[i])
            p['Semantic Relevance Score'] = float(sim)
            
        papers_list.sort(key=lambda x: x.get('Semantic Relevance Score', 0.0), reverse=True)
        return papers_list
    except Exception as e:
        st.error(f"Error during semantic embedding: {e}")
        return papers_list

class ResearchIntelligence:
    def __init__(self, model):
        self.model = model
        self.index = None
        self.chunks = []
        self.bm25 = None
        self.meta = {} # Store paper metadata (Title, Abstract)

    def tokenize(self, text):
        return re.findall(r"\b\w+\b", text.lower())

    def ingest(self, chunks, meta):
        self.chunks = chunks
        self.meta = meta # Anchoring point for global context
        texts = [c['text'] for c in chunks]
        
        # 1. Vector Indexing (Semantic)
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(np.array(embeddings).astype('float32'))
        
        # 2. Keyword Indexing (BM25 with improved regex tokenization)
        tokenized_corpus = [self.tokenize(t) for t in texts]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def retrieve(self, query, k=15):
        if not self.index or not self.bm25: return []
        
        # Hybrid Strategy: Top 10 from Vector + Top 10 from BM25
        # Vector search
        query_vec = self.model.encode([query], normalize_embeddings=True)
        _, v_indices = self.index.search(np.array(query_vec).astype('float32'), 10)
        
        # BM25 search
        tokenized_query = self.tokenize(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        b_indices = np.argsort(bm25_scores)[-10:][::-1]
        
        # Combine unique indices
        combined_indices = list(dict.fromkeys(list(v_indices[0]) + list(b_indices)))
        
        return [self.chunks[idx] for idx in combined_indices if idx < len(self.chunks)]

    def deep_retrieve(self, query, k_final=4):
        candidates = self.retrieve(query, k=20)
        if not candidates: return []
        
        reranker = load_reranker()
        if reranker:
            try:
                scores = reranker.predict([(query, c['text']) for c in candidates])
                scored = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
                return [s[1] for s in scored[:k_final]]
            except Exception as e:
                return candidates[:k_final]
        return candidates[:k_final]

    def get_abstract_chunk(self):
        """Return the Abstract/Intro chunk which contains the paper's core thesis."""
        abstract_chunks = [c for c in self.chunks if c.get('section') == 'Abstract/Intro']
        if abstract_chunks:
            return abstract_chunks[0]
        for c in self.chunks:
            if len(c['text'].strip()) >= 200:
                return c
        return None

    def answer_question(self, question):
        q_lower = question.lower()
        title = self.meta.get('Title', 'Unknown Research Paper')
        abstract = self.meta.get('Abstract', '')
        
        is_gemini = st.session_state.get('gemini_active', False)
        
        # Scale retrieval based on model intelligence
        k = 10 if is_gemini else 5
        raw_results = self.deep_retrieve(question, k_final=k)
        if not raw_results: return "No relevant context found.", []

        results = [r for r in raw_results if len(r['text'].strip()) >= 150]
        if not results: results = raw_results
        
        if is_gemini:
             # Gemini can handle a lot more context (3000 chars per passage)
             passages = "\n\n".join([f"Passage {i+1}:\n{r['text'][:3000]}" for i, r in enumerate(results[:5])])
             prompt = (
                 f"You are a Senior Research AI. Use the full Title, Abstract, and Passages to answer this user query.\n\n"
                 f"PAPER TITLE: {title}\n"
                 f"ABSTRACT: {abstract}\n\n"
                 f"RELEVANT CHUNKS:\n{passages}\n\n"
                 f"QUERY: {question}\n\n"
                 f"RESPONSE (Analyze deeply, use bullet points if needed, citing passage numbers):"
             )
        else:
            # Local model needs tighter bounds
            passages = "\n\n".join([f"Passage {i+1}:\n{r['text'][:600]}" for i, r in enumerate(results[:3])])
            prompt = (
                f"Paper: {title}\n"
                f"Context: {passages}\n\n"
                f"Question: {question}\n\n"
                f"Researcher Answer (3-5 sentences):"
            )

        answer = generate_answer(prompt)
        if answer is None:
            return "Generation failed. Verify API key or model load.", results
        return answer, results


    def summarize(self):
        is_gemini = st.session_state.get('gemini_active', False)
        categories = {
            "Core Problem": "What research problem does this paper address?",
            "Methodology": "What methods or algorithms does this paper use?",
            "Key Findings": "What are the main results or findings?",
            "Limitations": "What limitations are mentioned?"
        }
        summary = {}
        
        for label, query in categories.items():
            k = 6 if is_gemini else 3
            results = self.deep_retrieve(query, k_final=k)
            if results:
                # Gemini gets much more context
                limit = 2000 if is_gemini else 500
                context = "\n".join([r['text'][:limit] for r in results])
                prompt = (
                    f"Research Question: {query}\n"
                    f"Context Passages: {context}\n\n"
                    f"Provide a definitive answer based only on the passages."
                )
                answer = generate_answer(prompt, max_new_tokens=200 if is_gemini else 120)
                if answer and not "error" in answer.lower():
                    summary[label] = answer
                else:
                    summary[label] = results[0]['text'][:300] + "..."
            else:
                summary[label] = "No relevant content found."
        return summary

    def extract_themes(self):
        is_gemini = st.session_state.get('gemini_active', False)
        texts = [c['text'] for c in self.chunks]
        if len(texts) < 3: return ["Not enough content."]

        if is_gemini:
            # High-intelligence zero-shot clustering
            context = "\n---\n".join([t[:1000] for t in texts[:10]])
            prompt = f"Identify the 5 primary research themes in this paper based on these chunks. Return ONLY a comma-separated list of phrases.\n\n{context}"
            ans = generate_answer(prompt)
            return [t.strip() for t in ans.split(',')] if ans else ["General Analysis"]

        # Local BERTopic
        from bertopic import BERTopic
        try:
            topic_model = BERTopic(embedding_model=self.model, min_topic_size=2)
            topics, _ = topic_model.fit_transform(texts)
            info = topic_model.get_topic_info()
            themes = info[info['Topic'] >= 0]['Name'].tolist()
            return [t.split('_', 1)[-1].replace('_', ' ').capitalize() for t in themes[:3]]
        except Exception:
            return ["Unable to extract complex themes from this specific document structure."]

@st.cache_resource
def get_shared_embedding_model():
    return load_embedding_model()

def scrape_abstract_fallback(url):
    """Deep scraper for abstracts from problematic publishers (Springer, Elsevier, etc.)"""
    if not url: return None
    try:
        scraper = cloudscraper.create_scraper()
        response = scraper.get(url, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'lxml')
            
            # 1. Look for meta tags (Standards)
            meta_abs = soup.find('meta', attrs={'name': ['description', 'citation_abstract', 'dc.description', 'og:description']})
            if meta_abs and len(meta_abs.get('content', '')) > 200:
                return meta_abs['content']
            
            # 2. Specialized selectors (Springer/Nature/Elsevier common patterns)
            abs_div = soup.find(['div', 'section'], class_=re.compile(r'abstract|Abs|Abstract', re.I))
            if abs_div:
                # Specifically for Springer multi-part abstracts
                ps = abs_div.find_all('p')
                if ps:
                    return "\n".join([p.get_text() for p in ps])
                return abs_div.get_text(strip=True)
                
            # 3. Fallback to largest text block in a section
            for section in soup.find_all('section'):
                if 'abstract' in section.get('id', '').lower():
                    return section.get_text(strip=True)
    except:
        pass
    return None

def is_full_paper(text):
    text_lower = text.lower()
    # Check for presence of key academic section variants
    sections = [
        r'\babstract\b', 
        r'\bintroduction\b', 
        r'\bmethod|methodology|methods\b', 
        r'\bresult|results\b', 
        r'\bconclusion|conclusions|discussion\b'
    ]
    matches = 0
    for sec in sections:
        if re.search(sec, text_lower):
            matches += 1
            
    # If the text has at least 3 distinct academic sections, it's very likely a full paper
    return matches >= 3

# Sci-Hub Ultimate Fallback Function (Now with Mirror Rotation)
def check_scihub_for_pdf(doi):
    if not doi:
        return None
        
    doi = doi.replace("https://doi.org/", "")
    mirrors = ["https://sci-hub.st", "https://sci-hub.se", "https://sci-hub.ru"]
    
    for base_url in mirrors:
        url = f"{base_url}/{doi}"
        try:
            # Sci-Hub heavily blocks standard libraries, must impersonate Chrome TLS
            resp = curl_requests.get(url, impersonate="chrome", timeout=12)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, "lxml")
                meta_pdf = soup.find('meta', attrs={'name': 'citation_pdf_url'})
                if meta_pdf and meta_pdf.get('content'):
                    pdf_path = meta_pdf['content']
                    if pdf_path.startswith('//'):
                        return "https:" + pdf_path
                    elif pdf_path.startswith('/'):
                        return base_url + pdf_path
                    return pdf_path
        except:
            continue
    return None

def find_google_scholar_pdf(title):
    if not title:
        return None
    try:
        # Heavily encoded query
        query = re.sub(r'[^\w\s]', ' ', title).strip()
        search_url = f"https://scholar.google.com/scholar?q={query.replace(' ', '+')}"
        
        # Chrome impersonation to bypass Scholar's aggressive bot detection
        resp = curl_requests.get(search_url, impersonate="chrome", timeout=15)
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, "lxml")
            # Look for typical Scholar PDF/HTML links on the right side
            # This is where ResearchGate, Academia, and Publisher-mirror links hide
            for side_link in soup.select('.gs_or_ggsm a'):
                href = side_link['href']
                text = side_link.get_text().lower()
                if '[pdf]' in text or '[html]' in text or href.lower().endswith('.pdf'):
                    return href
            
            # Fallback to general links if no side-link found
            for a in soup.find_all('a', href=True):
                anchor_text = a.get_text().lower()
                href = a['href']
                if '[pdf]' in anchor_text or '[html]' in anchor_text:
                    if 'pdf' in href.lower() or 'article/download' in href.lower():
                        return href
    except:
        pass
    return None

def download_and_save_pdf(urls_to_try, title):
    """Attempts to download a PDF from multiple sources and save it locally."""
    scraper = cloudscraper.create_scraper(browser={
        'browser': 'chrome',
        'platform': 'windows',
        'desktop': True
    })
    
    filename = slugify(title) + ".pdf"
    filepath = os.path.join(DOWNLOAD_DIR, filename)
    
    # If the file already exists locally, just return it
    if os.path.exists(filepath):
        return filepath, "Already Downloaded"

    for url in urls_to_try:
        if not url:
            continue
        try:
            # 1. Direct PDF markers
            if "arxiv.org/abs/" in url:
                url = url.replace("arxiv.org/abs/", "arxiv.org/pdf/") + ".pdf"
            
            headers = {
                "Accept": "application/pdf,text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
            }
            
            # Use high-fidelity impersonation as primary check for ResearchGate/Springer
            try:
                response = curl_requests.get(url, headers=headers, impersonate="chrome", timeout=25)
            except:
                response = scraper.get(url, headers=headers, timeout=20)
                
            if response.status_code == 200:
                content_type = response.headers.get("Content-Type", "").lower()
                content = response.content
                
                # A: Direct PDF Match
                if "application/pdf" in content_type or content.startswith(b"%PDF"):
                    with open(filepath, "wb") as f:
                        f.write(content)
                    return filepath, "Success"
                
                # B: HTML Landing Page Scout
                elif "text/html" in content_type:
                    soup = BeautifulSoup(content, "lxml")
                    hidden_url = None
                    
                    # Academic Meta Scout
                    meta_pdf = soup.find('meta', attrs={'name': ['citation_pdf_url', 'eprints.document_url']})
                    if meta_pdf and meta_pdf.get('content'):
                        hidden_url = meta_pdf['content']
                    else:
                        # Smart Link Scout
                        for a in soup.find_all('a', href=True):
                            link_text = a.get_text().lower()
                            href = a['href']
                            if 'pdf' in href.lower() and ('download' in link_text or 'full text' in link_text):
                                hidden_url = urljoin(url, href)
                                if hidden_url.startswith('//'): hidden_url = "https:" + hidden_url
                                break
                    
                    if hidden_url:
                        try:
                            pdf_resp = curl_requests.get(hidden_url, headers=headers, impersonate="chrome", timeout=20)
                            if pdf_resp.status_code == 200 and ( "application/pdf" in pdf_resp.headers.get("Content-Type", "").lower() or pdf_resp.content.startswith(b"%PDF")):
                                with open(filepath, "wb") as f:
                                    f.write(pdf_resp.content)
                                return filepath, "Success (Found via Scout)"
                        except:
                            pass
        except:
            continue
            
    return None, "Failed"

def extract_and_chunk_full_paper(urls_to_try, fallback_abstract="", local_path=None):
    try:
        text_content = ""
        source_msg = ""
        
        # PRIORITIZE LOCAL CONTENT (Fastest and more reliable)
        if local_path and os.path.exists(local_path):
            try:
                # Use high-fidelity PyMuPDF4LLM for structured Markdown extraction
                md_text = pymupdf4llm.to_markdown(local_path)
                
                if md_text.strip():
                    text_content = md_text
                    source_msg = "Local PDF (Structured Markdown)"
            except Exception:
                pass
        
        # Fallback to scraping if no local content or local extraction failed
        if not text_content:
            scraper = cloudscraper.create_scraper(browser={
                'browser': 'chrome',
                'platform': 'windows',
                'desktop': True
            })
        
        for url in urls_to_try:
            if not url:
                continue
            
            try:
                if "arxiv.org/abs/" in url:
                    url = url.replace("arxiv.org/abs/", "arxiv.org/pdf/") + ".pdf"
                    
                headers = {
                    "Accept": "application/pdf,text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.5"
                }
                
                try:
                    response = scraper.get(url, headers=headers, timeout=20)
                except Exception:
                    response = curl_requests.get(url, headers=headers, impersonate="chrome", timeout=20)
                
                max_retries = 5
                retry_delay = 3
                retries = 0
                while response.status_code == 202 and retries < max_retries:
                    time.sleep(retry_delay)
                    try:
                        response = scraper.get(url, headers=headers, timeout=20)
                    except:
                        response = curl_requests.get(url, headers=headers, impersonate="chrome", timeout=20)
                    retries += 1
                
                if response.status_code != 200:
                    continue
                    
                content_type = response.headers.get("Content-Type", "")
                
                if "application/pdf" in content_type or response.content.startswith(b"%PDF"):
                    # High-fidelity Markdown extraction from memory stream
                    md_text = pymupdf4llm.to_markdown(stream=response.content)
                    
                    if md_text.strip():
                        # Structural Sanitization (Eliminates the ???? and orphan artifacts)
                        md_text = re.sub(r'[^\x00-\x7F\u00A0-\u00FF\u2010-\u2022\u2122]+', ' ', md_text)
                        
                        if len(md_text) > 2000: # Heuristic for a full document
                            text_content = md_text
                            source_msg = "Remote PDF (Structured Markdown)"
                            break
                        
                elif "text/html" in content_type:
                    soup = BeautifulSoup(response.content, "lxml")
                    pdf_url_from_html = None
                    meta_pdf = soup.find('meta', attrs={'name': 'citation_pdf_url'})
                    if meta_pdf and meta_pdf.get('content'):
                        pdf_url_from_html = meta_pdf['content']
                    else:
                        for a in soup.find_all('a', href=True):
                            href = a['href']
                            link_text = a.get_text().lower()
                            if ('pdf' in href.lower() or 'download pdf' in link_text or 'full text pdf' in link_text):
                                if not href.startswith('#') and not href.startswith('javascript'):
                                    pdf_url_from_html = urljoin(url, href)
                                    break
                                    
                    if pdf_url_from_html and pdf_url_from_html not in urls_to_try:
                        urls_to_try.append(pdf_url_from_html)
                        continue
                    
                    for element in soup(["script", "style", "nav", "footer", "header", "noscript", "aside"]):
                        element.decompose()
                    
                    scraped_text = soup.get_text(separator='\n', strip=True)
                    scraped_text = re.sub(r'\n{3,}', '\n\n', scraped_text)
                    
                    if len(scraped_text) > 1500: # Accepting even slightly shorter papers if they are on a direct link
                        text_content = scraped_text
                        source_msg = "Live Web Page Extraction (BeautifulSoup)"
                        break
            except Exception:
                continue
                
        if not text_content:
            text_content = fallback_abstract
            source_msg = "API Abstract Fallback (No open 'Full Paper' content found)"
            
        if not text_content or len(text_content.strip()) < 10:
            return None, "No readable text or abstract available to process.", ""
            
        # UNIVERSAL CLEANING
        # 1. Remove non-printable / noise artifacts (Fixes the ???? issues)
        text_content = re.sub(r'[\ufffd\x80-\xff]', '', text_content)
        # 2. Fix ligatures
        text_content = text_content.replace('\uf001', 'fi').replace('\uf002', 'fl').replace('\uf003', 'ff').replace('\uf004', 'ffi')
        # 3. Fix hyphenated words at line breaks
        text_content = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text_content)
        # 4. Vertical whitespace and encoding squashes
        text_content = re.sub(r'[ \t]+', ' ', text_content)
        text_content = re.sub(r'(?<!\n)\n(?!\n)', ' ', text_content)
        text_content = re.sub(r'\n{2,}', '\n\n', text_content).strip()
            
        # Professional researcher chunking (1200 chars ~ 300 words, 15% overlap)
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=700, 
            chunk_overlap=120,
            separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""]
        )
        
        raw_chunks = text_splitter.split_text(text_content)
        final_chunks = []
        for idx, text in enumerate(raw_chunks):
            # Heuristic section labeling if full document
            section_label = "General Content"
            if idx == 0: section_label = "Abstract/Intro"
            elif idx > len(raw_chunks) - 2: section_label = "Conclusion/References"
            
            final_chunks.append({
                "section": section_label,
                "text": text
            })
        
        return final_chunks, "Success", source_msg
        
    except Exception as e:
        return None, f"Error processing text: {str(e)}", ""

# Smart Unpaywall Fallback Function
def check_unpaywall_for_pdf(doi):
    if not doi:
        return None
        
    # Remove 'https://doi.org/' if present
    doi = doi.replace("https://doi.org/", "")
    
    url = f"https://api.unpaywall.org/v2/{doi}?email=test_user_app@example.com"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("is_oa") and data.get("best_oa_location"):
                return data["best_oa_location"].get("url_for_pdf")
    except:
        pass
    return None

def search_openalex(query, limit=20):
    url = "https://api.openalex.org/works"
    params = {
        "search": query,
        "per-page": limit
    }
    try:
        response = requests.get(url, params=params, timeout=15)
        if response.status_code == 200:
            papers = response.json().get("results", [])
            results = []
            for p in papers:
                # Reconstruct abstract from inverted index
                abstract = "No abstract available."
                if p.get("abstract_inverted_index"):
                    inv = p["abstract_inverted_index"]
                    words = sorted([(pos, word) for word, positions in inv.items() for pos in positions])
                    abstract = " ".join([word for pos, word in words])
                
                # Check for hidden abstract if API gives us generic 'No abstract available'
                if abstract == "No abstract available." and p.get("doi"):
                    abstract = scrape_abstract_fallback(p["doi"]) or abstract
                
                pdf_url = ""
                if p.get("open_access") and p["open_access"].get("oa_url") and p["open_access"]["oa_url"].endswith(".pdf"):
                    pdf_url = p["open_access"]["oa_url"]
                elif p.get("best_oa_location") and p["best_oa_location"].get("pdf_url"):
                    pdf_url = p["best_oa_location"]["pdf_url"]
                elif p.get("open_access") and p["open_access"].get("oa_url"):
                    # fallback to general oa_url if it doesn't end in .pdf explicitly but exists
                    pdf_url = p["open_access"]["oa_url"]
                
                authors = ", ".join([a.get("author", {}).get("display_name", "") for a in p.get("authorships", []) if a.get("author")])
                
                is_oa = p.get("open_access", {}).get("is_oa", False)
                results.append({
                    "Title": p.get("title") or "No Title",
                    "Authors": authors if authors else "Unknown Authors",
                    "Year": p.get("publication_year") or 0,
                    "Citation Count": p.get("cited_by_count") or 0,
                    "Views/Influential Cites": 0,
                    "URL": p.get("doi") or p.get("id") or "",
                    "PDF_URL": pdf_url,
                    "Abstract": abstract,
                    "FullTextAvailable": is_oa or bool(pdf_url),
                    "Source API": "OpenAlex"
                })
            return results
    except Exception as e:
        return None
    return None

def search_europe_pmc(query, limit=20):
    url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
    params = {
        "query": query,
        "format": "json",
        "resultType": "core",
        "pageSize": limit
    }
    try:
        response = requests.get(url, params=params, timeout=15)
        if response.status_code == 200:
            papers = response.json().get("resultList", {}).get("result", [])
            results = []
            for p in papers:
                url_link = f"https://europepmc.org/article/{p.get('source', 'MED')}/{p.get('id', '')}" if p.get('id') else ""
                is_oa = p.get("isOpenAccess") == "Y"
                abstract = p.get("abstractText", "No abstract available.")
                if abstract == "No abstract available." and p.get("doi"):
                    abstract = scrape_abstract_fallback(f"https://doi.org/{p['doi']}") or abstract
                
                results.append({
                    "Title": p.get("title", "No Title"),
                    "Authors": p.get("authorString", "Unknown Authors"),
                    "Year": int(p.get("pubYear", 0)),
                    "Citation Count": int(p.get("citedByCount", 0)),
                    "Views/Influential Cites": 0,
                    "URL": url_link,
                    "PDF_URL": "", 
                    "Abstract": abstract,
                    "FullTextAvailable": is_oa,
                    "Source API": "Europe PMC"
                })
            return results
    except Exception as e:
        return None
    return None

def search_arxiv(query, category_name, limit=20):
    # Mapping to correct arxiv category prefixes
    if category_name == "Computer Science":
        cat_query = "cat:cs.*"
    elif category_name == "Mathematics":
        cat_query = "cat:math.*"
    elif category_name == "Physics":
        cat_query = "cat:physics.*"
    elif category_name == "Economics":
        cat_query = "cat:econ.*"
    else:
        cat_query = ""
        
    if cat_query:
        full_query = f"{cat_query} AND all:{query}"
    else:
        full_query = f"all:{query}"
        
    client = arxiv.Client()
    search = arxiv.Search(
        query=full_query,
        max_results=limit,
        sort_by=arxiv.SortCriterion.Relevance
    )
    
    results = []
    try:
        for p in client.results(search):
            year = p.published.year if p.published else 0
            results.append({
                "Title": p.title,
                "Authors": ", ".join([a.name for a in p.authors]),
                "Year": int(year),
                "Citation Count": 0,  # ArXiv doesn't provide this natively
                "Views/Influential Cites": 0,
                "URL": p.entry_id,
                "PDF_URL": p.pdf_url,
                "Abstract": p.summary.replace('\n', ' '),
                "FullTextAvailable": True,
                "Source API": "arXiv"
            })
        return results
    except Exception as e:
        return None

def search_crossref(query, limit=20):
    url = "https://api.crossref.org/works"
    params = {
        "query": query,
        "select": "title,author,created,is-referenced-by-count,URL",
        "rows": limit
    }
    try:
        response = requests.get(url, params=params, timeout=15)
        if response.status_code == 200:
            papers = response.json().get("message", {}).get("items", [])
            results = []
            for p in papers:
                authors = ", ".join([f"{a.get('given', '')} {a.get('family', '')}".strip() for a in p.get("author", [])])
                year = 0
                if p.get("created") and p["created"].get("date-parts"):
                    year = p["created"]["date-parts"][0][0]
                title = p.get("title", ["No Title"])[0] if p.get("title") else "No Title"
                
                pdf_url = ""
                if p.get("link"):
                    for link in p["link"]:
                        if link.get("content-type") == "application/pdf":
                            pdf_url = link.get("URL", "")
                            break
                            
                abstract = "No abstract available."
                if p.get("DOI"):
                    abstract = scrape_abstract_fallback(f"https://doi.org/{p['DOI']}") or abstract
                
                results.append({
                    "Title": title,
                    "Authors": authors if authors else "Unknown Authors",
                    "Year": int(year),
                    "Citation Count": int(p.get("is-referenced-by-count", 0)),
                    "Views/Influential Cites": 0,
                    "URL": p.get("URL", ""),
                    "PDF_URL": pdf_url,
                    "Abstract": abstract,
                    "FullTextAvailable": bool(pdf_url),
                    "Source API": "Crossref"
                })
            return results
    except Exception as e:
        return None
    return None

def search_semantic_scholar(query, category, limit=20):
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "limit": limit,
        "fields": "title,url,year,citationCount,authors,abstract,venue,publicationDate,influentialCitationCount,openAccessPdf"
    }
    if category not in ["All (Default)", "Interdisciplinary"]:
        params["fieldsOfStudy"] = category
        
    headers = {"User-Agent": "ResearchPaperDiscovery/1.0 (mailto:test@example.com)"}
    
    try:
        response = requests.get(url, params=params, headers=headers, timeout=15)
        if response.status_code == 200:
            data = response.json()
            papers = data.get("data", [])
            if not papers and category == "All (Default)":
                return []
            results = []
            for p in papers:
                authors = ", ".join([a.get("name", "") for a in p.get("authors", [])]) if p.get("authors") else "Unknown Authors"
                year = p.get("year") or 0
                citation_count = p.get("citationCount") or 0
                influential_citations = p.get("influentialCitationCount") or 0
                
                pdf_url = ""
                if p.get("openAccessPdf"):
                    pdf_url = p["openAccessPdf"].get("url", "")
                    
                abstract = p.get("abstract") or "No abstract available."
                if abstract == "No abstract available." and p.get("externalIds", {}).get("DOI"):
                    abstract = scrape_abstract_fallback(f"https://doi.org/{p['externalIds']['DOI']}") or abstract
                
                results.append({
                    "Title": p.get("title", "No Title"),
                    "Authors": authors,
                    "Year": year,
                    "Citation Count": citation_count,
                    "Views/Influential Cites": influential_citations,
                    "URL": p.get("url", ""),
                    "PDF_URL": pdf_url,
                    "Abstract": abstract,
                    "FullTextAvailable": p.get("isOpenAccess", False),
                    "Source API": "Semantic Scholar"
                })
            return results
        else:
            return None
    except Exception as e:
        return None

if search_button:
    if not topic.strip():
        st.warning("Please enter a topic to search.")
    else:
        with st.spinner("Fetching amazing research from multiple databases..."):
            papers = None
            
            # Map categories to priority APIs
            arxiv_fields = ["Computer Science", "Mathematics", "Physics", "Economics"]
            epmc_fields = ["Medicine", "Biology", "Agricultural and Food Sciences"]
            
            # Formulate robust broad query for general purpose APIs like OpenAlex
            general_query = topic if category in ["All (Default)", "Interdisciplinary"] else f"{topic} {category}"
            
            # 1. Primary API router
            if sort_by == "Most Viewed/Influential":
                # Semantic Scholar natively supports influential citations; prioritize it heavily!
                papers = search_semantic_scholar(topic, category, limit=num_results)
            elif sort_by == "Most Cited" and category in arxiv_fields:
                # arXiv does not track citations; instantly route to robust citation engines
                papers = search_semantic_scholar(topic, category, limit=num_results)
                if not papers:
                    papers = search_openalex(general_query, limit=num_results)
            elif category in arxiv_fields:
                papers = search_arxiv(topic, category, limit=num_results)
            elif category in epmc_fields:
                papers = search_europe_pmc(topic, limit=num_results)
            else:
                papers = search_openalex(general_query, limit=num_results)
                
            # 2. Aggressive Fallback Cascade (if primary API returned None or [] and we still need answers)
            if not papers:
                papers = search_openalex(general_query, limit=num_results)
            if not papers:
                papers = search_semantic_scholar(topic, category, limit=num_results)
            if not papers:
                papers = search_europe_pmc(topic, limit=num_results) 
            if not papers:
                papers = search_crossref(general_query, limit=num_results)
            if not papers and sort_by not in ["Most Cited", "Most Viewed/Influential"]: # Last resort
                papers = search_arxiv(topic, category, limit=num_results)
                
            if papers:
                if only_full_text:
                    papers = [p for p in papers if p.get("FullTextAvailable")]
                
                if sort_by == "Deep Semantic Relevance (AI)":
                    papers = semantic_rerank(general_query, papers)
                    
                st.session_state['df'] = pd.DataFrame(papers)
                st.session_state['sort_by'] = sort_by
                st.session_state['category'] = category
            else:
                st.session_state['df'] = pd.DataFrame()
                st.error("No papers found for the specified topic and category across any of our 5 database APIs! Try broadening your keywords.")

if 'df' in st.session_state and not st.session_state['df'].empty:
    df = st.session_state['df']
    saved_sort = st.session_state.get('sort_by', "Relevance")
    saved_category = st.session_state.get('category', "All (Default)")
    
    # Apply Sorting safely on the rendered dataframe
    if saved_sort == "Latest (Year)":
        df = df.sort_values(by="Year", ascending=False)
    elif saved_sort == "Most Cited":
        if df.iloc[0]["Source API"] == "arXiv":
            st.info("💡 Note: Citation counts are not available via the arXiv API. Returning default relevance sorting.")
        else:
            df = df.sort_values(by="Citation Count", ascending=False)
    st.success(f"Successfully retrieved {len(df)} papers using the **{df.iloc[0]['Source API']}** database!")
    st.divider()
    
    # Render results using compact expanders to reduce scrolling (As requested)
    for i, row in df.iterrows():
        with st.expander(f"📄 {row['Title']} ({row['Year']})", expanded=False):
            st.markdown(f"**Authors:** {row['Authors']}")
            
            m_col1, m_col2, m_col3 = st.columns(3)
            m_col1.metric("Source", row["Source API"])
            if row["Source API"] in ["OpenAlex", "Europe PMC", "Crossref", "Semantic Scholar"]:
                m_col2.metric("Citations", int(row['Citation Count']))
            if row.get('Semantic Relevance Score'):
                m_col3.metric("AI Match", f"{row['Semantic Relevance Score']*100:.1f}%")
            
            # Restore visible abstract with truncation for better UX
            abs_text = row['Abstract']
            st.markdown(f"**Abstract:** {abs_text[:400]}..." if len(abs_text) > 400 else f"**Abstract:** {abs_text}")
            with st.expander("📝 Read Full Abstract"):
                st.write(abs_text)
            
            st.divider()
            
            # SIDE-BY-SIDE Action Buttons (Returning to the side-logic user liked)
            url_val = row['URL']
            target_pdf_url = row.get('PDF_URL', "") or url_val
            
            col_link, col_extract = st.columns([1, 1])
            if url_val:
                col_link.link_button("� Read / View Source", url_val, use_container_width=True)
            
            # DEEP AI & EXTRACTION LOGIC
            if col_extract.button("�️ Extract & Run Deep AI", key=f"ingest_{i}", use_container_width=True):
                with st.spinner("Executing Pipeline: Search Mirrors -> Download -> Chunk -> Vectorize..."):
                    urls_to_try = [target_pdf_url, url_val]
                    if "doi.org" in url_val:
                        sh = check_scihub_for_pdf(url_val)
                        if sh: urls_to_try.insert(0, sh)
                    
                    filepath, status = download_and_save_pdf(urls_to_try, row['Title'])
                    
                    if filepath:
                        st.success("✅ Step 1: Full PDF Saved Locally.")
                        with open(filepath, "rb") as f:
                            st.download_button(
                                label="📥 Download This Local PDF Copy",
                                data=f,
                                file_name=os.path.basename(filepath),
                                mime="application/pdf",
                                key=f"dl_local_{i}"
                            )
                            
                        # STEP 2: Logic for AI Analysis
                        chunks, msg, source = extract_and_chunk_full_paper(urls_to_try, row.get('Abstract', ''), local_path=filepath)
                        if chunks:
                            # PER-USER STATE: Store analyzer in session state, not global cache
                            embedding_model = get_shared_embedding_model()
                            analyzer = ResearchIntelligence(embedding_model)
                            analyzer.ingest(chunks, row.to_dict())
                            
                            st.session_state['analyzer'] = analyzer
                            st.session_state['active_analyzer_paper'] = row['Title']
                            st.session_state['analyzer_active'] = True
                            
                            # STORAGE OPTIMIZATION: Parse-and-Purge
                            # The text is now in memory/FAISS. We can delete the local PDF to save disk space.
                            try:
                                os.remove(filepath)
                                st.success(f"✅ Ingested {len(chunks)} blocks. Local storage purged to save memory.")
                            except:
                                st.success(f"✅ Ingested {len(chunks)} blocks.")
                        else:
                            st.error(f"Extraction failed: {msg}")
                    else:
                        st.warning("Could not download full PDF. Analyzing available web/abstract content instead...")
                        chunks, msg, source = extract_and_chunk_full_paper(urls_to_try, row.get('Abstract', ''), local_path=None)
                        if chunks:
                            embedding_model = get_shared_embedding_model()
                            analyzer = ResearchIntelligence(embedding_model)
                            analyzer.ingest(chunks, row.to_dict())
                            
                            st.session_state['analyzer'] = analyzer
                            st.session_state['active_analyzer_paper'] = row['Title']
                            st.session_state['analyzer_active'] = True
                            st.success("Ingested via fallback source.")

            if st.session_state.get('active_analyzer_paper') == row['Title']:
                st.info("💡 Advanced analysis unlocked for this paper.")
                s_tool, t_tool = st.columns(2)
                
                analyzer = st.session_state.get('analyzer')
                
                if analyzer and s_tool.button("📝 Quick Summary", key=f"sum_{i}", use_container_width=True):
                    with st.spinner("🤖 Generative synthesis..."):
                        summary = analyzer.summarize()
                        for k, v in summary.items():
                            st.markdown(f"**{k}:** {v}")
                
                if analyzer and t_tool.button("🔍 Extract Themes", key=f"theme_{i}", use_container_width=True):
                    with st.spinner("🔍 Mapping thematic clusters..."):
                        themes = analyzer.extract_themes()
                        st.write(", ".join([f"**{t}**" for t in themes]) if themes else "No themes found.")

    # UNIVERSAL CHAT SEARCH BAR AT THE END (As requested)
    if st.session_state.get('analyzer_active'):
        st.header("💬 Global Research Q&A Chat")
        active_title = st.session_state.get('active_analyzer_paper', 'Selected Paper')
        st.info(f"Currently chatting with: **{active_title}**")
        
        # Improved Chat Input with Button
        c1, c2 = st.columns([4, 1])
        chat_query = c1.text_input("Ask any question about the analyzed paper...", key="global_chat_input", placeholder="e.g., What are the main research conclusions?")
        ask_button = c2.button("🚀 Ask AI", use_container_width=True)
        
        if ask_button and chat_query.strip():
            st.session_state['last_query'] = chat_query
            analyzer = st.session_state.get('analyzer')
            if analyzer:
                with st.spinner("🧠 Reasoning Engine: Hybrid Retrieval -> Reranking -> Generative Synthesis..."):
                    answer, context_results = analyzer.answer_question(chat_query)
                    # Persist results to survive Streamlit re-runs
                    st.session_state['last_answer'] = answer
                    st.session_state['last_context'] = context_results

        # Display persistent results if they exist
        if st.session_state.get('last_answer'):
            st.markdown("---")
            st.markdown("### 🤖 Neural Insight")
            st.info(st.session_state['last_answer'])
            
            if st.session_state.get('last_context'):
                st.markdown("### 📚 Supporting Evidence")
                for cr in st.session_state['last_context']:
                    with st.expander(f"Context from {cr['section']}"):
                        st.write(cr['text'])
    else:
        st.info("💡 Analysis required: Click 'Analyze & Extract Context' on any paper above to enable the Q&A Chat at the bottom of the page.")
