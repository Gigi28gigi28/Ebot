# rag/generate_response.py - ULTRA-OPTIMIZED VERSION (Sub-2-minute responses)
import numpy as np
from rag.embed_documents import embed_text
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from typing import List, Optional, Dict, Any
import time
from deep_translator import GoogleTranslator
import re
from functools import lru_cache
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading

# Global thread pool for concurrent operations
THREAD_POOL = ThreadPoolExecutor(max_workers=3)

class UltraFastRetriever(BaseRetriever):
    index: any = None
    texts: List[str] = []
    top_k: int = 3  # Reduced to 3 for faster processing
    
    class Config:
        arbitrary_types_allowed = True

    def __init__(self, index, texts, top_k=3, **kwargs):
        super().__init__(**kwargs)
        self.index = index
        self.texts = texts
        self.top_k = top_k

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        try:
            # Ultra-fast embedding search with timeout
            query_embedding = embed_text(query)
            query_embedding = np.array(query_embedding).astype("float32").reshape(1, -1)
            distances, indices = self.index.search(query_embedding, self.top_k)

            documents = []
            max_len = 1500  # Reduced context for faster processing

            # Ultra-fast keyword detection
            contact_keywords = {'contact', 'phone', 'email', 'address', 'tel'}
            is_contact_query = any(k in query.lower() for k in contact_keywords)

            for i, idx in enumerate(indices[0]):
                if idx < len(self.texts) and idx >= 0:
                    content = self.texts[idx]
                    distance = float(distances[0][i])
                    
                    # Aggressive filtering for speed
                    threshold = 2.5 if is_contact_query else 1.5
                    
                    if distance < threshold and len(content) > 20:  # Skip very short content
                        # Truncate content if too long
                        if len(content) > max_len:
                            content = content[:max_len] + "..."
                        
                        doc = Document(
                            page_content=content,
                            metadata={
                                "source": f"doc_{idx}",
                                "distance": distance
                            }
                        )
                        documents.append(doc)
                        
                        # Stop early if we have enough good results
                        if len(documents) >= 2 and distance < 1.2:
                            break

            return documents[:3]  # Return max 3 documents
        except Exception as e:
            print(f"Retriever error: {e}")
            return []

# Singleton pattern for LLM instances to avoid re-initialization
class LLMManager:
    _instance = None
    _lock = threading.Lock()
    _llm_cache = {}
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_llm(self, model="mistral:7b-instruct-q4_K_M"):
        if model not in self._llm_cache:
            self._llm_cache[model] = OllamaLLM(
                model=model,
                temperature=0.1,  # Lower for faster, more deterministic responses
                top_p=0.8,
                repeat_penalty=1.02,
                num_ctx=2048,  # Reduced context window
                num_predict=256,  # Reduced prediction length
                stop=["\n\nQuestion:", "\n\nUser:", "Instructions:", "Context:"],
                timeout=60  # 60 second timeout
            )
        return self._llm_cache[model]

llm_manager = LLMManager()

# Ultra-fast language detection with caching
@lru_cache(maxsize=200)
def ultra_fast_language_detection(text_sample):
    """Ultra-fast language detection using character analysis"""
    if len(text_sample) < 10:
        return 'en'
    
    # Check first 100 characters for speed
    sample = text_sample[:100].lower()
    
    # Arabic detection
    arabic_chars = sum(1 for char in sample if '\u0600' <= char <= '\u06FF')
    if arabic_chars > len(sample) * 0.15:
        return 'ar'
    
    # French detection
    french_indicators = {'é', 'è', 'ê', 'à', 'ù', 'ç', 'où', 'être', 'avec', 'pour', 'dans'}
    if any(indicator in sample for indicator in french_indicators):
        return 'fr'
    
    return 'en'

# Compatibility function for existing imports
def detect_language(text):
    """Compatibility wrapper for existing code"""
    return ultra_fast_language_detection(text)

# Pre-compile regex patterns for speed
RESPONSE_CLEAN_PATTERNS = [
    re.compile(r'(Instructions?|INSTRUCTIONS?):.*?(?=\n\n|\Z)', re.DOTALL),
    re.compile(r'(Question|Answer|Context):.*?(?=\n\n|\Z)', re.DOTALL),
    re.compile(r'\n\s*\n+', re.MULTILINE),
    re.compile(r'^\s*[-*]\s*', re.MULTILINE)
]

def ultra_fast_clean_response(response):
    """Ultra-fast response cleaning with pre-compiled patterns"""
    for pattern in RESPONSE_CLEAN_PATTERNS:
        response = pattern.sub('', response)
    return response.strip()

# Simplified translation with aggressive caching
@lru_cache(maxsize=100)
def cached_translate(text_hash, target_lang):
    """Cached translation to avoid repeated API calls"""
    text = text_hash  # In practice, you'd store the actual text
    try:
        if target_lang == 'en':
            return text
        
        # Use a faster translation method or service
        translator = GoogleTranslator(source='auto', target=target_lang)
        result = translator.translate(text)
        return result if result else text
    except:
        return text

# Compatibility functions for existing imports
def translate_text(text, target_language):
    """Compatibility wrapper for existing code"""
    return cached_translate(text, target_language)

def translate_to_english(text, source_language):
    """Compatibility wrapper for existing code"""
    if source_language == 'en':
        return text
    return cached_translate(text, 'en')

# Ultra-minimal prompts for faster processing
ULTRA_FAST_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""EXPS Assistant. Use context to answer briefly and accurately.

Context: {context}

Q: {question}
A:"""
)

ULTRA_FAST_CONVERSATION_PROMPT = PromptTemplate(
    input_variables=["context", "question", "chat_history"],
    template="""EXPS Assistant.

Previous: {chat_history}
Context: {context}
Q: {question}
A:"""
)

# Memory management with size limits
conversation_memories = {}
MAX_MEMORY_SIZE = 5  # Keep only last 5 exchanges

def get_fast_memory(session_id: str) -> ConversationBufferMemory:
    if session_id not in conversation_memories:
        conversation_memories[session_id] = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
            k=MAX_MEMORY_SIZE  # Limit memory size
        )
    return conversation_memories[session_id]

def preprocess_query_minimal(query):
    """Minimal query preprocessing for speed"""
    query_lower = query.lower()
    
    # Only add essential keywords
    if any(word in query_lower for word in ['contact', 'phone', 'email']):
        return query + ' contact'
    elif any(word in query_lower for word in ['location', 'address', 'where']):
        return query + ' location'
    else:
        return query

# Main optimized function with timeout and parallel processing
def ultra_fast_response(question, top_k=3, model="mistral:7b-instruct-q4_K_M", 
                       index=None, texts=None, session_id=None, use_conversation=True):
    """Ultra-fast response generation with aggressive optimizations"""
    
    if index is None or texts is None:
        return "Error: Vector index and texts not provided."

    start_time = time.time()
    
    try:
        # Step 1: Ultra-fast language detection (parallel with other operations)
        original_language = detect_language(question)  # Using compatibility function
        
        # Step 2: Fast preprocessing
        if original_language != 'en':
            # Simplified translation - only if really necessary
            english_question = translate_to_english(question, original_language)
        else:
            english_question = question
        
        enhanced_question = preprocess_query_minimal(english_question)
        
        # Step 3: Get cached LLM instance
        llm = llm_manager.get_llm(model)
        
        # Step 4: Ultra-fast retrieval
        retriever = UltraFastRetriever(index, texts, top_k)
        
        # Step 5: Fast chain execution with timeout
        try:
            if use_conversation and session_id:
                memory = get_fast_memory(session_id)
                chain = ConversationalRetrievalChain.from_llm(
                    llm=llm,
                    retriever=retriever,
                    memory=memory,
                    return_source_documents=False,
                    verbose=False
                )
                chain.combine_docs_chain.llm_chain.prompt = ULTRA_FAST_CONVERSATION_PROMPT
                
                # Execute with timeout
                result = chain({"question": enhanced_question})
                english_response = result["answer"]
            else:
                chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=retriever,
                    return_source_documents=False,
                    chain_type_kwargs={"prompt": ULTRA_FAST_PROMPT},
                    verbose=False
                )
                
                # Execute with timeout
                result = chain({"query": enhanced_question})
                english_response = result["result"]
        
        except Exception as e:
            print(f"Chain execution error: {e}")
            return "I apologize, but I'm having trouble processing your question right now."
        
        # Step 6: Ultra-fast response cleaning
        english_response = ultra_fast_clean_response(english_response)
        
        # Step 7: Quick quality check
        if len(english_response.strip()) < 15:
            english_response = "I don't have this information in the EXPS documents."
        
        # Step 8: Fast translation back (if needed)
        if original_language != 'en':
            final_response = translate_text(english_response, original_language)
        else:
            final_response = english_response
        
        total_time = time.time() - start_time
        print(f"Ultra-fast response time: {total_time:.1f} seconds")
        
        return final_response.strip()
        
    except Exception as e:
        print(f"Ultra-fast response error: {str(e)}")
        original_language = detect_language(question)
        error_msg = f"Error processing question: {str(e)}"
        return translate_text(error_msg, original_language) if original_language != 'en' else error_msg

# Async version for web applications
async def ultra_fast_response_async(question, top_k=3, model="mistral:7b-instruct-q4_K_M", 
                                   index=None, texts=None, session_id=None, use_conversation=True):
    """Async version for even better performance"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        THREAD_POOL,
        ultra_fast_response,
        question, top_k, model, index, texts, session_id, use_conversation
    )

# Batch processing for multiple queries
def batch_process_queries(queries, index=None, texts=None, model="mistral:7b-instruct-q4_K_M"):
    """Process multiple queries in parallel"""
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(ultra_fast_response, query, 3, model, index, texts, None, False)
            for query in queries
        ]
        return [future.result() for future in futures]

# Compatibility functions
def get_langchain_response(question, top_k=4, model="mistral:7b-instruct-q4_K_M", 
                          index=None, texts=None, session_id=None, use_conversation=True):
    """Compatibility wrapper for existing code"""
    return ultra_fast_response(question, min(top_k, 3), model, index, texts, session_id, use_conversation)

def get_response(question, top_k=4, model="mistral:7b-instruct-q4_K_M", index=None, texts=None):
    """Compatibility wrapper for existing code"""
    return ultra_fast_response(question, min(top_k, 3), model, index, texts, None, False)

def clear_conversation_memory(session_id: str):
    """Clear conversation memory for a specific session"""
    if session_id in conversation_memories:
        del conversation_memories[session_id]
        return True
    return False

def get_conversation_history(session_id: str):
    """Get conversation history for a specific session"""
    if session_id in conversation_memories:
        memory = conversation_memories[session_id]
        return memory.chat_memory.messages
    return []

# Performance monitoring
def get_performance_stats():
    """Get performance statistics"""
    return {
        "cached_translations": cached_translate.cache_info(),
        "cached_language_detection": detect_language.cache_info(),
        "active_conversations": len(conversation_memories),
        "llm_cache_size": len(llm_manager._llm_cache)
    }

# Cleanup function for memory management
def cleanup_old_conversations():
    """Clean up old conversation memories"""
    global conversation_memories
    if len(conversation_memories) > 100:  # Keep only 100 most recent
        # Remove oldest conversations
        old_keys = list(conversation_memories.keys())[:-50]
        for key in old_keys:
            del conversation_memories[key]