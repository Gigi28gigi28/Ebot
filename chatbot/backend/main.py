from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from rag.generate_response import get_response, get_langchain_response, clear_conversation_memory, get_conversation_history
from rag.store_vectors import load_index, load_texts
from rag.feedback_system import FeedbackSystem, FeedbackData
import logging
import time
import asyncio
from datetime import datetime
from functools import lru_cache
import uuid
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename="expert_bot.log",
    filemode="a"
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Expert Bot API with Feedback & Auto-Learning",
    description="RAG-powered chatbot with feedback system and auto-learning capabilities",
    version="4.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS config
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Globals
INDEX = None
TEXTS = None
feedback_system = None
model_usage = {}
start_time = time.time()
request_count = 0

@app.on_event("startup")
async def startup_event():
    global INDEX, TEXTS, feedback_system
    try:
        logger.info("Loading index and texts...")
        INDEX = load_index("vectorstore/index.faiss")
        TEXTS = load_texts("vectorstore/texts.json")
        logger.info(f"Loaded index and {len(TEXTS)} texts")
        
        # Initialize feedback system
        feedback_system = FeedbackSystem()
        logger.info("Feedback system initialized")
    except Exception as e:
        logger.error(f"Startup load error: {e}")

# Existing models
class Query(BaseModel):
    question: str = Field(..., min_length=1, example="Quels sont les services de Expert ?")
    model: str = Field(default="mistral:7b-instruct-q4_K_M")
    top_k: int = Field(default=3, ge=1, le=10)

class ChatResponse(BaseModel):
    answer: str
    processing_time: float
    model_used: str
    documents_retrieved: int
    detected_language: str
    timestamp: str
    status: str = "success"
    response_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

class ConversationQuery(BaseModel):
    question: str = Field(..., min_length=1, example="Quels sont les services de Expert ?")
    session_id: str = Field(..., example="user_123_session")
    model: str = Field(default="mistral:7b-instruct-q4_K_M")
    top_k: int = Field(default=3, ge=1, le=10)
    use_conversation: bool = Field(default=True, description="Enable conversation memory")

class ConversationResponse(BaseModel):
    answer: str
    processing_time: float
    model_used: str
    documents_retrieved: int
    session_id: str
    conversation_enabled: bool
    detected_language: str
    timestamp: str
    status: str = "success"
    response_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

# New Feedback Models
class UserFeedback(BaseModel):
    response_id: str = Field(..., description="ID of the response being rated")
    session_id: Optional[str] = Field(None, description="Session ID if from conversation")
    feedback_type: str = Field(..., description="'like' or 'dislike'")
    question: str = Field(..., description="Original question")
    answer: str = Field(..., description="Bot's answer")
    model_used: str = Field(..., description="Model that generated the response")
    user_comment: Optional[str] = Field(None, description="Optional user comment")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

class FeedbackResponse(BaseModel):
    status: str
    message: str
    feedback_id: str
    timestamp: str
    auto_learning_triggered: bool = False

class FeedbackStats(BaseModel):
    total_feedback: int
    likes: int
    dislikes: int
    like_ratio: float
    recent_feedback: List[Dict[str, Any]]
    learning_stats: Dict[str, Any]

# Enhanced response models to include response_id
class EnhancedChatResponse(ChatResponse):
    feedback_enabled: bool = True

class EnhancedConversationResponse(ConversationResponse):
    feedback_enabled: bool = True

# Existing endpoints with enhanced response tracking
@app.get("/")
def root():
    return {
        "message": "🤖 Expert Bot API with Feedback & Auto-Learning is online!",
        "status": "running",
        "version": "4.0.0",
        "features": ["RAG", "LangChain", "Conversation Memory", "Multilingual Support", "Feedback System", "Auto-Learning"],
        "supported_languages": ["English", "French", "Arabic"],
        "docs": "/docs",
    }

@app.get("/health")
def health():
    uptime = time.time() - start_time
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "Expert Bot API with Feedback & Auto-Learning",
        "version": "4.0.0",
        "uptime": round(uptime, 2),
        "feedback_system": "active" if feedback_system else "inactive"
    }

# Enhanced chat endpoints
@app.post("/ask", response_model=EnhancedChatResponse)
async def ask(query: Query):
    """Enhanced endpoint with feedback tracking"""
    global request_count, INDEX, TEXTS, model_usage
    request_count += 1
    
    if INDEX is None or TEXTS is None:
        raise HTTPException(status_code=500, detail="Vector index not loaded")
    
    response_id = str(uuid.uuid4())
    logger.info(f"[#{request_count}] Original Q: {query.question[:80]}... (ID: {response_id})")
    start = time.time()
    
    from rag.generate_response import detect_language
    detected_lang = detect_language(query.question)
    
    try:
        answer = await asyncio.to_thread(get_response, query.question, query.top_k, query.model, INDEX, TEXTS)
        processing_time = time.time() - start
        model_usage[query.model] = model_usage.get(query.model, 0) + 1
        
        # Store interaction for feedback tracking
        if feedback_system:
            feedback_system.store_interaction(
                response_id=response_id,
                question=query.question,
                answer=answer,
                model_used=query.model,
                session_id=None,
                detected_language=detected_lang
            )
        
        return EnhancedChatResponse(
            answer=answer,
            processing_time=round(processing_time, 2),
            model_used=query.model,
            documents_retrieved=query.top_k,
            detected_language=detected_lang,
            timestamp=datetime.now().isoformat(),
            response_id=response_id,
            feedback_enabled=True
        )
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=EnhancedConversationResponse)
async def chat(query: ConversationQuery):
    """Enhanced LangChain endpoint with feedback tracking"""
    global request_count, INDEX, TEXTS, model_usage
    request_count += 1
    
    if INDEX is None or TEXTS is None:
        raise HTTPException(status_code=500, detail="Vector index not loaded")
    
    response_id = str(uuid.uuid4())
    logger.info(f"[#{request_count}] LangChain Q: {query.question[:80]}... (Session: {query.session_id}, ID: {response_id})")
    start = time.time()
    
    from rag.generate_response import detect_language
    detected_lang = detect_language(query.question)
    
    try:
        answer = await asyncio.to_thread(
            get_langchain_response,
            question=query.question,
            top_k=query.top_k,
            model=query.model,
            index=INDEX,
            texts=TEXTS,
            session_id=query.session_id,
            use_conversation=query.use_conversation
        )
        
        processing_time = time.time() - start
        model_usage[query.model] = model_usage.get(query.model, 0) + 1
        
        # Store interaction for feedback tracking
        if feedback_system:
            feedback_system.store_interaction(
                response_id=response_id,
                question=query.question,
                answer=answer,
                model_used=query.model,
                session_id=query.session_id,
                detected_language=detected_lang
            )
        
        return EnhancedConversationResponse(
            answer=answer,
            processing_time=round(processing_time, 2),
            model_used=query.model,
            documents_retrieved=query.top_k,
            session_id=query.session_id,
            conversation_enabled=query.use_conversation,
            detected_language=detected_lang,
            timestamp=datetime.now().isoformat(),
            response_id=response_id,
            feedback_enabled=True
        )
    except Exception as e:
        logger.error(f"Error processing LangChain query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Feedback endpoints
@app.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(feedback: UserFeedback):
    """Submit user feedback for a response"""
    if not feedback_system:
        raise HTTPException(status_code=500, detail="Feedback system not initialized")
    
    try:
        feedback_id = str(uuid.uuid4())
        
        # Store feedback
        feedback_data = FeedbackData(
            feedback_id=feedback_id,
            response_id=feedback.response_id,
            session_id=feedback.session_id,
            feedback_type=feedback.feedback_type,
            question=feedback.question,
            answer=feedback.answer,
            model_used=feedback.model_used,
            user_comment=feedback.user_comment,
            timestamp=feedback.timestamp
        )
        
        feedback_system.store_feedback(feedback_data)
        
        # Check if auto-learning should be triggered
        auto_learning_triggered = False
        if feedback.feedback_type == "dislike":
            # Trigger auto-learning for negative feedback
            auto_learning_triggered = await feedback_system.trigger_auto_learning(feedback_data)
        
        logger.info(f"Feedback received: {feedback.feedback_type} for response {feedback.response_id}")
        
        return FeedbackResponse(
            status="success",
            message=f"Thank you for your {feedback.feedback_type} feedback!",
            feedback_id=feedback_id,
            timestamp=datetime.now().isoformat(),
            auto_learning_triggered=auto_learning_triggered
        )
        
    except Exception as e:
        logger.error(f"Error processing feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/feedback/stats", response_model=FeedbackStats)
def get_feedback_stats():
    """Get feedback statistics"""
    if not feedback_system:
        raise HTTPException(status_code=500, detail="Feedback system not initialized")
    
    return feedback_system.get_feedback_stats()

@app.get("/feedback/recent")
def get_recent_feedback(limit: int = 10):
    """Get recent feedback entries"""
    if not feedback_system:
        raise HTTPException(status_code=500, detail="Feedback system not initialized")
    
    return feedback_system.get_recent_feedback(limit)

@app.post("/feedback/retrain")
async def manual_retrain():
    """Manually trigger retraining based on feedback"""
    if not feedback_system:
        raise HTTPException(status_code=500, detail="Feedback system not initialized")
    
    try:
        result = await feedback_system.retrain_model()
        return {
            "status": "success",
            "message": "Manual retraining completed",
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Manual retraining error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Enhanced stats endpoint
@app.get("/stats")
def stats():
    uptime = time.time() - start_time
    feedback_stats = feedback_system.get_feedback_stats() if feedback_system else None
    
    return {
        "uptime_seconds": round(uptime, 2),
        "requests": request_count,
        "models_used": model_usage,
        "version": "4.0.0",
        "features": ["RAG", "LangChain Enhanced", "Conversation Memory", "Multilingual Support", "Feedback System", "Auto-Learning"],
        "supported_languages": ["English", "French", "Arabic"],
        "feedback_stats": feedback_stats.dict() if feedback_stats else None,
        "timestamp": datetime.now().isoformat()
    }

# Existing endpoints (session management, models, etc.)
@app.post("/session/new")
def create_session():
    """Create a new conversation session"""
    session_id = f"session_{uuid.uuid4().hex[:8]}_{int(time.time())}"
    return {
        "session_id": session_id,
        "created_at": datetime.now().isoformat(),
        "status": "created"
    }

@app.delete("/session/{session_id}")
def clear_session(session_id: str):
    """Clear conversation memory for a session"""
    success = clear_conversation_memory(session_id)
    return {
        "session_id": session_id,
        "cleared": success,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/session/{session_id}/history")
def get_session_history(session_id: str):
    """Get conversation history for a session"""
    history = get_conversation_history(session_id)
    return {
        "session_id": session_id,
        "message_count": len(history),
        "messages": [{"role": msg.type, "content": msg.content} for msg in history],
        "timestamp": datetime.now().isoformat()
    }

@app.get("/models")
def list_models():
    try:
        import ollama
        models = ollama.list().get('models', [])
        names = [m['name'] for m in models]
        best = sorted(names, key=lambda x: len(x))[0] if names else "gemma:2b"
        return {"available_models": names, "recommended_model": best, "status": "success"}
    except Exception as e:
        return {"available_models": [], "recommended_model": "gemma:2b", "status": f"error: {e}"}

@app.get("/logs")
def logs():
    try:
        with open("expert_bot.log", "r", encoding="utf-8") as f:
            return {"logs": f.readlines()[-50:]}
    except:
        return {"logs": ["Log file not found."]}

# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception(request, exc):
    return JSONResponse(status_code=exc.status_code, content={
        "error": "HTTP error",
        "detail": exc.detail,
        "timestamp": datetime.now().isoformat()
    })

@app.exception_handler(Exception)
async def general_exception(request, exc):
    logger.error(f"Unhandled error: {exc}")
    return JSONResponse(status_code=500, content={
        "error": "Server error",
        "detail": str(exc),
        "timestamp": datetime.now().isoformat()
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=6500, reload=True)