import json
import os
import sqlite3
import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FeedbackData:
    feedback_id: str
    response_id: str
    session_id: Optional[str]
    feedback_type: str  # 'like' or 'dislike'
    question: str
    answer: str
    model_used: str
    user_comment: Optional[str]
    timestamp: str
    processed: bool = False

@dataclass
class InteractionData:
    response_id: str
    question: str
    answer: str
    model_used: str
    session_id: Optional[str]
    detected_language: str
    timestamp: str

class FeedbackSystem:
    def __init__(self, db_path: str = "feedback_system.db"):
        self.db_path = db_path
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.auto_learning_threshold = 3  # Trigger retraining after 3 dislikes
        self.learning_data_path = "vectorstore/learning_data.json"
        self.negative_patterns_path = "vectorstore/negative_patterns.json"
        
        # Initialize database
        self._init_database()
        
        # Load existing learning data
        self.learning_data = self._load_learning_data()
        self.negative_patterns = self._load_negative_patterns()
        
        logger.info("FeedbackSystem initialized successfully")

    def _init_database(self):
        """Initialize SQLite database for storing feedback and interactions"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create feedback table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    feedback_id TEXT UNIQUE NOT NULL,
                    response_id TEXT NOT NULL,
                    session_id TEXT,
                    feedback_type TEXT NOT NULL,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    model_used TEXT NOT NULL,
                    user_comment TEXT,
                    timestamp TEXT NOT NULL,
                    processed BOOLEAN DEFAULT FALSE
                )
            ''')
            
            # Create interactions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    response_id TEXT UNIQUE NOT NULL,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    model_used TEXT NOT NULL,
                    session_id TEXT,
                    detected_language TEXT NOT NULL,
                    timestamp TEXT NOT NULL
                )
            ''')
            
            # Create learning_sessions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS learning_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT UNIQUE NOT NULL,
                    feedback_count INTEGER NOT NULL,
                    improvements_made TEXT,
                    timestamp TEXT NOT NULL,
                    success BOOLEAN DEFAULT FALSE
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Database initialization error: {e}")
            raise

    def store_interaction(self, response_id: str, question: str, answer: str, 
                         model_used: str, session_id: Optional[str], detected_language: str):
        """Store interaction data for potential feedback"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            interaction_data = InteractionData(
                response_id=response_id,
                question=question,
                answer=answer,
                model_used=model_used,
                session_id=session_id,
                detected_language=detected_language,
                timestamp=datetime.now().isoformat()
            )
            
            cursor.execute('''
                INSERT OR REPLACE INTO interactions 
                (response_id, question, answer, model_used, session_id, detected_language, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                interaction_data.response_id,
                interaction_data.question,
                interaction_data.answer,
                interaction_data.model_used,
                interaction_data.session_id,
                interaction_data.detected_language,
                interaction_data.timestamp
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing interaction: {e}")

    def store_feedback(self, feedback_data: FeedbackData):
        """Store user feedback"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO feedback 
                (feedback_id, response_id, session_id, feedback_type, question, answer, 
                 model_used, user_comment, timestamp, processed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                feedback_data.feedback_id,
                feedback_data.response_id,
                feedback_data.session_id,
                feedback_data.feedback_type,
                feedback_data.question,
                feedback_data.answer,
                feedback_data.model_used,
                feedback_data.user_comment,
                feedback_data.timestamp,
                feedback_data.processed
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Feedback stored: {feedback_data.feedback_type} for response {feedback_data.response_id}")
            
        except Exception as e:
            logger.error(f"Error storing feedback: {e}")
            raise

    async def trigger_auto_learning(self, feedback_data: FeedbackData) -> bool:
        """Trigger auto-learning for negative feedback"""
        try:
            if feedback_data.feedback_type == "dislike":
                # Analyze the negative feedback
                await self._analyze_negative_feedback(feedback_data)
                
                # Check if we should trigger retraining
                recent_dislikes = self._get_recent_dislikes()
                
                if len(recent_dislikes) >= self.auto_learning_threshold:
                    logger.info(f"Auto-learning triggered: {len(recent_dislikes)} recent dislikes")
                    await self._perform_auto_learning(recent_dislikes)
                    return True
                else:
                    logger.info(f"Auto-learning not triggered: {len(recent_dislikes)} dislikes (threshold: {self.auto_learning_threshold})")
                    
            return False
            
        except Exception as e:
            logger.error(f"Error in auto-learning trigger: {e}")
            return False

    async def _analyze_negative_feedback(self, feedback_data: FeedbackData):
        """Analyze negative feedback to identify patterns"""
        try:
            # Extract key patterns from negative feedback
            question_embedding = self.model.encode([feedback_data.question])[0]
            
            # Store negative pattern
            negative_pattern = {
                "question": feedback_data.question,
                "answer": feedback_data.answer,
                "model_used": feedback_data.model_used,
                "user_comment": feedback_data.user_comment,
                "timestamp": feedback_data.timestamp,
                "embedding": question_embedding.tolist()
            }
            
            self.negative_patterns.append(negative_pattern)
            self._save_negative_patterns()
            
            logger.info(f"Negative pattern analyzed and stored for question: {feedback_data.question[:50]}...")
            
        except Exception as e:
            logger.error(f"Error analyzing negative feedback: {e}")

    def _get_recent_dislikes(self, hours: int = 24) -> List[Dict]:
        """Get recent dislike feedback within specified hours"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Calculate cutoff time
            cutoff_time = (datetime.now() - timedelta(hours=hours)).isoformat()
            
            cursor.execute('''
                SELECT * FROM feedback 
                WHERE feedback_type = 'dislike' 
                AND timestamp >= ? 
                AND processed = FALSE
                ORDER BY timestamp DESC
            ''', (cutoff_time,))
            
            rows = cursor.fetchall()
            conn.close()
            
            # Convert to dictionaries
            columns = ['id', 'feedback_id', 'response_id', 'session_id', 'feedback_type', 
                      'question', 'answer', 'model_used', 'user_comment', 'timestamp', 'processed']
            
            return [dict(zip(columns, row)) for row in rows]
            
        except Exception as e:
            logger.error(f"Error getting recent dislikes: {e}")
            return []

    async def _perform_auto_learning(self, recent_dislikes: List[Dict]):
        """Perform auto-learning based on negative feedback"""
        try:
            learning_session_id = f"auto_learning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            improvements_made = []
            
            logger.info(f"Starting auto-learning session: {learning_session_id}")
            
            # Group dislikes by similar questions
            question_groups = self._group_similar_questions(recent_dislikes)
            
            for group in question_groups:
                if len(group) >= 2:  # Only process if multiple similar questions
                    improvement = await self._generate_improvement_for_group(group)
                    if improvement:
                        improvements_made.append(improvement)
            
            # Store improved responses in learning data
            await self._update_learning_data(improvements_made)
            
            # Mark feedback as processed
            await self._mark_feedback_as_processed(recent_dislikes)
            
            # Log learning session
            await self._log_learning_session(learning_session_id, len(recent_dislikes), improvements_made)
            
            logger.info(f"Auto-learning completed: {len(improvements_made)} improvements made")
            
        except Exception as e:
            logger.error(f"Error in auto-learning: {e}")
            raise

    def _group_similar_questions(self, dislikes: List[Dict], similarity_threshold: float = 0.7) -> List[List[Dict]]:
        """Group similar questions together"""
        try:
            if not dislikes:
                return []
            
            # Get embeddings for all questions
            questions = [d['question'] for d in dislikes]
            embeddings = self.model.encode(questions)
            
            # Group similar questions
            groups = []
            used_indices = set()
            
            for i, embedding in enumerate(embeddings):
                if i in used_indices:
                    continue
                    
                group = [dislikes[i]]
                used_indices.add(i)
                
                for j, other_embedding in enumerate(embeddings):
                    if j in used_indices or i == j:
                        continue
                        
                    similarity = np.dot(embedding, other_embedding) / (
                        np.linalg.norm(embedding) * np.linalg.norm(other_embedding)
                    )
                    
                    if similarity > similarity_threshold:
                        group.append(dislikes[j])
                        used_indices.add(j)
                
                groups.append(group)
            
            return groups
            
        except Exception as e:
            logger.error(f"Error grouping similar questions: {e}")
            return []

    async def _generate_improvement_for_group(self, group: List[Dict]) -> Optional[Dict]:
        """Generate improvement suggestions for a group of similar negative feedback"""
        try:
            # Analyze common patterns in the group
            questions = [item['question'] for item in group]
            answers = [item['answer'] for item in group]
            comments = [item['user_comment'] for item in group if item['user_comment']]
            
            # Create improvement suggestion
            improvement = {
                "question_pattern": questions[0],  # Representative question
                "failed_answers": answers,
                "user_feedback": comments,
                "suggested_improvement": self._generate_better_response(questions[0], answers[0]),
                "confidence": len(group) / len(questions),  # Higher confidence for more similar questions
                "timestamp": datetime.now().isoformat()
            }
            
            return improvement
            
        except Exception as e:
            logger.error(f"Error generating improvement: {e}")
            return None

    def _generate_better_response(self, question: str, failed_answer: str) -> str:
        """Generate a better response based on the failed answer (simple heuristic)"""
        try:
            # Simple improvement heuristics
            improvements = []
            
            if len(failed_answer) < 50:
                improvements.append("Provide more detailed information")
            
            if "I don't have" in failed_answer:
                improvements.append("Try to provide related information from available documents")
            
            if not any(keyword in failed_answer.lower() for keyword in ['expert', 'petroleum', 'services', 'exps']):
                improvements.append("Ensure response mentions Expert Petroleum Services (EXPS)")
            
            if not improvements:
                improvements.append("Provide more comprehensive and specific information")
            
            suggestion = f"Improved response should: {', '.join(improvements)}. "
            suggestion += f"For the question '{question}', provide a more detailed and helpful answer about Expert Petroleum Services."
            
            return suggestion
            
        except Exception as e:
            logger.error(f"Error generating better response: {e}")
            return "Provide more detailed and helpful information"

    async def _update_learning_data(self, improvements: List[Dict]):
        """Update learning data with improvements"""
        try:
            for improvement in improvements:
                self.learning_data.append(improvement)
            
            # Save to file
            self._save_learning_data()
            
            logger.info(f"Learning data updated with {len(improvements)} improvements")
            
        except Exception as e:
            logger.error(f"Error updating learning data: {e}")

    async def _mark_feedback_as_processed(self, feedback_list: List[Dict]):
        """Mark feedback as processed"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for feedback in feedback_list:
                cursor.execute('''
                    UPDATE feedback 
                    SET processed = TRUE 
                    WHERE feedback_id = ?
                ''', (feedback['feedback_id'],))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Marked {len(feedback_list)} feedback items as processed")
            
        except Exception as e:
            logger.error(f"Error marking feedback as processed: {e}")

    async def _log_learning_session(self, session_id: str, feedback_count: int, improvements: List[Dict]):
        """Log the learning session"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO learning_sessions 
                (session_id, feedback_count, improvements_made, timestamp, success)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                session_id,
                feedback_count,
                json.dumps(improvements, ensure_ascii=False),
                datetime.now().isoformat(),
                len(improvements) > 0
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Learning session logged: {session_id}")
            
        except Exception as e:
            logger.error(f"Error logging learning session: {e}")

    async def retrain_model(self) -> Dict[str, Any]:
        """Manually trigger model retraining"""
        try:
            # Get all unprocessed negative feedback
            unprocessed_dislikes = self._get_unprocessed_dislikes()
            
            if not unprocessed_dislikes:
                return {"message": "No unprocessed negative feedback to learn from", "improvements": 0}
            
            # Perform learning
            await self._perform_auto_learning(unprocessed_dislikes)
            
            return {
                "message": "Retraining completed successfully",
                "processed_feedback": len(unprocessed_dislikes),
                "improvements": len(self.learning_data),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in manual retraining: {e}")
            raise

    def _get_unprocessed_dislikes(self) -> List[Dict]:
        """Get all unprocessed dislike feedback"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM feedback 
                WHERE feedback_type = 'dislike' 
                AND processed = FALSE
                ORDER BY timestamp DESC
            ''')
            
            rows = cursor.fetchall()
            conn.close()
            
            columns = ['id', 'feedback_id', 'response_id', 'session_id', 'feedback_type', 
                      'question', 'answer', 'model_used', 'user_comment', 'timestamp', 'processed']
            
            return [dict(zip(columns, row)) for row in rows]
            
        except Exception as e:
            logger.error(f"Error getting unprocessed dislikes: {e}")
            return []

    def get_feedback_stats(self) -> Dict[str, Any]:
        """Get feedback statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get total feedback counts
            cursor.execute('SELECT feedback_type, COUNT(*) FROM feedback GROUP BY feedback_type')
            feedback_counts = dict(cursor.fetchall())
            
            total_feedback = sum(feedback_counts.values())
            likes = feedback_counts.get('like', 0)
            dislikes = feedback_counts.get('dislike', 0)
            like_ratio = likes / total_feedback if total_feedback > 0 else 0
            
            # Get recent feedback
            cursor.execute('''
                SELECT feedback_type, question, answer, user_comment, timestamp 
                FROM feedback 
                ORDER BY timestamp DESC 
                LIMIT 5
            ''')
            recent_feedback = [dict(zip(['feedback_type', 'question', 'answer', 'user_comment', 'timestamp'], row)) 
                             for row in cursor.fetchall()]
            
            # Get learning stats
            cursor.execute('SELECT COUNT(*) FROM learning_sessions WHERE success = TRUE')
            successful_learning_sessions = cursor.fetchone()[0]
            
            conn.close()
            
            return {
                "total_feedback": total_feedback,
                "likes": likes,
                "dislikes": dislikes,
                "like_ratio": round(like_ratio, 2),
                "recent_feedback": recent_feedback,
                "learning_stats": {
                    "successful_sessions": successful_learning_sessions,
                    "total_improvements": len(self.learning_data),
                    "negative_patterns": len(self.negative_patterns)
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting feedback stats: {e}")
            return {
                "total_feedback": 0,
                "likes": 0,
                "dislikes": 0,
                "like_ratio": 0,
                "recent_feedback": [],
                "learning_stats": {"successful_sessions": 0, "total_improvements": 0, "negative_patterns": 0}
            }

    def get_recent_feedback(self, limit: int = 10) -> List[Dict]:
        """Get recent feedback entries"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT feedback_type, question, answer, user_comment, timestamp, processed
                FROM feedback 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (limit,))
            
            rows = cursor.fetchall()
            conn.close()
            
            columns = ['feedback_type', 'question', 'answer', 'user_comment', 'timestamp', 'processed']
            return [dict(zip(columns, row)) for row in rows]
            
        except Exception as e:
            logger.error(f"Error getting recent feedback: {e}")
            return []

    def _load_learning_data(self) -> List[Dict]:
        """Load learning data from file"""
        try:
            if os.path.exists(self.learning_data_path):
                with open(self.learning_data_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return []
        except Exception as e:
            logger.error(f"Error loading learning data: {e}")
            return []

    def _save_learning_data(self):
        """Save learning data to file"""
        try:
            os.makedirs(os.path.dirname(self.learning_data_path), exist_ok=True)
            with open(self.learning_data_path, 'w', encoding='utf-8') as f:
                json.dump(self.learning_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Error saving learning data: {e}")

    def _load_negative_patterns(self) -> List[Dict]:
        """Load negative patterns from file"""
        try:
            if os.path.exists(self.negative_patterns_path):
                with open(self.negative_patterns_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return []
        except Exception as e:
            logger.error(f"Error loading negative patterns: {e}")
            return []

    def _save_negative_patterns(self):
        """Save negative patterns to file"""
        try:
            os.makedirs(os.path.dirname(self.negative_patterns_path), exist_ok=True)
            with open(self.negative_patterns_path, 'w', encoding='utf-8') as f:
                json.dump(self.negative_patterns, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Error saving negative patterns: {e}")

    def get_learning_insights(self) -> Dict[str, Any]:
        """Get insights from learning data"""
        try:
            if not self.learning_data:
                return {"message": "No learning data available"}
            
            # Analyze learning patterns
            total_improvements = len(self.learning_data)
            recent_improvements = [
                improvement for improvement in self.learning_data
                if datetime.fromisoformat(improvement['timestamp']) > datetime.now() - timedelta(days=7)
            ]
            
            # Most common improvement types
            improvement_types = defaultdict(int)
            for improvement in self.learning_data:
                suggestion = improvement.get('suggested_improvement', '')
                if 'more detailed' in suggestion:
                    improvement_types['more_detail'] += 1
                elif 'related information' in suggestion:
                    improvement_types['related_info'] += 1
                elif 'comprehensive' in suggestion:
                    improvement_types['comprehensive'] += 1
            
            return {
                "total_improvements": total_improvements,
                "recent_improvements": len(recent_improvements),
                "improvement_types": dict(improvement_types),
                "learning_rate": len(recent_improvements) / 7,  # improvements per day
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting learning insights: {e}")
            return {"error": str(e)}