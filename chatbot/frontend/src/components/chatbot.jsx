import React, { useState, useRef, useEffect } from 'react';
import { Send, Bot, User, AlertCircle, ThumbsUp, ThumbsDown, MessageCircle } from 'lucide-react';

const ChatbotInterface = () => {
    const [messages, setMessages] = useState([
        {
            id: 1,
            text: "Salut ! 👋 Je suis votre assistant IA Expert Petroleum Services. Comment puis-je vous aider aujourd'hui ?",
            sender: 'bot',
            timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
            responseId: null
        }
    ]);
    const [inputMessage, setInputMessage] = useState('');
    const [isTyping, setIsTyping] = useState(false);
    const [connectionStatus, setConnectionStatus] = useState('connected');
    const [feedbackGiven, setFeedbackGiven] = useState(new Set());
    const messagesEndRef = useRef(null);

    // Company logo component
    const CompanyLogo = () => (
        <img
            src="/logo.jpg"
            alt="Company Logo"
            className="w-12 h-12 rounded-xl shadow-lg object-cover"
        />
    );

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    // Test connection on startup
    useEffect(() => {
        const testConnection = async () => {
            try {
                const response = await fetch("http://127.0.0.1:6500/health");
                if (response.ok) {
                    setConnectionStatus('connected');
                } else {
                    setConnectionStatus('error');
                }
            } catch (error) {
                console.error("Serveur non accessible:", error);
                setConnectionStatus('disconnected');
            }
        };

        testConnection();
    }, []);

    const handleFeedback = async (messageId, responseId, feedbackType, originalQuestion, botAnswer) => {
        if (feedbackGiven.has(messageId)) return;

        try {
            const response = await fetch("http://127.0.0.1:6500/feedback", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({
                    response_id: responseId,
                    feedback_type: feedbackType,
                    question: originalQuestion,
                    answer: botAnswer,
                    model_used: "llama3.2:3b",
                    user_comment: null
                })
            });

            if (response.ok) {
                setFeedbackGiven(prev => new Set(prev).add(messageId));
                console.log(`Feedback ${feedbackType} sent successfully`);
            }
        } catch (error) {
            console.error("Error sending feedback:", error);
        }
    };

    const handleSendMessage = async () => {
        if (inputMessage.trim() === '') return;

        const currentInput = inputMessage;
        const userMessage = {
            id: messages.length + 1,
            text: currentInput,
            sender: 'user',
            timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
            responseId: null
        };

        setMessages(prev => [...prev, userMessage]);
        setInputMessage('');
        setIsTyping(true);

        try {
            console.log("Envoi de la requête vers:", "http://127.0.0.1:6500/ask");

            const response = await fetch("http://127.0.0.1:6500/ask", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                },
                body: JSON.stringify({
                    question: currentInput,
                    model: "llama3.2:3b",
                    top_k: 3
                })
            });

            console.log("Statut de réponse:", response.status);

            if (!response.ok) {
                const errorText = await response.text();
                console.error("Erreur serveur:", errorText);
                throw new Error(`HTTP error! status: ${response.status} - ${errorText}`);
            }

            const data = await response.json();
            console.log("Réponse reçue:", data);

            const botResponse = {
                id: Date.now(),
                text: data.answer,
                sender: 'bot',
                timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
                responseId: data.response_id,
                originalQuestion: currentInput
            };

            setMessages(prev => [...prev, botResponse]);
            setConnectionStatus('connected');

        } catch (error) {
            console.error("Erreur API détaillée:", error);

            let errorMessage = "Désolé, une erreur s'est produite.";

            if (error.name === 'TypeError' && error.message.includes('Failed to fetch')) {
                errorMessage = "❌ Impossible de se connecter au serveur. Vérifiez que le backend est démarré sur le port 6500.";
                setConnectionStatus('disconnected');
            } else if (error.message.includes('HTTP error')) {
                errorMessage = `❌ Erreur du serveur: ${error.message}`;
                setConnectionStatus('error');
            }

            const errorMsg = {
                id: Date.now(),
                text: errorMessage,
                sender: 'bot',
                timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
                isError: true,
                responseId: null
            };

            setMessages(prev => [...prev, errorMsg]);

        } finally {
            setIsTyping(false);
        }
    };

    const handleKeyPress = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSendMessage();
        }
    };

    const getStatusColor = () => {
        switch (connectionStatus) {
            case 'connected': return 'bg-emerald-400';
            case 'disconnected': return 'bg-red-400';
            case 'error': return 'bg-amber-400';
            default: return 'bg-gray-400';
        }
    };

    const getStatusText = () => {
        switch (connectionStatus) {
            case 'connected': return 'En ligne';
            case 'disconnected': return 'Déconnecté';
            case 'error': return 'Erreur';
            default: return 'Inconnu';
        }
    };

    return (
        <div className="flex flex-col h-screen bg-gradient-to-br from-green-50 to-emerald-50">
            {/* Header */}
            <div className="bg-white shadow-lg border-b border-green-200 p-4">
                <div className="flex items-center justify-between max-w-6xl mx-auto">
                    <div className="flex items-center space-x-4">
                        <CompanyLogo />
                        <div>
                            <h1 className="text-2xl font-bold text-gray-800">Expert Petroleum Services</h1>
                            <p className="text-sm text-green-600 font-medium">Assistant IA - Toujours prêt à vous aider</p>
                        </div>
                    </div>
                    <div className="flex items-center space-x-3">
                        <div className={`w-3 h-3 rounded-full ${getStatusColor()} ${connectionStatus === 'connected' ? 'animate-pulse' : ''}`}></div>
                        <span className="text-sm font-medium text-gray-700">{getStatusText()}</span>
                        {connectionStatus === 'disconnected' && (
                            <AlertCircle className="w-5 h-5 text-red-500" />
                        )}
                    </div>
                </div>
            </div>

            {/* Messages */}
            <div className="flex-1 overflow-y-auto p-6">
                <div className="max-w-4xl mx-auto space-y-6">
                    {messages.map((message) => (
                        <div
                            key={message.id}
                            className={`flex ${message.sender === 'user' ? 'justify-end' : 'justify-start'} animate-fade-in-up`}
                        >
                            <div className={`flex items-start space-x-4 max-w-xs md:max-w-md lg:max-w-2xl ${message.sender === 'user' ? 'flex-row-reverse space-x-reverse' : ''}`}>
                                <div className={`w-10 h-10 rounded-full flex items-center justify-center flex-shrink-0 shadow-lg ${message.sender === 'user'
                                    ? 'bg-gradient-to-r from-green-500 to-emerald-600'
                                    : message.isError
                                        ? 'bg-gradient-to-r from-red-500 to-red-600'
                                        : 'bg-gradient-to-r from-emerald-500 to-green-600'
                                    }`}>
                                    {message.sender === 'user' ? (
                                        <User className="w-5 h-5 text-white" />
                                    ) : message.isError ? (
                                        <AlertCircle className="w-5 h-5 text-white" />
                                    ) : (
                                        <Bot className="w-5 h-5 text-white" />
                                    )}
                                </div>
                                <div className="flex flex-col space-y-2">
                                    <div className={`rounded-2xl px-6 py-4 shadow-lg ${message.sender === 'user'
                                        ? 'bg-gradient-to-r from-green-500 to-emerald-600 text-white'
                                        : message.isError
                                            ? 'bg-red-50 text-red-800 border border-red-200'
                                            : 'bg-white text-gray-800 border border-green-200'
                                        }`}>
                                        <p className="text-sm leading-relaxed whitespace-pre-wrap">{message.text}</p>
                                        <p className={`text-xs mt-3 ${message.sender === 'user'
                                            ? 'text-green-100'
                                            : message.isError
                                                ? 'text-red-500'
                                                : 'text-gray-500'
                                            }`}>
                                            {message.timestamp}
                                        </p>
                                    </div>

                                    {/* Feedback buttons for bot messages */}
                                    {message.sender === 'bot' && !message.isError && message.responseId && (
                                        <div className="flex items-center space-x-2 ml-2">
                                            <span className="text-xs text-gray-500">Cette réponse était-elle utile ?</span>
                                            <button
                                                onClick={() => handleFeedback(message.id, message.responseId, 'like', message.originalQuestion, message.text)}
                                                disabled={feedbackGiven.has(message.id)}
                                                className={`p-2 rounded-full transition-all duration-200 ${feedbackGiven.has(message.id)
                                                    ? 'bg-green-100 text-green-600 cursor-default'
                                                    : 'bg-gray-100 hover:bg-green-100 text-gray-600 hover:text-green-600 hover:scale-110'
                                                    }`}
                                            >
                                                <ThumbsUp className="w-4 h-4" />
                                            </button>
                                            <button
                                                onClick={() => handleFeedback(message.id, message.responseId, 'dislike', message.originalQuestion, message.text)}
                                                disabled={feedbackGiven.has(message.id)}
                                                className={`p-2 rounded-full transition-all duration-200 ${feedbackGiven.has(message.id)
                                                    ? 'bg-red-100 text-red-600 cursor-default'
                                                    : 'bg-gray-100 hover:bg-red-100 text-gray-600 hover:text-red-600 hover:scale-110'
                                                    }`}
                                            >
                                                <ThumbsDown className="w-4 h-4" />
                                            </button>
                                        </div>
                                    )}
                                </div>
                            </div>
                        </div>
                    ))}

                    {isTyping && (
                        <div className="flex justify-start animate-fade-in-up">
                            <div className="flex items-start space-x-4">
                                <div className="w-10 h-10 rounded-full bg-gradient-to-r from-emerald-500 to-green-600 flex items-center justify-center shadow-lg">
                                    <Bot className="w-5 h-5 text-white" />
                                </div>
                                <div className="bg-white rounded-2xl px-6 py-4 shadow-lg border border-green-200">
                                    <div className="flex space-x-2">
                                        <div className="w-2 h-2 bg-emerald-500 rounded-full animate-bounce"></div>
                                        <div className="w-2 h-2 bg-emerald-500 rounded-full animate-bounce delay-100"></div>
                                        <div className="w-2 h-2 bg-emerald-500 rounded-full animate-bounce delay-200"></div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    )}
                    <div ref={messagesEndRef} />
                </div>
            </div>

            {/* Input */}
            <div className="bg-white border-t border-green-200 p-6 shadow-lg">
                <div className="max-w-4xl mx-auto">
                    <div className="flex items-end space-x-4">
                        <div className="flex-1 relative">
                            <textarea
                                value={inputMessage}
                                onChange={(e) => setInputMessage(e.target.value)}
                                onKeyDown={handleKeyPress}
                                placeholder="Posez votre question sur Expert Petroleum Services..."
                                className="w-full resize-none border-2 border-green-300 rounded-2xl px-6 py-4 pr-14 focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500 transition-all duration-200 shadow-sm text-gray-800 placeholder-gray-500"
                                rows="1"
                                style={{
                                    minHeight: '56px',
                                    maxHeight: '140px',
                                    overflow: 'auto'
                                }}
                            />
                            <MessageCircle className="absolute right-4 top-4 w-5 h-5 text-green-400" />
                        </div>
                        <button
                            onClick={handleSendMessage}
                            disabled={inputMessage.trim() === '' || connectionStatus === 'disconnected'}
                            className="bg-gradient-to-r from-emerald-500 to-green-600 hover:from-emerald-600 hover:to-green-700 disabled:from-gray-300 disabled:to-gray-400 text-white p-4 rounded-2xl transition-all duration-200 shadow-lg hover:shadow-xl disabled:cursor-not-allowed transform hover:scale-105 active:scale-95"
                        >
                            <Send className="w-6 h-6" />
                        </button>
                    </div>
                    <div className="flex items-center justify-between mt-3">
                        <p className="text-xs text-gray-500">
                            Appuyez sur <kbd className="px-2 py-1 bg-gray-100 rounded text-xs">Entrée</kbd> pour envoyer • <kbd className="px-2 py-1 bg-gray-100 rounded text-xs">Maj + Entrée</kbd> pour une nouvelle ligne
                        </p>
                        <div className="flex items-center space-x-2 text-xs text-gray-500">
                            <span>Propulsé par</span>
                            <span className="font-semibold text-emerald-600">Expert Petroleum Services</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default ChatbotInterface;