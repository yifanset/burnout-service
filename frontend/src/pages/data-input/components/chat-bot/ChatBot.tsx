import { useState, useRef, useEffect } from 'react';
import classes from './ChatBot.module.css';

interface Message {
    id: string;
    text: string;
    isUser: boolean;
    timestamp: Date;
}

const ChatBot = () => {
    const [messages, setMessages] = useState<Message[]>([]);
    const [inputText, setInputText] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const messagesEndRef = useRef<HTMLDivElement>(null);

    // Авто-скролл к последнему сообщению
    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages]);

    // Приветственное сообщение при загрузке
    useEffect(() => {
        const welcomeMessage: Message = {
            id: '1',
            text: 'Привет! Я ваш помощник для ввода данных сотрудников. Задавайте вопросы или предоставьте информацию о сотрудниках.',
            isUser: false,
            timestamp: new Date()
        };
        setMessages([welcomeMessage]);
    }, []);

    const handleSendMessage = async () => {
        if (!inputText.trim() || isLoading) return;

        const userMessage: Message = {
            id: Date.now().toString(),
            text: inputText,
            isUser: true,
            timestamp: new Date()
        };

        setMessages(prev => [...prev, userMessage]);
        setInputText('');
        setIsLoading(true);

        try {
            // Отправка сообщения на бэкенд
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message: inputText }),
            });

            const data = await response.json();

            const botMessage: Message = {
                id: (Date.now() + 1).toString(),
                text: data.reply || 'Hello World', // Заглушка "Hello World"
                isUser: false,
                timestamp: new Date()
            };

            setMessages(prev => [...prev, botMessage]);
        } catch (error) {
            console.error('Ошибка отправки сообщения:', error);
            
            const errorMessage: Message = {
                id: (Date.now() + 1).toString(),
                text: 'Извините, произошла ошибка. Пожалуйста, попробуйте еще раз.',
                isUser: false,
                timestamp: new Date()
            };
            
            setMessages(prev => [...prev, errorMessage]);
        } finally {
            setIsLoading(false);
        }
    };

    const handleKeyPress = (e: React.KeyboardEvent) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSendMessage();
        }
    };

    const clearChat = () => {
        setMessages([]);
        // Добавляем новое приветственное сообщение после очистки
        const welcomeMessage: Message = {
            id: Date.now().toString(),
            text: 'Привет! Я ваш помощник для ввода данных сотрудников. Задавайте вопросы или предоставьте информацию о сотрудниках.',
            isUser: false,
            timestamp: new Date()
        };
        setMessages([welcomeMessage]);
    };

    return (
        <div className={classes.chatContainer}>
            <div className={classes.chatHeader}>
                <div className={classes.botInfo}>
                    <div className={classes.botAvatar}>AI</div>
                    <div>
                        <h3>Помощник по сотрудникам</h3>
                        <span className={classes.status}>Online</span>
                    </div>
                </div>
                <button 
                    className={classes.clearButton}
                    onClick={clearChat}
                    title="Очистить чат"
                >
                    🗑️
                </button>
            </div>

            <div className={classes.messagesContainer}>
                {messages.map((message) => (
                    <div
                        key={message.id}
                        className={`${classes.message} ${
                            message.isUser ? classes.userMessage : classes.botMessage
                        }`}
                    >
                        <div className={classes.messageContent}>
                            <div className={classes.messageText}>{message.text}</div>
                            <div className={classes.timestamp}>
                                {message.timestamp.toLocaleTimeString('ru-RU', {
                                    hour: '2-digit',
                                    minute: '2-digit'
                                })}
                            </div>
                        </div>
                    </div>
                ))}
                {isLoading && (
                    <div className={`${classes.message} ${classes.botMessage}`}>
                        <div className={classes.messageContent}>
                            <div className={classes.typingIndicator}>
                                <span></span>
                                <span></span>
                                <span></span>
                            </div>
                        </div>
                    </div>
                )}
                <div ref={messagesEndRef} />
            </div>

            <div className={classes.inputContainer}>
                <div className={classes.inputWrapper}>
                    <textarea
                        value={inputText}
                        onChange={(e) => setInputText(e.target.value)}
                        onKeyPress={handleKeyPress}
                        placeholder="Введите ваше сообщение..."
                        className={classes.textInput}
                        rows={1}
                        disabled={isLoading}
                    />
                    <button
                        onClick={handleSendMessage}
                        disabled={!inputText.trim() || isLoading}
                        className={classes.sendButton}
                    >
                        📤
                    </button>
                </div>
            </div>
        </div>
    );
};

export default ChatBot;