import { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import classes from './LoginPage.module.css';

const LoginPage = () => {
    const [username, setUsername] = useState('');
    const [password, setPassword] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const navigate = useNavigate();

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        
        if (!username.trim() || !password.trim()) {
            alert('Пожалуйста, заполните все поля');
            return;
        }

        setIsLoading(true);

        try {
            // Отправка данных на бэкенд
            const response = await fetch('/api/auth/login', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ username, password }),
            });

            if (response.ok) {
                const data = await response.json();
                // Сохраняем токен (если используется)
                if (data.token) {
                    localStorage.setItem('authToken', data.token);
                }
                alert('Вход выполнен успешно!');
                navigate('/home');
            } else {
                alert('Ошибка входа. Проверьте username и пароль.');
            }
        } catch (error) {
            console.error('Ошибка при входе:', error);
            alert('Произошла ошибка при входе');
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className={classes.container}>
            <div className={classes.loginCard}>
                <div className={classes.header}>
                    <h1>Вход в систему</h1>
                    <p>Введите ваши учетные данные</p>
                </div>

                <form onSubmit={handleSubmit} className={classes.form}>
                    <div className={classes.inputGroup}>
                        <label htmlFor="username" className={classes.label}>
                            Имя пользователя
                        </label>
                        <input
                            type="text"
                            id="username"
                            value={username}
                            onChange={(e) => setUsername(e.target.value)}
                            className={classes.input}
                            placeholder="Введите username"
                            disabled={isLoading}
                            required
                        />
                    </div>

                    <div className={classes.inputGroup}>
                        <label htmlFor="password" className={classes.label}>
                            Пароль
                        </label>
                        <input
                            type="password"
                            id="password"
                            value={password}
                            onChange={(e) => setPassword(e.target.value)}
                            className={classes.input}
                            placeholder="Введите пароль"
                            disabled={isLoading}
                            required
                        />
                    </div>

                    <button 
                        type="submit" 
                        className={classes.submitButton}
                        disabled={isLoading}
                    >
                        {isLoading ? 'Вход...' : 'Войти'}
                    </button>
                </form>

                <div className={classes.footer}>
                    <p>Нет аккаунта? <Link to="/signup" className={classes.link}>Зарегистрироваться</Link></p>
                </div>
            </div>
        </div>
    );
};

export default LoginPage;