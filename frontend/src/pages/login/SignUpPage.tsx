import { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import classes from './LoginPage.module.css';

const SignUpPage = () => {
    const [username, setUsername] = useState('');
    const [password, setPassword] = useState('');
    const [confirmPassword, setConfirmPassword] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const navigate = useNavigate();

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        
        if (!username.trim() || !password.trim() || !confirmPassword.trim()) {
            alert('Пожалуйста, заполните все поля');
            return;
        }

        if (password !== confirmPassword) {
            alert('Пароли не совпадают');
            return;
        }

        if (password.length < 6) {
            alert('Пароль должен содержать минимум 6 символов');
            return;
        }

        setIsLoading(true);

        try {
            // Отправка данных на бэкенд для регистрации
            const response = await fetch('/api/auth/register', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ username, password }),
            });

            if (response.ok) {
                alert('Регистрация прошла успешно! Теперь вы можете войти.');
                navigate('/login');
            } else {
                const errorData = await response.json();
                alert(errorData.message || 'Ошибка регистрации. Возможно, пользователь уже существует.');
            }
        } catch (error) {
            console.error('Ошибка при регистрации:', error);
            alert('Произошла ошибка при регистрации');
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className={classes.container}>
            <div className={classes.loginCard}>
                <div className={classes.header}>
                    <h1>Регистрация</h1>
                    <p>Создайте новый аккаунт</p>
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
                            placeholder="Придумайте username"
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
                            placeholder="Придумайте пароль (мин. 6 символов)"
                            disabled={isLoading}
                            required
                            minLength={6}
                        />
                    </div>

                    <div className={classes.inputGroup}>
                        <label htmlFor="confirmPassword" className={classes.label}>
                            Подтвердите пароль
                        </label>
                        <input
                            type="password"
                            id="confirmPassword"
                            value={confirmPassword}
                            onChange={(e) => setConfirmPassword(e.target.value)}
                            className={classes.input}
                            placeholder="Повторите пароль"
                            disabled={isLoading}
                            required
                        />
                    </div>

                    <button 
                        type="submit" 
                        className={classes.submitButton}
                        disabled={isLoading}
                    >
                        {isLoading ? 'Регистрация...' : 'Зарегистрироваться'}
                    </button>
                </form>

                <div className={classes.footer}>
                    <p>Уже есть аккаунт? <Link to="/login" className={classes.link}>Войти</Link></p>
                </div>
            </div>
        </div>
    );
};

export default SignUpPage;