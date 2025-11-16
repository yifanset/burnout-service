import './App.css';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import LoginPage from './pages/login/LoginPage';
import SignUpPage from './pages/login/SignUpPage.tsx';
import DataInputLayout from './pages/data-input/DataInputLayout';
import SingleEmployeeForm from './pages/data-input/components/single-test/SingleEmployeeForm.tsx';
import BulkUploadForm from './pages/data-input/components/excel-upload/BulkUploadForm.tsx';
import ChatBot from './pages/data-input/components/chat-bot/ChatBot.tsx';
import Header from "./components/Header/Header.tsx";

function App() {
    return (
        <Router>
            <Header/>
            <Routes>
                <Route path="/login" element={<LoginPage />} />
                <Route path="/signup" element={<SignUpPage />} />
                <Route path="/home" element={<DataInputLayout />}>
                    <Route path="single" element={<SingleEmployeeForm />} />
                    <Route path="bulk" element={<BulkUploadForm />} />
                    <Route path="chat" element={<ChatBot />} />
                    <Route index element={<Navigate to="chat" replace />} />
                </Route>
                <Route path="*" element={<Navigate to="/home" replace />} />
            </Routes>
        </Router>
    );
}

export default App;