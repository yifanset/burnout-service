import { useState, useRef } from "react";
import classes from "./BulkUploadForm.module.css";

const BulkUploadForm = () => {
    const [isLoading, setIsLoading] = useState(false);
    const fileInputRef = useRef<HTMLInputElement>(null);

    const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (!file) return;

        setIsLoading(true);

        const formData = new FormData();
        formData.append("file", file);

        fetch("/api/upload-employees", {
            method: "POST",
            body: formData,
        })
        .then(response => response.json())
        .then(() => {
            setIsLoading(false);
            alert("Файл успешно загружен на сервер");
        })
        .catch(error => {
            console.error("Ошибка загрузки:", error);
            setIsLoading(false);
            alert("Ошибка при загрузке файла");
        });
    };

    const handleDragOver = (event: React.DragEvent) => {
        event.preventDefault();
    };

    const downloadExampleFile = async () => {
        try {
            const response = await fetch('/example.xlsx');
            if (!response.ok) {
                throw new Error('Файл не найден');
            }
            
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.href = url;
            link.download = 'example.xlsx';
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            window.URL.revokeObjectURL(url);
            
            alert('Файл example.xlsx успешно скачан');
        } catch (error) {
            console.error('Ошибка при скачивании файла:', error);
            alert('Ошибка при скачивании файла');
        }
    };

    const handleDrop = (event: React.DragEvent) => {
        event.preventDefault();
        const file = event.dataTransfer.files[0];
        if (file && fileInputRef.current) {
            fileInputRef.current.files = event.dataTransfer.files;
            handleFileUpload({ target: { files: event.dataTransfer.files } } as any);
        }
    };


    return (
        <div>
            <h2 className={classes.title}>Массовая загрузка сотрудников</h2>

            <div className={classes.uploadSection}>
                <div 
                    className={classes.dropZone}
                    onDragOver={handleDragOver}
                    onDrop={handleDrop}
                    onClick={() => fileInputRef.current?.click()}
                >
                    <input
                        ref={fileInputRef}
                        type="file"
                        accept=".xlsx,.xls"
                        onChange={handleFileUpload}
                        className={classes.fileInput}
                    />
                    <div className={classes.dropZoneContent}>
                        <div className={classes.uploadIcon}>📊</div>
                        <h3>Загрузите Excel файл</h3>
                        <p>Перетащите файл сюда или нажмите для выбора</p>
                        <small>Файл будет отправлен на обработку</small>
                    </div>
                </div>

                {isLoading && (
                    <div className={classes.loading}>
                        <div className={classes.spinner}></div>
                        <p>Обработка файла...</p>
                    </div>
                )}
            </div>

            <div className={classes.template}>
                <h4>Шаблон Excel файла</h4>
                <p>Скачайте шаблон для заполнения данных:</p>
                <button 
                    className={classes.templateButton}
                    onClick={downloadExampleFile}
                >
                    📥 Скачать шаблон
                </button>
            </div>
        </div>
    );
};

export default BulkUploadForm;

