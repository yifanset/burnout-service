// components/CustomSelect.tsx
import { useState, useRef, useEffect } from 'react';
import classes from './CustomSelect.module.css';

interface Option {
    value: string;
    label: string;
}

interface CustomSelectProps {
    value: string;
    onChange: (value: string) => void;
    options: Option[];
    placeholder?: string;
}

const CustomSelect = ({ value, onChange, options, placeholder = "Выберите период" }: CustomSelectProps) => {
    const [isOpen, setIsOpen] = useState(false);
    const selectRef = useRef<HTMLDivElement>(null);

    const selectedOption = options.find(opt => opt.value === value);

    useEffect(() => {
        const handleClickOutside = (event: MouseEvent) => {
            if (selectRef.current && !selectRef.current.contains(event.target as Node)) {
                setIsOpen(false);
            }
        };

        document.addEventListener('mousedown', handleClickOutside);
        return () => document.removeEventListener('mousedown', handleClickOutside);
    }, []);

    return (
        <div className={classes.customSelect} ref={selectRef}>
            <div 
                className={`${classes.selectHeader} ${isOpen ? classes.open : ''}`}
                onClick={() => setIsOpen(!isOpen)}
            >
                <span>{selectedOption ? selectedOption.label : placeholder}</span>
                <svg className={classes.arrow} width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M7 10l5 5 5-5z"/>
                </svg>
            </div>
            
            {isOpen && (
                <div className={classes.selectOptions}>
                    {options.map(option => (
                        <div
                            key={option.value}
                            className={`${classes.option} ${value === option.value ? classes.selected : ''}`}
                            onClick={() => {
                                onChange(option.value);
                                setIsOpen(false);
                            }}
                        >
                            {option.label}
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
};

export default CustomSelect;