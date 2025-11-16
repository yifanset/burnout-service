import type { ReactNode } from 'react';
import classes from "./QuestionSection.module.css";

interface QuestionSectionProps {
    children: ReactNode;
    title?: string;
}

const QuestionSection = ({ children, title }: QuestionSectionProps) => {
    return (
        <section className={classes.section}>
            {title && <h3 className={classes.sectionTitle}>{title}</h3>}
            {children}
        </section>
    );
};

export default QuestionSection;