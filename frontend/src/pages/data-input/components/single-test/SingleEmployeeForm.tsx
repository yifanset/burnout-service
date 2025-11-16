import type { ChangeEvent, FormEvent } from "react";
import CustomSelect from "../../../../components/CustomSelect/CustomSelect"
import { useState, useMemo } from "react";
import classes from "./SingleEmployeeForm.module.css";
import QuestionSection from "../../../../components/QuestionSection/QuestionSection";
import type { FormData } from "../../../../types";


const SingleEmployeeForm = () => {
    const [formData, setFormData] = useState<FormData>({
        fullName: "",
        gender: "",
        city: "",
        position: "",
        experience: "",
        age: "",
        positionType: "",
        certification: "",
        training: "",
        lastVacation: "",
        kpiMonths: "",
        sickLeave: "",
        reprimands: "",
        corporateEvents: ""
    });

    const kpiMonthFields = useMemo(() => {
        const monthsCount = Math.min(parseInt(formData.kpiMonths) || 0, 4);
        const months = [];
        const currentDate = new Date();
        
        for (let i = 0; i < monthsCount; i++) {
            const date = new Date(currentDate);
            date.setMonth(currentDate.getMonth() - i);
            const monthName = date.toLocaleString('ru-RU', { month: 'long' });
            const year = date.getFullYear();
            const fieldName = `kpi_${date.getMonth() + 1}_${year}`;
            
            months.push({
                number: i + 1,
                name: `${monthName} ${year}`,
                fieldName: fieldName
            });
        }
        
        return months;
    }, [formData.kpiMonths]);

    const handleChange = (e: ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
        const { name, value } = e.target;
        setFormData(prev => ({
            ...prev,
            [name]: value
        }));
    };

    const handleKpiChange = (value: string) => {
        setFormData(prev => ({
            ...prev,
            kpiMonths: value
        }));
    };

    const handleKpiInputChange = (e: ChangeEvent<HTMLInputElement>) => {
        const { name, value } = e.target;
        setFormData(prev => ({
            ...prev,
            [name]: value
        }));
    };

    const handleSubmit = (e: FormEvent) => {
        e.preventDefault();
        console.log("Данные сотрудника:", formData);
        alert("Данные сотрудника отправлены!");
    };

    return (
        <div className={classes.page}>
            <form className={classes.form} onSubmit={handleSubmit}>
                <h2 className={classes.title}>Быстрый тест</h2>
                <div className={classes.grid}>
                    {/* Ряд 1 */}
                    <QuestionSection title="1. ФИО сотрудника">
                        <input 
                            type="text" 
                            name="fullName"
                            placeholder="ФИО сотрудника"
                            value={formData.fullName}
                            onChange={handleChange}
                            className={classes.input}
                            required
                        />
                    </QuestionSection>

                    <QuestionSection title="2. Пол">
                        <div className={classes.radioGroup}>
                            <label className={classes.radioLabel}>
                                <input 
                                    type="radio" 
                                    name="gender" 
                                    value="male"
                                    checked={formData.gender === "male"}
                                    onChange={handleChange}
                                />
                                Мужской
                            </label>
                            <label className={classes.radioLabel}>
                                <input 
                                    type="radio" 
                                    name="gender" 
                                    value="female"
                                    checked={formData.gender === "female"}
                                    onChange={handleChange}
                                />
                                Женский
                            </label>
                        </div>
                    </QuestionSection>

                    <QuestionSection title="3. Город">
                        <input 
                            type="text" 
                            name="city"
                            placeholder="Город сотрудника"
                            value={formData.city}
                            onChange={handleChange}
                            className={classes.input}
                            required
                        />
                    </QuestionSection>

                    {/* Ряд 2 */}
                    <QuestionSection title="4. Должность">
                        <input 
                            type="text" 
                            name="position"
                            placeholder="Должность сотрудника"
                            value={formData.position}
                            onChange={handleChange}
                            className={classes.input}
                            required
                        />
                    </QuestionSection>

                    <QuestionSection title="5. Стаж (лет)">
                        <input 
                            type="number" 
                            name="experience"
                            placeholder="Стаж"
                            value={formData.experience}
                            onChange={handleChange}
                            className={classes.input}
                            min="0"
                            max="50"
                            required
                        />
                    </QuestionSection>

                    <QuestionSection title="6. Возраст">
                        <input 
                            type="number" 
                            name="age"
                            placeholder="Возраст"
                            value={formData.age}
                            onChange={handleChange}
                            className={classes.input}
                            min="18"
                            max="70"
                            required
                        />
                    </QuestionSection>

                    {/* Ряд 3 */}
                    <QuestionSection title="7. Тип должности">
                        <div className={classes.radioGroup}>
                            <label className={classes.radioLabel}>
                                <input 
                                    type="radio" 
                                    name="positionType" 
                                    value="manager"
                                    checked={formData.positionType === "manager"}
                                    onChange={handleChange}
                                />
                                Руководитель
                            </label>
                            <label className={classes.radioLabel}>
                                <input 
                                    type="radio" 
                                    name="positionType" 
                                    value="employee"
                                    checked={formData.positionType === "employee"}
                                    onChange={handleChange}
                                />
                                Сотрудник
                            </label>
                        </div>
                    </QuestionSection>

                    <QuestionSection title="8. Прошли ли вы аттестацию?">
                        <div className={classes.radioGroup}>
                            <label className={classes.radioLabel}>
                                <input 
                                    type="radio" 
                                    name="certification" 
                                    value="yes"
                                    checked={formData.certification === "yes"}
                                    onChange={handleChange}
                                />
                                Да
                            </label>
                            <label className={classes.radioLabel}>
                                <input 
                                    type="radio" 
                                    name="certification" 
                                    value="no"
                                    checked={formData.certification === "no"}
                                    onChange={handleChange}
                                />
                                Нет
                            </label>
                        </div>
                    </QuestionSection>

                    <QuestionSection title="9. Прошли ли вы обучение?">
                        <div className={classes.radioGroup}>
                            <label className={classes.radioLabel}>
                                <input 
                                    type="radio" 
                                    name="training" 
                                    value="yes"
                                    checked={formData.training === "yes"}
                                    onChange={handleChange}
                                />
                                Да
                            </label>
                            <label className={classes.radioLabel}>
                                <input 
                                    type="radio" 
                                    name="training" 
                                    value="no"
                                    checked={formData.training === "no"}
                                    onChange={handleChange}
                                />
                                Нет
                            </label>
                        </div>
                    </QuestionSection>

                    {/* Ряд 4 */}
                    <QuestionSection title="10. Дата последнего отпуска">
                        <input 
                            type="date" 
                            name="lastVacation"
                            value={formData.lastVacation}
                            onChange={handleChange}
                            className={classes.input}
                            required
                        />
                    </QuestionSection>

                    <QuestionSection title="11. За сколько месяцев можете предоставить KPI?">
                        <CustomSelect
                            value={formData.kpiMonths}
                            onChange={handleKpiChange}
                            options={[
                                { value: "", label: "Не выбрано" },
                                { value: "1", label: "1 месяц" },
                                { value: "2", label: "2 месяца" },
                                { value: "3", label: "3 месяца" },
                                { value: "4", label: "4 месяца" }
                            ]}
                        />
                    </QuestionSection>

                    {/* Динамические KPI поля - сразу после выбора количества месяцев */}
                    {kpiMonthFields.map((month, index) => (
                        <QuestionSection key={month.fieldName} title={`11.${month.number} KPI за ${month.name}`}>
                            <div className={classes.kpiInputWrapper}>
                                <input 
                                    type="number" 
                                    name={month.fieldName}
                                    value={formData[month.fieldName] || ""}
                                    onChange={handleKpiInputChange}
                                    className={classes.input}
                                    placeholder="0"
                                    min="0"
                                    max="100"
                                    step="0.1"
                                />
                            </div>
                        </QuestionSection>
                    ))}
                    {/* Остальные вопросы с правильной нумерацией */}
                    <QuestionSection title={`${12 + kpiMonthFields.length}. Брал ли больничный за последний год?`}>
                        <div className={classes.radioGroup}>
                            <label className={classes.radioLabel}>
                                <input 
                                    type="radio" 
                                    name="sickLeave" 
                                    value="yes"
                                    checked={formData.sickLeave === "yes"}
                                    onChange={handleChange}
                                />
                                Да
                            </label>
                            <label className={classes.radioLabel}>
                                <input 
                                    type="radio" 
                                    name="sickLeave" 
                                    value="no"
                                    checked={formData.sickLeave === "no"}
                                    onChange={handleChange}
                                />
                                Нет
                            </label>
                        </div>
                    </QuestionSection>

                    <QuestionSection title={`${13 + kpiMonthFields.length}. Имеет ли выговоры?`}>
                        <div className={classes.radioGroup}>
                            <label className={classes.radioLabel}>
                                <input 
                                    type="radio" 
                                    name="reprimands" 
                                    value="yes"
                                    checked={formData.reprimands === "yes"}
                                    onChange={handleChange}
                                />
                                Да
                            </label>
                            <label className={classes.radioLabel}>
                                <input 
                                    type="radio" 
                                    name="reprimands" 
                                    value="no"
                                    checked={formData.reprimands === "no"}
                                    onChange={handleChange}
                                />
                                Нет
                            </label>
                        </div>
                    </QuestionSection>

                    <QuestionSection title={`${14 + kpiMonthFields.length}. Участие в корпоративах`}>
                        <div className={classes.radioGroup}>
                            <label className={classes.radioLabel}>
                                <input 
                                    type="radio" 
                                    name="corporateEvents" 
                                    value="yes"
                                    checked={formData.corporateEvents === "yes"}
                                    onChange={handleChange}
                                />
                                Да
                            </label>
                            <label className={classes.radioLabel}>
                                <input 
                                    type="radio" 
                                    name="corporateEvents" 
                                    value="no"
                                    checked={formData.corporateEvents === "no"}
                                    onChange={handleChange}
                                />
                                Нет
                            </label>
                        </div>
                    </QuestionSection>
                </div>

                <button type="submit" className={classes.submitButton}>
                    Отправить опросник
                </button>
            </form>
        </div>
    );
};

export default SingleEmployeeForm;