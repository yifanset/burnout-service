import { NavLink, Outlet } from "react-router-dom";
import classes from "./DataInputLayout.module.css";

const DataInputLayout = () => {
    return (
        <div className={classes.container}>
            <div className={classes.header}>
                <h1>Ввод данных сотрудников</h1>
                <nav className={classes.nav}>
                    <NavLink 
                        to="/home/chat" 
                        className={({ isActive }) => 
                            `${classes.navLink} ${isActive ? classes.active : ''}`
                        }
                    >
                        Чат-бот
                    </NavLink>
                    <NavLink 
                        to="/home/single" 
                        className={({ isActive }) => 
                            `${classes.navLink} ${isActive ? classes.active : ''}`
                        }
                    >
                        Тест
                    </NavLink>
                    <NavLink 
                        to="/home/bulk" 
                        className={({ isActive }) => 
                            `${classes.navLink} ${isActive ? classes.active : ''}`
                        }
                    >
                        Загрузить файл
                    </NavLink>
                    
                </nav>
            </div>
            <div className={classes.content}>
                <Outlet />
            </div>
        </div>
    );
};

export default DataInputLayout;