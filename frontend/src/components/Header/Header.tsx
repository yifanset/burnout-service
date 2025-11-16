import classes from "./Header.module.css"
import logo from "../../assets/logo-svg-dark.svg"
import {Link} from "react-router-dom";


const Header = () => {
    return (
        <header>
            <div className={classes.headerContainer}>
                <Link to="/ ">
                    <img src={logo} alt="" width={133} height={50}/>
                </Link>

                <nav>
                    <ul>
                        <li>
                            <Link to="/">
                                О нас
                            </Link>
                        </li>
                        <li>
                            <Link to="/login">
                                Вход
                            </Link>
                        </li>
                    </ul>
                </nav>
            </div>
        </header>
    );
};

export default Header;