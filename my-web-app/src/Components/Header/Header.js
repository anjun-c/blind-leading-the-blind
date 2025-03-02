import React from 'react';
import './Header.css';

const Header = () => {
  return (
    <header className="header">
    <flex>
      <div className="logo">Unblind</div>
    </flex>
      <nav className="nav">
        {/* <a href="#features">About</a>
        <a href="#gallery">Gallery</a>
        <a href="#contact">Contact</a> */}
      </nav>
    </header>
  );
};

export default Header;

