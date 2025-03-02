// import React from 'react';
// import './Header.css';

// const Header = () => {
//   return (
//     <header className="header">
//     <flex>
//       <div className="logo">Unblind</div>
//     </flex>
//       <nav className="nav">
//         {/* <a href="#features">About</a>
//         <a href="#gallery">Gallery</a>
//         <a href="#contact">Contact</a> */}
//       </nav>
//     </header>
//   );
// };

// export default Header;

import React from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { FaArrowLeft } from 'react-icons/fa'; // Import the arrow icon
import './Header.css'; // Ensure you have a CSS file for the header

const Header = () => {
  const location = useLocation();
  const navigate = useNavigate();

  const handleBack = () => {
    navigate('/'); // Navigate back to the home page
  };

  return (
    <header className="header">
      {location.pathname === '/unblind' && (
        <button className="back-button" onClick={handleBack}>
          <FaArrowLeft /> {/* Use the arrow icon */}
        </button>
      )}
      <h1 className="logo">Unblind</h1> 
    </header>
  );
};

export default Header;

