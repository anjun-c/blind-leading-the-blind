// import React from 'react';
// import './HeroSection.css';

// const HeroSection = () => {
//   return (
//     <section className="hero">
//       <h1>Expand Your Imagination with Luma AI</h1>
//       <p>Transform your ideas into stunning visuals with our cutting-edge AI technology.</p>
//       <button className="cta-button">Get Started</button>
//     </section>
//   );
// };

// export default HeroSection;
import React from 'react';
import { useNavigate } from 'react-router-dom';
import './HeroSection.css';

const HeroSection = () => {
  const navigate = useNavigate();

  const handleGetStarted = () => {
    navigate('/unblind');
  };

  return (
    <section className="hero">
      <h1>Enhance your conversations with Unblind</h1>
      <p>Using our Unblind AI the visually impaired can understand the full context in a converstation.</p>
      <button className="cta-button" onClick={handleGetStarted}>Get Started</button>
    </section>
  );
};

export default HeroSection;
