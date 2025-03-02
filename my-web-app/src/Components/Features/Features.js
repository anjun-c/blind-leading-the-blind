import React from 'react';
import './Features.css';

const Features = () => {
  return (
    <section id="features" className="features">
      <h2>Features</h2>
      <div className="feature-cards">
        <div className="feature-card">
          <h3>Text-to-3D</h3>
          <p>Create interactive 3D scenes from text instructions.</p>
        </div>
        <div className="feature-card">
          <h3>High-Quality Videos</h3>
          <p>Generate realistic and imaginative videos effortlessly.</p>
        </div>
        <div className="feature-card">
          <h3>3D Capture</h3>
          <p>Capture and embed 3D models on your website with ease.</p>
        </div>
      </div>
    </section>
  );
};

export default Features;
