// import React from 'react';
// import './Features.css';

// const Features = () => {
//   return (
//     <section id="features" className="features">
//       <h2>About</h2>
//       <div className="feature-cards">
//         <div className="feature-card">
//           <h3>Text-to-3D</h3>
//           <p>Create interactive 3D scenes from text instructions.</p>
//         </div>
//         <div className="feature-card">
//           <h3>High-Quality Videos</h3>
//           <p>Generate realistic and imaginative videos effortlessly.</p>
//         </div>
//         <div className="feature-card">
//           <h3>3D Capture</h3>
//           <p>Capture and embed 3D models on your website with ease.</p>
//         </div>
//       </div>
//     </section>
//   );
// };

// export default Features;

import React from 'react';
import './Features.css'; // Optional: Create a CSS file for styling

const About = () => {
  return (
    <section className="about-section">
      <h2>About Us</h2>
      <p>
        Welcome to our platform! We are dedicated to helping the visually impaired and have created
        an AI that can help them understand the full context of a conversation through relaying the
        facial expressions of the person they are talking to.
      </p>
      <p>
        Our mission is to seemlessly integrate our AI into everyday life so that the visually impaired
        can have a more enriching experience and won't have to worry about missing out on any social cues.
      </p>
    </section>
  );
};

export default About;
