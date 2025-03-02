// import React from "react";
// import Topbar from "./Components/topbar";

// function App() {
//   return (
//     <div>
//       <Topbar />
//     </div>
//   );
// }

// export default App;

import React from 'react';
import Header from './Components/Header/Header';
import HeroSection from './Components/HeroSection/HeroSection';
import Features from './Components/Features/Features';
import Gallery from './Components/Gallery/Gallery';
import Footer from './Components/Footer/Footer';
import './App.css';
//import './global.css'; // Import global styles

const App = () => {
  return (
    <div className="App">
      <Header />
      <HeroSection />
      <Features />
      <Gallery />
      <Footer />
    </div>
  );
};

export default App;
