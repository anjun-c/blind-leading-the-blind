

// import React from 'react';
// import Header from './Components/Header/Header';
// import HeroSection from './Components/HeroSection/HeroSection';
// import Features from './Components/Features/Features';
// //import Gallery from './Components/Gallery/Gallery';
// import Footer from './Components/Footer/Footer';
// import UnblindComponent from './Components/UnblindComponent/UnblindComponent';
// import './App.css';
// //import './global.css'; // Import global styles

// const App = () => {
//   return (
//     <div className="App">
//       <Header />
//       <HeroSection />
//       <Features />
//       <UnblindComponent />
//       {/* <Gallery /> */}
//       <Footer />
//     </div>
//   );
// };

// export default App;

import React from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import Header from './Components/Header/Header';
import HeroSection from './Components/HeroSection/HeroSection';
import Features from './Components/Features/Features';
//import Gallery from './Components/Gallery/Gallery';
// import Footer from './Components/Footer/Footer';
import UnblindPage from './Components/UnblindPage/UnblindPage'; 
import './App.css';

const App = () => {
  return (
    <Router>
      <div className="App">
        <Header />
        <Routes>
          <Route path="/" element={
            <>
              <HeroSection />
              <Features />
            </>
          } />
          <Route path="/unblind" element={<UnblindPage />} />
        </Routes>
        {/* <Footer /> */}
      </div>
    </Router>
  );
};

export default App;
