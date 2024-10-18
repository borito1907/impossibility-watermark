import React from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import Home from './Home';
import Inspect from './Inspect';
import Diff from './Diff';
import Attack from './Attack';
import Trace from './Trace';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/inspect/:id" element={<Inspect />} />
        <Route path="/diff" element={<Diff />} />
        <Route path="/attack" element={<Attack />} />
        <Route path="/trace" element={<Trace />} />
      </Routes>
    </Router>
  );
}

export default App;