import React from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import { useSelector } from 'react-redux';

import Navbar from './components/Navbar';
import Footer from './components/Footer';
import Home from './pages/Home';
import Login from './pages/Login';
import Register from './pages/Register';
import Profile from './pages/Profile';
import VideoUpload from './pages/VideoUpload';
import VideoList from './pages/VideoList';
import VideoDetail from './pages/VideoDetail';
import MatchList from './pages/MatchList';
import MatchDetail from './pages/MatchDetail';
import CreateMatch from './pages/CreateMatch';
import Leaderboard from './pages/Leaderboard';
import NotFound from './pages/NotFound';

function App() {
  const { isAuthenticated } = useSelector(state => state.auth);

  return (
    <div className="App">
      <Navbar />
      <main className="main-content">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/login" element={<Login />} />
          <Route path="/register" element={<Register />} />
          <Route path="/videos" element={<VideoList />} />
          <Route path="/videos/:id" element={<VideoDetail />} />
          <Route path="/matches" element={<MatchList />} />
          <Route path="/matches/:id" element={<MatchDetail />} />
          <Route path="/leaderboard" element={<Leaderboard />} />
          
          {/* 需要登录的路由 */}
          <Route 
            path="/profile" 
            element={isAuthenticated ? <Profile /> : <Navigate to="/login" />} 
          />
          <Route 
            path="/upload" 
            element={isAuthenticated ? <VideoUpload /> : <Navigate to="/login" />} 
          />
          <Route 
            path="/create-match" 
            element={isAuthenticated ? <CreateMatch /> : <Navigate to="/login" />} 
          />
          
          <Route path="*" element={<NotFound />} />
        </Routes>
      </main>
      <Footer />
    </div>
  );
}

export default App;