import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import HomePage from './pages/HomePage';
import ResultsPage from './pages/ResultsPage';
import NotFoundPage from './pages/NotFoundPage';
import Header from './components/Header';
import Footer from './components/Footer';
import { AppContainer, MainContent } from './styles/componentStyles';

export const ImageContext = React.createContext();

function App() {
    const [imageData, setImageData] = useState(null);
    const [analysisResult, setAnalysisResult] = useState(null);
    const [loading, setLoading] = useState(false);

    return (
        <ImageContext.Provider value={{
            imageData,
            setImageData,
            analysisResult,
            setAnalysisResult,
            loading,
            setLoading
        }}>
            <Router>
                <AppContainer>
                    <Header />
                    <MainContent>
                        <Routes>
                            <Route path="/" element={<HomePage />} />
                            <Route
                                path="/results"
                                element={
                                    imageData ? <ResultsPage /> : <Navigate to="/" replace />
                                }
                            />
                            <Route path="*" element={<NotFoundPage />} />
                        </Routes>
                    </MainContent>
                    <Footer />
                </AppContainer>
            </Router>
        </ImageContext.Provider>
    );
}

export default App;