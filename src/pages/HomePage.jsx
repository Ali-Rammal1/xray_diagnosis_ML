// src/pages/HomePage.jsx

import React, { useContext } from 'react';
import { useNavigate } from 'react-router-dom';
import styled, { keyframes, css } from 'styled-components'; // Import css helper
import ImageUploader from '../components/ImageUploader'; // Assuming path is correct
import { ImageContext } from '../App'; // Assuming path is correct
import { analyzeWithOurAI } from '../services/api'; // Assuming path is correct
import LoadingSpinner from '../components/LoadingSpinner'; // Optional: for button loading state

// --- Animations ---
const fadeIn = keyframes`
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
`;

const pulse = keyframes`
    0% { box-shadow: 0 0 0 0 rgba(123, 104, 238, 0.6); } // Use theme primary color if possible
    70% { box-shadow: 0 0 0 12px rgba(123, 104, 238, 0); }
    100% { box-shadow: 0 0 0 0 rgba(123, 104, 238, 0); }
`;

// Float animation is defined but not used in this version. Could be added later.
const float = keyframes`
    0% { transform: translateY(0px); }
    50% { transform: translateY(-10px); }
    100% { transform: translateY(0px); }
`;

const shimmer = keyframes`
    0% { background-position: -200% 0; } // Adjusted for better visual effect
    100% { background-position: 200% 0; }
`;

// --- Styled Components ---
const HomeContainer = styled.div`
    display: flex;
    flex-direction: column;
    align-items: center;
    width: 100%;
    max-width: 1000px; // Increased max-width slightly
    margin: 0 auto;
    padding: 3rem 2rem; // Generous padding
    overflow: hidden; // Prevent animations from causing overflow issues
`;

const HeroSection = styled.div`
    text-align: center;
    margin-bottom: 4rem;
    animation: ${fadeIn} 0.8s ease-out forwards; // Use forwards to keep end state

    @media (max-width: 768px) {
        margin-bottom: 2.5rem;
    }
`;

const Title = styled.h1`
    font-size: 3.5rem; // Larger title
    margin-bottom: 1.5rem;
    text-align: center;
    // More vibrant gradient using theme colors if possible, otherwise fallback
    background: linear-gradient(135deg, ${({ theme }) => theme.colors.primary || '#9d50bb'} 0%, ${({ theme }) => theme.colors.primaryHover || '#6e48aa'} 50%, #4776E6 100%);
    background-size: 200% auto; // For shimmer effect
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800; // Bold weight
    letter-spacing: -0.03em;
    line-height: 1.1; // Tight line height
    animation: ${shimmer} 4s linear infinite; // Slower shimmer

    @media (max-width: 768px) {
        font-size: 2.8rem; // Adjust mobile size
    }
     @media (max-width: 480px) {
        font-size: 2.2rem;
    }
`;

const Description = styled.p`
    font-size: 1.25rem; // Slightly larger description
    text-align: center;
    margin: 0 auto 2.5rem auto; // Center horizontally
    max-width: 700px;
    color: ${({ theme }) => theme.colors.textSecondary};
    line-height: 1.7; // Improved readability

    @media (max-width: 768px) {
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
`;

const UploadSection = styled.div`
    width: 100%;
    max-width: 650px;
    background: ${({ theme }) => theme.colors.cardBackground};
    border-radius: 24px; // More pronounced radius
    padding: 2.5rem;
    margin-bottom: 2.5rem; // Reduced margin slightly
    box-shadow: ${({ theme }) => theme.shadows.lg}; // Use theme shadow
    // Apply fadeIn animation with delay
    opacity: 0; // Start hidden for animation
    animation: ${fadeIn} 0.8s ease-out 0.2s forwards; // Delay and keep end state
    transition: transform 0.3s ease, box-shadow 0.3s ease;

    &:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.2); // Enhanced hover shadow
    }

    display: flex;
    flex-direction: column;
    align-items: center; // Centers the ImageUploader

    @media (max-width: 768px) {
        padding: 2rem 1.5rem; // Adjust mobile padding
        border-radius: 16px;
    }
`;

// Action Button - Completed Styling
const ActionButton = styled.button`
    background: linear-gradient(to right, ${({ theme }) => theme.colors.primary}, ${({ theme }) => theme.colors.primaryHover});
    color: white;
    font-size: 1.1rem;
    font-weight: 600;
    padding: 0.9rem 2.2rem; // Adjusted padding slightly
    border: none;
    border-radius: 12px; // Consistent radius
    cursor: pointer;
    transition: all 0.3s ease;
    max-width: 300px; // Constrain width
    margin: 1rem auto 0 auto; // Margin top added, centered
    display: flex; // Use flex for centering icon/text if needed
    align-items: center;
    justify-content: center;
    position: relative; // Needed for pseudo-elements and pulse
    overflow: hidden; // Contain effects

    // Subtle light sweep effect on hover
    &:before {
        content: '';
        position: absolute;
        top: 0;
        left: -80%; // Start off-screen
        width: 60%; // Width of the sweep gradient
        height: 100%;
        background: linear-gradient(
            to right,
            rgba(255, 255, 255, 0) 0%,
            rgba(255, 255, 255, 0.25) 50%, // Brighter highlight
            rgba(255, 255, 255, 0) 100%
        );
        transform: skewX(-25deg); // Angled sweep
        transition: left 0.7s ease-in-out; // Smoother transition
    }

    &:hover:not(:disabled):before {
        left: 120%; // Move across the button
    }

    // Transform and shadow on hover (when enabled)
    &:hover:not(:disabled) {
        transform: translateY(-3px) scale(1.03); // Add slight scale
        box-shadow: 0 12px 24px rgba(123, 104, 238, 0.35); // Stronger hover shadow
    }

    // Active state (when clicked)
    &:active:not(:disabled) {
        transform: translateY(-1px) scale(1.01);
        box-shadow: 0 6px 12px rgba(123, 104, 238, 0.25);
    }

    // Disabled state styling
    &:disabled {
        opacity: 0.5;
        cursor: not-allowed;
        background: ${({ theme }) => theme.colors.backgroundHover}; // Muted background when disabled
        box-shadow: none; // Remove shadow when disabled
    }

    // Conditionally apply pulse animation ONLY when button is enabled AND not loading
    ${({ disabled, loading, theme }) => !disabled && !loading && css`
        animation: ${pulse} 2s infinite cubic-bezier(0.66, 0, 0, 1); // Use cubic-bezier for smoother pulse

        // Optional: Pause animation on hover for better interaction
        &:hover {
             /* animation-play-state: paused; */ // Uncomment if you want pulse to pause on hover
        }
    `}
`;

// Simple spinner for loading state inside the button
const ButtonSpinner = styled.div`
  width: 18px;
  height: 18px;
  border: 2px solid rgba(255, 255, 255, 0.5);
  border-top-color: white;
  border-radius: 50%;
  animation: spin 1s linear infinite;
  margin-right: 0.5rem; // Space between spinner and text

  @keyframes spin {
    to { transform: rotate(360deg); }
  }
`;

// --- HomePage Component Logic ---
const HomePage = () => {
    // Use context to manage image data, analysis results, and loading state
    const { imageData, setAnalysisResult, setLoading, loading } = useContext(ImageContext);
    const navigate = useNavigate();

    // Handler for the analyze button click
    const handleAnalyzeWithOurAI = async () => {
        // Prevent action if no image is selected or if already loading
        if (!imageData || loading) return;

        setLoading(true); // Set loading state to true
        setError(''); // Clear any previous errors (if error state is added later)
        try {
            // Call the API function to analyze the image
            const result = await analyzeWithOurAI(imageData);
            setAnalysisResult(result); // Store the analysis result in context
            navigate('/results'); // Navigate to the results page
        } catch (error) {
            console.error('Analysis failed:', error);
            // Basic error handling - consider replacing alert with an inline message
            alert(`Analysis failed: ${error.message}. Please check the console or try again.`);
            setError(`Analysis failed: ${error.message}`); // Set error state if added
        } finally {
            setLoading(false); // Ensure loading state is set to false after completion or error
        }
    };

    // Placeholder for error state (optional, replace alert with this)
    const [error, setError] = React.useState('');

    return (
        <HomeContainer>
            <HeroSection>
                <Title>AI-Powered X-Ray Analysis</Title>
                <Description>
                    Upload your chest X-ray image. Our advanced AI will analyze it for potential
                    conditions like Pneumonia and Tuberculosis, providing insights quickly and efficiently.
                </Description>
            </HeroSection>

            <UploadSection>
                {/* The ImageUploader component handles file selection and preview */}
                <ImageUploader />
                {/* Optional: Display error message related to upload here */}
            </UploadSection>

            {/* Display error message related to analysis */}
            {error && <p style={{ color: 'red', marginTop: '1rem' }}>{error}</p>}

            {/* Analysis Button: Shows loading state and is disabled when appropriate */}
            <ActionButton
                onClick={handleAnalyzeWithOurAI}
                disabled={!imageData || loading} // Disable button if no image or loading
            >
                {loading ? (
                    <>
                        <ButtonSpinner />
                        Analyzing...
                    </>
                ) : (
                    'Analyze with Our AI'
                )}
            </ActionButton>
        </HomeContainer>
    );
};

export default HomePage;