// src/pages/HomePage.jsx

import React, { useContext } from 'react';
import { useNavigate } from 'react-router-dom';
import styled from 'styled-components';
import ImageUploader from '../components/ImageUploader';
import { ImageContext } from '../App';
import { analyzeWithOurAI } from '../services/api';

const HomeContainer = styled.div`
    display: flex;
    flex-direction: column;
    align-items: center;
    width: 100%;
    max-width: 900px;
    margin: 0 auto;
    padding: 2rem;
`;

const HeroSection = styled.div`
    text-align: center;
    margin-bottom: 3rem;
    animation: fadeIn 0.8s ease-out;

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
`;

const Title = styled.h1`
    font-size: 3rem;
    margin-bottom: 1.5rem;
    text-align: center;
    background: linear-gradient(135deg, #9d50bb 0%, #6e48aa 50%, #4776E6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700;
`;

const Description = styled.p`
    font-size: 1.2rem;
    text-align: center;
    margin-bottom: 2.5rem;
    max-width: 650px;
    color: ${({ theme }) => theme.colors.textSecondary};
    line-height: 1.6;
`;

const UploadSection = styled.div`
    width: 100%;
    max-width: 600px;
    background: ${({ theme }) => theme.colors.cardBackground};
    border-radius: 16px;
    padding: 2rem;
    margin-bottom: 2rem;
    box-shadow: ${({ theme }) => theme.shadows.lg};
    animation: slideUp 0.8s ease-out;
    animation-delay: 0.2s;
    animation-fill-mode: both;
    // Add display flex to help center children if needed, though margin auto on button is preferred
    display: flex;
    flex-direction: column;
    align-items: center; // Centers children like ImageUploader if they don't have width: 100%

    @keyframes slideUp {
        from { opacity: 0; transform: translateY(40px); }
        to { opacity: 1; transform: translateY(0); }
    }
`;

const ActionButton = styled.button`
    background: linear-gradient(to right, ${({ theme }) => theme.colors.primary}, ${({ theme }) => theme.colors.primaryHover});
    color: white;
    font-size: 1.1rem;
    font-weight: 600;
    padding: 0.9rem 2rem;
    border: none;
    border-radius: 8px;
    cursor: pointer;
    transition: all 0.3s ease;
    // Remove width: 100% if you want it to size to content + padding
    // width: 100%; // Keep if you want it constrained by max-width
    max-width: 300px; // Button won't exceed this width

    // *** CHANGE IS HERE ***
    margin-top: 1.5rem;
    margin-left: auto;  // Added
    margin-right: auto; // Added
    display: block; // Ensure it's block or inline-block for margin auto to work

    &:hover {
        transform: translateY(-2px);
        box-shadow: 0 7px 14px rgba(50, 50, 93, 0.1), 0 3px 6px rgba(0, 0, 0, 0.08);
    }

    &:active {
        transform: translateY(1px);
    }

    &:disabled {
        background: ${({ theme }) => theme.colors.textTertiary};
        cursor: not-allowed;
        transform: none;
        box-shadow: none;
        // Keep margin auto even when disabled
        margin-left: auto;
        margin-right: auto;
    }
`;


const FeatureGrid = styled.div`
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 2rem;
    width: 100%;
    margin-top: 3rem;
    animation: fadeIn 0.8s ease-out;
    animation-delay: 0.4s;
    animation-fill-mode: both;
`;

const FeatureCard = styled.div`
    background: ${({ theme }) => theme.colors.backgroundAlt};
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    transition: transform 0.3s ease;

    &:hover {
        transform: translateY(-5px);
    }
`;

const FeatureTitle = styled.h3`
    font-size: 1.2rem;
    margin: 1rem 0 0.5rem;
    color: ${({ theme }) => theme.colors.textPrimary};
`;

const FeatureDescription = styled.p`
    font-size: 0.95rem;
    color: ${({ theme }) => theme.colors.textSecondary};
    line-height: 1.5;
`;

const FeatureIcon = styled.div`
    font-size: 2rem;
    margin-bottom: 1rem;
    color: ${({ theme }) => theme.colors.primary};
`;

const HomePage = () => {
    const { imageData, setAnalysisResult, setLoading } = useContext(ImageContext);
    const navigate = useNavigate();

    const handleAnalyzeWithOurAI = async () => {
        if (!imageData) return;

        setLoading(true);
        try {
            const result = await analyzeWithOurAI(imageData);
            setAnalysisResult(result);
            navigate('/results');
        } catch (error) {
            console.error('Analysis failed:', error);
            // Provide more specific feedback if possible
            const errorMessage = error.message || 'An unknown error occurred during analysis.';
            alert(`Analysis failed: ${errorMessage}`);
        } finally {
            setLoading(false);
        }
    };

    return (
        <HomeContainer>
            <HeroSection>
                <Title>AI Image Analysis</Title>
                <Description>
                    Upload an image for advanced AI analysis. Our system provides detailed insights
                    by identifying patterns and content with state-of-the-art technology.
                </Description>
            </HeroSection>

            <UploadSection>
                <ImageUploader />
                <ActionButton
                    onClick={handleAnalyzeWithOurAI}
                    disabled={!imageData}
                >
                    Analyze Image
                </ActionButton>
            </UploadSection>

            <FeatureGrid>
                <FeatureCard>
                    <FeatureIcon>🔍</FeatureIcon>
                    <FeatureTitle>Deep Analysis</FeatureTitle>
                    <FeatureDescription>
                        Our AI engine performs detailed analysis of image content and structure
                    </FeatureDescription>
                </FeatureCard>

                <FeatureCard>
                    <FeatureIcon>🌐</FeatureIcon>
                    <FeatureTitle>Multiple AI Sources</FeatureTitle>
                    <FeatureDescription>
                        Compare results from both our specialized AI and external AI engines
                    </FeatureDescription>
                </FeatureCard>

                <FeatureCard>
                    <FeatureIcon>📊</FeatureIcon>
                    <FeatureTitle>Comprehensive Reports</FeatureTitle>
                    <FeatureDescription>
                        Generate detailed reports with combined analysis from multiple sources
                    </FeatureDescription>
                </FeatureCard>
            </FeatureGrid>
        </HomeContainer>
    );
};

export default HomePage;