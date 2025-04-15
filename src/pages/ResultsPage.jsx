// src/pages/ResultsPage.jsx

import React, { useContext, useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import styled, { css, keyframes } from 'styled-components';
import { ImageContext } from '../App'; // Adjust path if needed
import ImagePreview from '../components/ImagePreview'; // Adjust path if needed
import LoadingSpinner from '../components/LoadingSpinner'; // Adjust path if needed
import { analyzeWithExternalAI, generateReport } from '../services/api'; // Adjust path if needed

// --- Animations ---
const fadeIn = keyframes`
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
`;

const slideInRight = keyframes`
    from { opacity: 0; transform: translateX(20px); }
    to { opacity: 1; transform: translateX(0); }
`;

const pulseEffect = keyframes`
    0% { box-shadow: 0 0 0 0 rgba(123, 104, 238, 0.4); }
    70% { box-shadow: 0 0 0 10px rgba(123, 104, 238, 0); }
    100% { box-shadow: 0 0 0 0 rgba(123, 104, 238, 0); }
`;

// --- Styled Components ---
const ResultsContainer = styled.div`
    width: 100%;
    max-width: 1200px;
    margin: 0 auto;
    padding: 1.5rem;
    animation: ${fadeIn} 0.5s ease-out;
`;

const ResultsGrid = styled.div`
    display: grid;
    grid-template-columns: 1fr 1.5fr; // Image | Analysis content
    gap: 2.5rem;

    @media (max-width: 900px) {
        grid-template-columns: 1fr; // Stack on smaller screens
    }
`;

const ImageSection = styled.div`
    position: sticky; // Keep image visible on scroll
    top: 2rem;       // Adjust top spacing as needed
    height: fit-content; // Only take height needed for image preview
    background: ${({ theme }) => theme.colors.backgroundAlt};
    border-radius: 16px;
    overflow: hidden;
    box-shadow: ${({ theme }) => theme.shadows.md};
    transition: transform 0.3s ease, box-shadow 0.3s ease;

    &:hover {
        transform: translateY(-5px);
        box-shadow: ${({ theme }) => theme.shadows.lg};
    }

    animation: ${fadeIn} 0.5s ease-out;

    @media (max-width: 900px) {
        position: relative; // Unstick on small screens
        top: auto;
    }
`;

const AnalysisSection = styled.div`
    display: flex;
    flex-direction: column;
    background: ${({ theme }) => theme.colors.cardBackground};
    border-radius: 16px;
    padding: 2rem 2.5rem;
    box-shadow: ${({ theme }) => theme.shadows.md};
    min-height: 450px; // Ensure it has some height
    position: relative; // For potential absolute positioning inside (like spinner)
    transition: transform 0.3s ease, box-shadow 0.3s ease;

    &:hover {
        box-shadow: ${({ theme }) => theme.shadows.lg};
    }

    animation: ${slideInRight} 0.5s ease-out;
`;

const AnalysisContent = styled.div`
    flex: 1; // Takes remaining vertical space
    overflow-y: auto; // Allow content scrolling
    margin-top: 1.25rem; // Space below tabs/source
    padding-right: 0.5rem; // Space for scrollbar
    min-height: 200px; // Ensure content area has some height

    &::-webkit-scrollbar {
        width: 6px;
    }

    &::-webkit-scrollbar-track {
        background: ${({ theme }) => theme.colors.backgroundAlt};
        border-radius: 10px;
    }

    &::-webkit-scrollbar-thumb {
        background: ${({ theme }) => theme.colors.primary}40;
        border-radius: 10px;

        &:hover {
            background: ${({ theme }) => theme.colors.primary}70;
        }
    }
`;

const AnalysisTabsContainer = styled.div`
    display: flex;
    justify-content: flex-end; // Tabs on the right
    margin-bottom: 1.5rem;
    border-bottom: 1px solid ${({ theme }) => theme.colors.border};
    padding-bottom: 0.5rem;
`;

const AnalysisTab = styled.button`
    background: none;
    border: none;
    padding: 0.6rem 1.2rem;
    font-size: 0.95rem;
    font-weight: 500;
    color: ${({ active, theme }) => active ? theme.colors.primary : theme.colors.textSecondary};
    position: relative;
    cursor: pointer;
    transition: all 0.3s ease;

    &:after {
        content: '';
        position: absolute;
        bottom: -0.5rem; // Position underline below border
        left: 0;
        width: 100%;
        height: 3px;
        background: ${({ active, theme }) => active ? theme.colors.primary : 'transparent'};
        border-radius: 3px;
        transition: all 0.3s ease;
        transform: scaleX(${({ active }) => active ? 1 : 0});
        transform-origin: center;
    }

    &:hover {
        color: ${({ theme }) => theme.colors.primary};

        &:after {
            transform: scaleX(1);
            background: ${({ theme }) => theme.colors.primary}80;
        }
    }

    &:disabled {
        opacity: 0.5;
        cursor: not-allowed;
    }
`;

const AnalysisHeader = styled.div`
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1.25rem;
`;

const AnalysisTitle = styled.h2`
    font-size: 1.85rem;
    margin: 0;
    background: linear-gradient(90deg, ${({ theme }) => theme.colors.primary} 0%, ${({ theme }) => theme.colors.primaryHover} 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    position: relative;
    display: inline-block;

    &:after {
        content: '';
        position: absolute;
        bottom: -5px;
        left: 0;
        width: 50px;
        height: 3px;
        background: linear-gradient(90deg, ${({ theme }) => theme.colors.primary} 0%, ${({ theme }) => theme.colors.primaryHover} 100%);
        border-radius: 3px;
    }
`;

const AnalysisSource = styled.div`
    font-size: 0.9rem;
    color: ${({ theme }) => theme.colors.textSecondary};
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-top: 0.75rem; // Add some space above content
    padding: 0.4rem 0.8rem;
    background: ${({ theme }) => theme.colors.backgroundAlt}80;
    border-radius: 8px;
    width: fit-content;

    span {
        font-weight: 500;
        color: ${({ theme }) => theme.colors.textPrimary}; // Make source name stand out slightly
        transition: color 0.2s ease;
    }

    &:hover span {
        color: ${({ theme }) => theme.colors.primary};
    }
`;

const BackButton = styled.button`
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: none;
    border: none;
    color: ${({ theme }) => theme.colors.textSecondary};
    font-size: 0.95rem;
    font-weight: 500;
    padding: 0.6rem 1rem;
    margin-bottom: 1.5rem;
    cursor: pointer;
    transition: all 0.3s ease;
    border-radius: 8px;

    &:hover {
        color: ${({ theme }) => theme.colors.primary};
        background: ${({ theme }) => theme.colors.backgroundAlt};
        transform: translateX(-3px);
    }

    &:before {
        content: '←';
        font-size: 1.1rem;
        transition: transform 0.2s ease;
    }

    &:hover:before {
        transform: translateX(-3px);
    }
`;

const ErrorMessage = styled.div`
    background: rgba(255, 92, 92, 0.1);
    border-left: 3px solid ${({ theme }) => theme.colors.error};
    padding: 1.25rem;
    border-radius: 8px;
    color: ${({ theme }) => theme.colors.textPrimary}; // Ensure text is readable
    margin: 1rem 0; // Add space around error
    font-size: 0.95rem;
    white-space: pre-wrap; // Allow newlines in errors
    box-shadow: 0 3px 8px rgba(255, 92, 92, 0.1);
    animation: ${fadeIn} 0.3s ease-out;

    &:hover {
        background: rgba(255, 92, 92, 0.15);
    }
`;

// Report specific styled components
const ReportWrapper = styled.div`
    display: flex;
    flex-direction: column;
    gap: 1.8rem;
    animation: ${fadeIn} 0.4s ease-out;
`;

const ReportSectionCard = styled.div`
    background: ${({ type, theme }) =>
    type === 'summary' ? 'rgba(123, 104, 238, 0.08)' :
        type === 'disclaimer' ? 'rgba(255, 172, 51, 0.08)' :
            theme.colors.backgroundAlt};
    border-radius: 12px;
    padding: 1.5rem;
    border-left: 4px solid ${({ type, theme }) =>
    type === 'summary' ? theme.colors.primary :
        type === 'disclaimer' ? theme.colors.warning :
            theme.colors.border};
    transition: all 0.3s ease;
    box-shadow: 0 3px 10px rgba(0, 0, 0, 0.03);
    
    &:hover {
        transform: translateY(-3px);
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.08);
        background: ${({ type, theme }) =>
    type === 'summary' ? 'rgba(123, 104, 238, 0.12)' :
        type === 'disclaimer' ? 'rgba(255, 172, 51, 0.12)' :
            theme.colors.backgroundAlt};
    }
    
    ${({ type }) => type === 'summary' && css`
        &:hover {
            animation: ${pulseEffect} 2s infinite;
        }
    `}
`;

const ReportSectionTitle = styled.h3`
    font-size: 1.15rem;
    margin: 0 0 0.9rem 0;
    color: ${({ theme, type }) =>
    type === 'summary' ? theme.colors.primary :
        type === 'disclaimer' ? theme.colors.warning :
            theme.colors.textPrimary};
    display: flex;
    align-items: center;
    
    &:before {
        content: ${({ type }) =>
    type === 'summary' ? '"📊"' :
        type === 'disclaimer' ? '"⚠️"' :
            '""'};
        margin-right: 0.5rem;
        font-size: 1.2rem;
    }
`;

const ReportText = styled.p`
    margin: 0;
    line-height: 1.7;
    color: ${({ type, theme }) => // Allow type prop for potential color changes
    type === 'disclaimer' ? theme.colors.warning :
        theme.colors.textPrimary}; // Default to primary text
    font-size: ${({ type }) => type === 'summary' ? '1.05rem' : '0.95rem'};
    font-weight: ${({ type }) => type === 'summary' ? '500' : '400'};
    white-space: pre-wrap; // Ensure newlines are respected here too
    transition: color 0.2s ease;
`;

// --- Main Component ---
const ResultsPage = () => {
    const navigate = useNavigate();
    const {
        imageData,
        analysisResult, // From context (Our AI result)
        setAnalysisResult,
        loading,
        setLoading,
        setImageData
    } = useContext(ImageContext);

    // Local state for results fetched/generated on this page
    const [externalAnalysisResult, setExternalAnalysisResult] = useState(null);
    const [reportData, setReportData] = useState(null);
    const [activeView, setActiveView] = useState('ourAI'); // Default view
    const [error, setError] = useState(''); // Error specific to actions on this page

    // Effect to handle initial state and redirection if needed
    useEffect(() => {
        if (!imageData) {
            console.log("No image data found on ResultsPage load, redirecting.");
            navigate('/', { replace: true });
            return;
        }
        // Set initial active view - default to 'ourAI'
        setActiveView('ourAI');
        // Clear state associated with other views when image potentially changes
        setExternalAnalysisResult(null);
        setReportData(null);
        setError('');

    }, [imageData, navigate]); // Depend only on imageData for resetting


    const handleTabClick = (view) => {
        setError(''); // Clear errors when switching tabs
        setActiveView(view);

        // Trigger data fetching if needed and not already loaded/loading
        if (view === 'externalAI' && !externalAnalysisResult && !loading) {
            handleExternalAIAnalysis();
        } else if (view === 'report' && !reportData && !loading) {
            handleGenerateReport();
        }
    };


    const handleExternalAIAnalysis = async () => {
        // Prevent re-fetch if already successful or currently loading
        if ((externalAnalysisResult?.success && !error) || loading) {
            // If successful, ensure view is set (might be called directly)
            if (!loading) setActiveView('externalAI');
            return;
        }
        if (!imageData) return;

        setLoading(true);
        setError('');
        setExternalAnalysisResult(null); // Clear previous result
        // setActiveView('externalAI'); // Set view immediately when action starts

        try {
            console.log("Initiating external AI analysis...");
            const result = await analyzeWithExternalAI(imageData);
            console.log("Received external AI analysis result:", result);
            setExternalAnalysisResult(result); // Store result regardless of success/failure
            if (!result.success) {
                // Throw error based on the message from the failed API result
                throw new Error(result.message || 'External analysis API returned failure.');
            }
            // No need to setActiveView here again if handleTabClick was used
        } catch (err) {
            console.error('External AI analysis failed:', err);
            setError(`External analysis failed: ${err.message}`);
            // Keep the result object in state even on error to show API message
        } finally {
            setLoading(false);
        }
    };

    const handleGenerateReport = async () => {
        // Prevent re-fetch if already successful or currently loading
        if ((reportData?.success && !error) || loading) {
            if (!loading) setActiveView('report');
            return;
        }
        if (!imageData) {
            setError("Image data missing."); // Should not happen if buttons disabled
            return;
        }
        // Optional: Check if primary analysis is needed for report
        // if (!analysisResult) { setError("Initial analysis needed for report."); return; }

        setLoading(true);
        setError('');
        setReportData(null); // Clear previous report
        // setActiveView('report'); // Set view immediately

        try {
            console.log("Initiating report generation...");
            const result = await generateReport(imageData, analysisResult, externalAnalysisResult);
            console.log("Received report generation result:", result);
            setReportData(result); // Store result regardless of success/failure
            if (!result.success) {
                throw new Error(result.message || 'Report generation API returned failure.');
            }
        } catch (err) {
            console.error('Report generation failed:', err);
            setError(`Report generation failed: ${err.message}`);
        } finally {
            setLoading(false);
        }
    };

    const handleGoBack = () => {
        navigate('/');
    };

    // --- Logging before rendering content ---
    console.log(`Rendering ResultsPage. Active view: ${activeView}`);
    if (activeView === 'externalAI') {
        console.log("State externalAnalysisResult for rendering:", externalAnalysisResult);
        console.log("Description property for rendering:", externalAnalysisResult?.description);
    }
    if (activeView === 'report') {
        console.log("State reportData for rendering:", reportData);
    }
    // --- End Logging ---


    // Helper function to render the main content based on the active view
    const renderAnalysisContent = () => {
        // Handle loading state first
        if (loading) {
            let loadingText = 'Processing...';
            if (activeView === 'externalAI') loadingText = 'Querying External AI...';
            else if (activeView === 'report') loadingText = 'Generating Report...';
            // Center spinner within the content area
            return <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}><LoadingSpinner text={loadingText} /></div>;
        }

        // Handle errors specific to the current view's action
        if (error && (activeView === 'externalAI' || activeView === 'report')) {
            return <ErrorMessage>{error}</ErrorMessage>;
        }


        switch (activeView) {
            case 'ourAI':
                // Display message if the backend simulation indicates it's unavailable
                if (analysisResult?.status === 'backend_unavailable') {
                    return <ErrorMessage>{analysisResult.message || 'Our AI backend service is unavailable.'}</ErrorMessage>;
                }
                // Display message if the API call failed generally but provided a message
                if (!analysisResult?.success && analysisResult?.message) {
                    return <ErrorMessage>{`Our AI Status: ${analysisResult.message}`}</ErrorMessage>;
                }
                // Display the description if available and successful
                return analysisResult?.description
                    ? <ReportText>{analysisResult.description}</ReportText>
                    : <ReportText>No analysis details available from Our AI.</ReportText>; // Fallback

            case 'externalAI':
                // Handle case where the fetch hasn't happened or cleared
                if (!externalAnalysisResult) {
                    // Don't show error if just hasn't been fetched yet
                    return <ReportText>Click the 'External AI' tab to initiate analysis.</ReportText>;
                }
                // Handle explicit failure message from API result object
                if (!externalAnalysisResult.success && externalAnalysisResult.message) {
                    return <ErrorMessage>{`External AI Status: ${externalAnalysisResult.message}`}</ErrorMessage>;
                }
                // Display the description in a modern card layout if successful
                if (externalAnalysisResult.success) {
                    return (
                        <ReportWrapper>
                            {/* Main Analysis Card */}
                            <ReportSectionCard type="summary">
                                <ReportSectionTitle type="summary">Analysis Details</ReportSectionTitle>
                                <ReportText type="summary">
                                    {externalAnalysisResult.description || 'Analysis successful, but no description provided.'}
                                </ReportText>
                            </ReportSectionCard>

                            {/* Add Medical Disclaimer Card */}
                            <ReportSectionCard type="disclaimer">
                                <ReportSectionTitle type="disclaimer">Important Note</ReportSectionTitle>
                                <ReportText type="disclaimer">
                                    This AI analysis is informational only and does not constitute medical advice.
                                    Please consult a qualified healthcare professional to discuss these potential findings.
                                </ReportText>
                            </ReportSectionCard>
                        </ReportWrapper>
                    );
                }
                // Fallback if state is unexpected (should be covered by above cases)
                return <ReportText>External analysis status unknown.</ReportText>;

            case 'report':
                // Handle case where fetch hasn't happened
                if (!reportData) {
                    return <ReportText>Click the 'Report' tab to generate the report.</ReportText>;
                }
                // Handle explicit failure message from API result object
                if (!reportData.success && reportData.message) {
                    return <ErrorMessage>{`Report Generation Status: ${reportData.message}`}</ErrorMessage>;
                }
                // Render the report sections if successful
                if (reportData.success && reportData.sections) {
                    return (
                        <ReportWrapper>
                            {reportData.sections.map((section, index) => (
                                <ReportSectionCard key={index} type={section.type}>
                                    <ReportSectionTitle type={section.type}>{section.title}</ReportSectionTitle>
                                    {/* Use ReportText for consistent styling */}
                                    <ReportText type={section.type}>{section.content}</ReportText>
                                </ReportSectionCard>
                            ))}
                        </ReportWrapper>
                    );
                }
                // Fallback if report structure is unexpected
                return <ReportText>Report data is unavailable or in an unexpected format.</ReportText>;


            default:
                return <ReportText>Select an analysis type.</ReportText>;
        }
    };

    // Helper to get the source text for the current view
    const getAnalysisSource = () => {
        switch (activeView) {
            case 'ourAI':
                return analysisResult?.source || 'Our AI Engine';
            case 'externalAI':
                // Show source even if failed, as long as the result object exists
                return externalAnalysisResult?.source || (externalAnalysisResult ? 'External AI Provider' : 'N/A');
            case 'report':
                return reportData?.source || (reportData ? 'Combined Analysis' : 'N/A');
            default:
                return 'Unknown Source';
        }
    };

    // Render the component
    return (
        <ResultsContainer>
            <BackButton onClick={handleGoBack}>Back to Upload</BackButton>
            <ResultsGrid>
                <ImageSection>
                    {/* Render image preview if imageData exists */}
                    {imageData ? <ImagePreview imageData={imageData} /> : <p>No image selected.</p>}
                </ImageSection>

                <AnalysisSection>
                    {/* Header */}
                    <AnalysisHeader>
                        <AnalysisTitle>
                            {activeView === 'ourAI' ? 'Our AI Analysis' :
                                activeView === 'externalAI' ? 'External AI Opinion' :
                                    'Generated Report'}
                        </AnalysisTitle>
                    </AnalysisHeader>

                    {/* Tabs */}
                    <AnalysisTabsContainer>
                        <AnalysisTab
                            onClick={() => handleTabClick('ourAI')}
                            active={activeView === 'ourAI'}
                            disabled={loading}
                        >
                            Our AI
                        </AnalysisTab>
                        <AnalysisTab
                            onClick={() => handleTabClick('externalAI')}
                            active={activeView === 'externalAI'}
                            disabled={loading || !imageData} // Disable if no image
                        >
                            External AI
                        </AnalysisTab>
                        <AnalysisTab
                            onClick={() => handleTabClick('report')}
                            active={activeView === 'report'}
                            // Disable report tab if loading, no image, or maybe if primary analysis failed (optional)
                            disabled={loading || !imageData /* || !analysisResult?.success */}
                        >
                            Report
                        </AnalysisTab>
                    </AnalysisTabsContainer>

                    {/* Source Line - Conditionally render confidence for Our AI */}
                    <AnalysisSource>
                        Source: <span>{getAnalysisSource()}</span>
                        {activeView === 'ourAI' && analysisResult?.success &&
                            analysisResult?.confidence && !loading &&
                            ` (Confidence: ${(analysisResult.confidence * 100).toFixed(0)}%)`}
                    </AnalysisSource>

                    {/* Main Content Area (Spinner or Rendered Content) */}
                    <AnalysisContent>
                        {renderAnalysisContent()}
                    </AnalysisContent>

                </AnalysisSection>
            </ResultsGrid>
        </ResultsContainer>
    );
};

export default ResultsPage;