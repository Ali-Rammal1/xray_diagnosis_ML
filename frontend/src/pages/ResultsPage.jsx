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

// --- NEW: Keyframes for colored shadow pulse/glow ---
// Define separate keyframes for different glow colors based on theme
const createRippleGlowAnimation = (color) => keyframes`
    0%, 100% {
        /* Base shadow + minimal glow */
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1), 0 0 5px 0px ${color}66; /* 40% opacity glow */
    }
    50% {
        /* Base shadow + expanded glow */
        box-shadow: 0 6px 15px rgba(0, 0, 0, 0.15), 0 0 15px 5px ${color}B3; /* 70% opacity glow */
    }
`;

// Generate keyframes for each color we might use
const rippleGlowSuccess = createRippleGlowAnimation(props => props.theme.colors.success);
const rippleGlowWarning = createRippleGlowAnimation(props => props.theme.colors.warning);
const rippleGlowError = createRippleGlowAnimation(props => props.theme.colors.error);
const rippleGlowPrimary = createRippleGlowAnimation(props => props.theme.colors.primary);
const rippleGlowDefault = createRippleGlowAnimation(props => props.theme.colors.textTertiary); // Default glow color

// Helper function to select the correct animation based on props
const getRippleAnimation = (theme, type, diagnosis, confidenceLevel) => {
    const isNormal = diagnosis === 'normal';
    const isHighConfidence = confidenceLevel === 'high';
    const isMediumConfidence = confidenceLevel === 'medium';

    if (type === 'summary') return rippleGlowPrimary;
    if (type === 'disclaimer') return rippleGlowWarning;

    // For DiagnosisCard or 'primary' type ReportSectionCard
    if (type === 'primary' || !type) { // Apply diagnosis logic if type is 'primary' or not set (DiagnosisCard case)
        if (isNormal && isHighConfidence) return rippleGlowSuccess;
        if (isNormal && isMediumConfidence) return rippleGlowWarning; // Cautious warning glow
        if (!isNormal && isHighConfidence) return rippleGlowError;
        if (!isNormal && isMediumConfidence) return rippleGlowWarning;
    }

    // Default for external or low confidence/unknown
    return rippleGlowDefault;
};


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
        transform: translateY(-5px); // Keep subtle lift on image hover
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
    position: relative;
    transition: transform 0.3s ease, box-shadow 0.3s ease;

    &:hover {
        box-shadow: ${({ theme }) => theme.shadows.lg};
    }

    animation: ${slideInRight} 0.5s ease-out;
`;

const AnalysisContent = styled.div`
    flex: 1;
    overflow-y: auto;
    margin-top: 1.25rem;
    padding-right: 0.5rem;
    min-height: 200px;

    &::-webkit-scrollbar { width: 6px; }
    &::-webkit-scrollbar-track { background: ${({ theme }) => theme.colors.backgroundAlt}; border-radius: 10px; }
    &::-webkit-scrollbar-thumb { background: ${({ theme }) => theme.colors.primary}40; border-radius: 10px; }
    &::-webkit-scrollbar-thumb:hover { background: ${({ theme }) => theme.colors.primary}70; }
`;

const AnalysisTabsContainer = styled.div`
    display: flex;
    justify-content: flex-end;
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
        bottom: -0.5rem;
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
    &:disabled { opacity: 0.5; cursor: not-allowed; }
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
    background-clip: text;
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
    margin-top: 0.75rem;
    padding: 0.4rem 0.8rem;
    background: ${({ theme }) => theme.colors.backgroundAlt}80;
    border-radius: 8px;
    width: fit-content;

    span {
        font-weight: 500;
        color: ${({ theme }) => theme.colors.textPrimary};
        transition: color 0.2s ease;
    }
    &:hover span { color: ${({ theme }) => theme.colors.primary}; }
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
        transform: translateX(-3px); // Keep slide on back button
    }
    &:before { content: '←'; font-size: 1.1rem; transition: transform 0.2s ease; }
    &:hover:before { transform: translateX(-3px); }
`;

const ErrorMessage = styled.div`
    background: rgba(255, 92, 92, 0.1);
    border-left: 3px solid ${({ theme }) => theme.colors.error};
    padding: 1.25rem;
    border-radius: 8px;
    color: ${({ theme }) => theme.colors.textPrimary};
    margin: 1rem 0;
    font-size: 0.95rem;
    white-space: pre-wrap;
    box-shadow: 0 3px 8px rgba(255, 92, 92, 0.1);
    animation: ${fadeIn} 0.3s ease-out;
    &:hover { background: rgba(255, 92, 92, 0.15); }
`;

const ConfidenceTag = styled.span`
    display: inline-block;
    padding: 0.25rem 0.75rem;
    font-size: 0.85rem;
    font-weight: 600;
    border-radius: 12px;
    margin-left: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: white;
    background-color: ${({ level, theme }) =>
            level === 'high' ? theme.colors.success :
                    level === 'medium' ? theme.colors.warning :
                            theme.colors.textTertiary
    };
    box-shadow: 0 2px 4px ${({ level, theme }) =>
            level === 'high' ? `${theme.colors.success}4D` :
                    level === 'medium' ? `${theme.colors.warning}4D` :
                            `${theme.colors.textTertiary}4D`
    };
`;

// --- MODIFIED DiagnosisCard ---
const DiagnosisCard = styled.div`
    // ** RESTORED Conditional background **
    background: ${({ theme, diagnosis, confidenceLevel }) => {
        const isNormal = diagnosis === 'normal';
        const isHighConfidence = confidenceLevel === 'high';
        const isMediumConfidence = confidenceLevel === 'medium';
        if (isNormal && isHighConfidence) return `${theme.colors.success}1A`;
        if (isNormal && isMediumConfidence) return `${theme.colors.success}0D`;
        if (!isNormal && (isHighConfidence || isMediumConfidence)) return `${theme.colors.warning}1A`;
        return `${theme.colors.backgroundAlt}`; // Use alt as base for ripple clarity
    }};
    border-left: 4px solid ${({ theme, diagnosis, confidenceLevel }) => {
        const isNormal = diagnosis === 'normal';
        const isHighConfidence = confidenceLevel === 'high';
        const isMediumConfidence = confidenceLevel === 'medium';
        if (isNormal && isHighConfidence) return theme.colors.success;
        if (isNormal && isMediumConfidence) return theme.colors.warning;
        if (!isNormal && isHighConfidence) return theme.colors.error;
        if (!isNormal && isMediumConfidence) return theme.colors.warning;
        return theme.colors.textTertiary;
    }};
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    // Base shadow
    box-shadow: 0 3px 8px rgba(0, 0, 0, 0.08);
    transition: background-color 0.3s ease, box-shadow 0.3s ease; // Transition background and shadow
    position: relative; // Needed for children positioning if any

    &:hover {
        // ** REMOVED Slide **
        /* transform: translateX(-5px); */
        // ** ADDED Background Brighten **
        background: ${({ theme, diagnosis, confidenceLevel }) => { // Slightly enhance background on hover
            const isNormal = diagnosis === 'normal';
            const isHighConfidence = confidenceLevel === 'high';
            const isMediumConfidence = confidenceLevel === 'medium';
            if (isNormal && isHighConfidence) return `${theme.colors.success}26`; // ~15% green
            if (isNormal && isMediumConfidence) return `${theme.colors.success}1A`; // ~10% green
            if (!isNormal && (isHighConfidence || isMediumConfidence)) return `${theme.colors.warning}26`; // ~15% orange
            return `${theme.colors.backgroundHover}`; // Use theme hover for default/low
        }};
        // ** ADDED Shadow Ripple Animation on Hover **
        animation: ${props => getRippleAnimation(props.theme, 'primary', props.diagnosis, props.confidenceLevel)} 3s ease-in-out infinite;
    }
`;

// --- MODIFIED ReportSectionCard ---
const ReportSectionCard = styled.div`
    // ** RESTORED Conditional background **
    background: ${({ theme, type, diagnosis, confidenceLevel }) => {
        if (type === 'summary') return `${theme.colors.primary}14`;
        if (type === 'disclaimer') return `${theme.colors.warning}14`;
        const isNormal = diagnosis === 'normal';
        const isHighConfidence = confidenceLevel === 'high';
        const isMediumConfidence = confidenceLevel === 'medium';
        if (type === 'primary') {
            if (isNormal && isHighConfidence) return `${theme.colors.success}1A`;
            if (isNormal && isMediumConfidence) return `${theme.colors.success}0D`;
            if (!isNormal && (isHighConfidence || isMediumConfidence)) return `${theme.colors.warning}1A`;
        }
        return theme.colors.backgroundAlt;
    }};
    border-radius: 12px;
    padding: 1.5rem;
    border-left: 4px solid ${({ theme, type, diagnosis, confidenceLevel }) => {
        if (type === 'summary') return theme.colors.primary;
        if (type === 'disclaimer') return theme.colors.warning;
        if (type === 'primary') {
            const isNormal = diagnosis === 'normal';
            const isHighConfidence = confidenceLevel === 'high';
            const isMediumConfidence = confidenceLevel === 'medium';
            if (isNormal && isHighConfidence) return theme.colors.success;
            if (isNormal && isMediumConfidence) return theme.colors.warning;
            if (!isNormal && isHighConfidence) return theme.colors.error;
            if (!isNormal && isMediumConfidence) return theme.colors.warning;
        }
        return theme.colors.border;
    }};
    // Base shadow
    box-shadow: 0 3px 10px rgba(0, 0, 0, 0.03);
    transition: background-color 0.3s ease, box-shadow 0.3s ease; // Transition background and shadow
    position: relative;

    &:hover {
        // ** REMOVED Slide **
        /* transform: translateX(-5px); */
        // ** ADDED Background Brighten **
        background: ${({ theme, type, diagnosis, confidenceLevel }) => {
            if (type === 'summary') return `${theme.colors.primary}26`;
            if (type === 'disclaimer') return `${theme.colors.warning}26`;
            if (type === 'primary') {
                const isNormal = diagnosis === 'normal';
                const isHighConfidence = confidenceLevel === 'high';
                const isMediumConfidence = confidenceLevel === 'medium';
                if (isNormal && isHighConfidence) return `${theme.colors.success}26`;
                if (isNormal && isMediumConfidence) return `${theme.colors.success}1A`;
                if (!isNormal && (isHighConfidence || isMediumConfidence)) return `${theme.colors.warning}26`;
            }
            return theme.colors.backgroundHover;
        }};
        // ** ADDED Shadow Ripple Animation on Hover **
        animation: ${props => getRippleAnimation(props.theme, props.type, props.diagnosis, props.confidenceLevel)} 3s ease-in-out infinite;
    }
`;

// Ensure content inside cards is not affected by shadow or hover changes if needed
const DiagnosisHeader = styled.div`
    display: flex; align-items: center; margin-bottom: 1rem; position: relative; z-index: 1;
`;
const DiagnosisTitle = styled.h3`
    margin: 0; font-size: 1.15rem; position: relative; z-index: 1;
    color: ${({ theme, diagnosis }) =>
            diagnosis === 'normal' ? theme.colors.success :
                    (diagnosis === 'pneumonia' || diagnosis === 'tuberculosis') ? theme.colors.warning :
                            theme.colors.textPrimary};
`;
const DiagnosisContent = styled.p`
    margin: 0; line-height: 1.7; font-size: 1rem; position: relative; z-index: 1;
    color: ${({ theme }) => theme.colors.textPrimary};
`;
const MedicalWarning = styled.div`
    background: ${({ theme }) => `${theme.colors.warning}1A`};
    border-left: 3px solid ${({ theme }) => theme.colors.warning};
    padding: 1rem 1.25rem; border-radius: 8px; margin-top: 1.25rem;
    font-size: 0.9rem; color: ${({ theme }) => theme.colors.textSecondary};
    position: relative; z-index: 1;
    &:before { content: '⚠️'; margin-right: 0.5rem; font-size: 1.1rem; }
`;
const ReportWrapper = styled.div`
    display: flex; flex-direction: column; gap: 1.8rem; animation: ${fadeIn} 0.4s ease-out;
`;
const ReportSectionTitle = styled.h3`
    font-size: 1.15rem; margin: 0 0 0.9rem 0; position: relative; z-index: 1;
    color: ${({ theme, type, diagnosis }) => { /* ... same logic as before ... */
        if (type === 'summary') return theme.colors.primary;
        if (type === 'disclaimer') return theme.colors.warning;
        if (type === 'primary') {
            if (diagnosis === 'normal') return theme.colors.success;
            if (diagnosis === 'pneumonia' || diagnosis === 'tuberculosis') return theme.colors.warning;
        }
        return theme.colors.textPrimary;
    }};
    display: flex; align-items: center;
    &:before { /* ... same logic as before ... */
        content: ${({ type }) => type === 'summary' ? '"📊"' : type === 'disclaimer' ? '"⚠️"' : type === 'external' ? '"🔬"' : type === 'primary' ? '"💻"' : '""'};
        margin-right: 0.6rem; font-size: 1.2rem;
    }
`;
const ReportText = styled.p`
    margin: 0; line-height: 1.7; position: relative; z-index: 1;
    color: ${({ theme }) => theme.colors.textPrimary};
    font-size: ${({ type }) => type === 'summary' ? '1.05rem' : '1rem'};
    font-weight: ${({ type }) => type === 'summary' ? '500' : '400'};
    white-space: pre-wrap; transition: color 0.2s ease;
`;

// --- Main Component Logic (Remains Unchanged from previous step) ---
const ResultsPage = () => {
    const navigate = useNavigate();
    const { imageData, analysisResult, setLoading, loading } = useContext(ImageContext);
    const [externalAnalysisResult, setExternalAnalysisResult] = useState(null);
    const [reportData, setReportData] = useState(null);
    const [activeView, setActiveView] = useState('ourAI');
    const [error, setError] = useState('');

    useEffect(() => {
        if (!imageData) {
            console.log("No image data found on ResultsPage load, redirecting.");
            navigate('/', { replace: true });
            return;
        }
        setActiveView(analysisResult ? 'ourAI' : 'externalAI');
        setExternalAnalysisResult(null);
        setReportData(null);
        setError('');
    }, [imageData, navigate, analysisResult]);

    const handleTabClick = async (view) => {
        setError('');
        setActiveView(view);
        if (view === 'externalAI' && !externalAnalysisResult && !loading) {
            await handleExternalAIAnalysis();
        } else if (view === 'report' && !reportData && !loading) {
            await handleGenerateReport();
        }
    };

    const handleExternalAIAnalysis = async () => {
        // ... (logic unchanged)
        if ((externalAnalysisResult?.success && !error) || loading) return;
        if (!imageData?.file) { setError("Image data missing or invalid."); return; }
        setLoading(true); setError(''); setExternalAnalysisResult(null);
        try {
            const result = await analyzeWithExternalAI(imageData);
            setExternalAnalysisResult(result);
            if (!result.success) throw new Error(result.message || 'External analysis API returned failure.');
        } catch (err) {
            console.error('External AI analysis failed:', err);
            setError(`External analysis failed: ${err.message}`);
            if (!externalAnalysisResult) setExternalAnalysisResult({ success: false, message: err.message || 'Unknown error during external analysis.', source: 'External AI Provider (Error)' });
        } finally { setLoading(false); }
    };

    const handleGenerateReport = async () => {
        // ... (logic unchanged)
        if ((reportData?.success && !error) || loading) return;
        if (!imageData?.file) { setError("Image data missing."); return; }
        if (!analysisResult && !externalAnalysisResult) { setError("At least one AI analysis is needed."); return; }
        setLoading(true); setError(''); setReportData(null);
        try {
            const result = await generateReport(imageData, analysisResult, externalAnalysisResult);
            setReportData(result);
            if (!result.success) throw new Error(result.message || 'Report generation failed.');
        } catch (err) {
            console.error('Report generation failed:', err);
            setError(`Report generation failed: ${err.message}`);
            if (!reportData) setReportData({ success: false, message: err.message || 'Unknown error during report generation.', source: 'Report Generator (Error)' });
        } finally { setLoading(false); }
    };

    const handleGoBack = () => { navigate('/'); };

    const renderAnalysisContent = () => {
        // ... (logic unchanged, relies on styled components now)
        if (loading) {
            let loadingText = 'Processing...';
            if (activeView === 'externalAI') loadingText = 'Querying External AI...';
            else if (activeView === 'report') loadingText = 'Generating Report...';
            return <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}><LoadingSpinner text={loadingText} /></div>;
        }
        if (error && ((activeView === 'externalAI' && externalAnalysisResult?.success === false) || (activeView === 'report' && reportData?.success === false)) ) {
            const message = activeView === 'externalAI' ? externalAnalysisResult?.message : reportData?.message;
            return <ErrorMessage>{error || message || "An error occurred."}</ErrorMessage>;
        }
        if (error && activeView !== 'ourAI') { return <ErrorMessage>{error}</ErrorMessage>; }

        switch (activeView) {
            case 'ourAI':
                if (analysisResult?.status === 'backend_unavailable') return <ErrorMessage>{analysisResult.message || 'Our AI backend service is unavailable.'}</ErrorMessage>;
                if (!analysisResult?.success && analysisResult?.message) return <ErrorMessage>{`Our AI Status: ${analysisResult.message}`}</ErrorMessage>;
                if (analysisResult?.success) {
                    const diagnosis = analysisResult.prediction?.toLowerCase() || 'unknown';
                    const confidenceLevel = analysisResult.confidenceLevel || 'low';
                    return (<DiagnosisCard diagnosis={diagnosis} confidenceLevel={confidenceLevel}> <DiagnosisHeader> <DiagnosisTitle diagnosis={diagnosis}> {analysisResult.prediction || 'Unknown'} </DiagnosisTitle> {confidenceLevel && <ConfidenceTag level={confidenceLevel}>{confidenceLevel}</ConfidenceTag>} </DiagnosisHeader> <DiagnosisContent> {analysisResult.diagnosisMessage || `Analysis suggests ${diagnosis} condition.`} </DiagnosisContent> {((diagnosis === 'pneumonia' || diagnosis === 'tuberculosis') && (confidenceLevel === 'medium' || confidenceLevel === 'high')) && (<MedicalWarning> This AI analysis is informational only, we recommend consulting a qualified health care professional.  </MedicalWarning> )} </DiagnosisCard>);
                }
                return <ReportText>Our AI analysis is not available...</ReportText>;
            case 'externalAI':
                if (!externalAnalysisResult && !error) return <ReportText>Click the 'External AI' tab...</ReportText>;
                if (externalAnalysisResult?.success) { return (<ReportWrapper> <ReportSectionCard type="external"> <ReportSectionTitle type="external">Analysis Details</ReportSectionTitle> <ReportText> {externalAnalysisResult.description || 'Analysis successful...'} </ReportText> </ReportSectionCard> <ReportSectionCard type="disclaimer"> <ReportSectionTitle type="disclaimer">Important Note</ReportSectionTitle> <ReportText type="disclaimer"> This AI analysis is informational only, consult a qualified healthcare professional if any concerns arise.</ReportText> </ReportSectionCard> </ReportWrapper>); }
                return <ReportText>External analysis status unknown...</ReportText>;
            case 'report':
                if (!reportData && !error) return <ReportText>Click the 'Report' tab...</ReportText>;
                if (reportData?.success && reportData.sections) { return (<ReportWrapper> {reportData.sections.map((section, index) => (<ReportSectionCard key={index} type={section.type} diagnosis={section.diagnosis} confidenceLevel={section.confidenceLevel}> <ReportSectionTitle type={section.type} diagnosis={section.diagnosis}> {section.title} </ReportSectionTitle> {section.type === 'primary' && section.confidenceLevel && (<DiagnosisHeader style={{ marginBottom: '0.5rem', marginTop: '-0.5rem' }}> <ConfidenceTag level={section.confidenceLevel}> {section.confidenceLevel} Confidence </ConfidenceTag> </DiagnosisHeader>)} <ReportText type={section.type}>{section.content}</ReportText> </ReportSectionCard> ))} </ReportWrapper>); }
                return <ReportText>Report data is unavailable...</ReportText>;
            default: return <ReportText>Select an analysis type.</ReportText>;
        }
    };

    const getAnalysisSource = () => { /* ... same logic ... */
        switch (activeView) {
            case 'ourAI': return analysisResult?.source || 'Our AI Engine';
            case 'externalAI': return externalAnalysisResult?.source || 'External AI Provider';
            case 'report': return reportData?.source || 'Combined Analysis';
            default: return 'Unknown Source';
        }
    };

    return (
        <ResultsContainer>
            <BackButton onClick={handleGoBack}>Back to Upload</BackButton>
            <ResultsGrid>
                <ImageSection>
                    {imageData ? <ImagePreview imageData={imageData} /> : <p>No image selected.</p>}
                </ImageSection>
                <AnalysisSection>
                    <AnalysisHeader>
                        <AnalysisTitle>
                            {activeView === 'ourAI' ? 'Our AI Analysis' : activeView === 'externalAI' ? 'External AI Opinion' : 'Generated Report'}
                        </AnalysisTitle>
                    </AnalysisHeader>
                    <AnalysisTabsContainer>
                        <AnalysisTab onClick={() => handleTabClick('ourAI')} active={activeView === 'ourAI'} disabled={loading || !analysisResult}> Our AI </AnalysisTab>
                        <AnalysisTab onClick={() => handleTabClick('externalAI')} active={activeView === 'externalAI'} disabled={loading || !imageData}> External AI </AnalysisTab>
                        <AnalysisTab onClick={() => handleTabClick('report')} active={activeView === 'report'} disabled={loading || !imageData || (!analysisResult && !externalAnalysisResult)}> Report </AnalysisTab>
                    </AnalysisTabsContainer>
                    <AnalysisSource> Source: <span>{getAnalysisSource()}</span> </AnalysisSource>
                    <AnalysisContent> {renderAnalysisContent()} </AnalysisContent>
                </AnalysisSection>
            </ResultsGrid>
        </ResultsContainer>
    );
};

export default ResultsPage;