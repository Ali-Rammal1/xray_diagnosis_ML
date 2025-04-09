// src/pages/ResultsPage.jsx

import React, { useContext, useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import styled, { css } from 'styled-components';
import { ImageContext } from '../App';
import ImagePreview from '../components/ImagePreview';
import AnalysisButton from '../components/AnalysisButton';
import LoadingSpinner from '../components/LoadingSpinner';
import { analyzeWithExternalAI, generateReport } from '../services/api';

// Updated Styled Components
const ResultsContainer = styled.div`
    width: 100%;
    max-width: 1200px;
    margin: 0 auto;
    padding: 1rem;
`;

const ResultsGrid = styled.div`
    display: grid;
    grid-template-columns: 1fr 1.5fr;
    gap: 2rem;

    @media (max-width: 900px) {
        grid-template-columns: 1fr;
    }
`;

const ImageSection = styled.div`
    position: sticky;
    top: 2rem;
    height: fit-content;
    background: ${({ theme }) => theme.colors.backgroundAlt};
    border-radius: 12px;
    overflow: hidden;
    box-shadow: ${({ theme }) => theme.shadows.md};
`;

const AnalysisSection = styled.div`
    display: flex;
    flex-direction: column;
    background: ${({ theme }) => theme.colors.cardBackground};
    border-radius: 12px;
    padding: 1.5rem 2rem;
    box-shadow: ${({ theme }) => theme.shadows.md};
    min-height: 450px;
    position: relative;
`;

const AnalysisContent = styled.div`
  flex: 1;
  overflow-y: auto;
  margin-bottom: 1rem;
  padding-right: 0.5rem;
  min-height: 200px;
`;

// Modern tab-style buttons that appear on the right side
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
    transition: all 0.2s;

    &:after {
        content: '';
        position: absolute;
        bottom: -0.5rem;
        left: 0;
        width: 100%;
        height: 3px;
        background: ${({ active, theme }) => active ? theme.colors.primary : 'transparent'};
        border-radius: 3px;
    }

    &:hover {
        color: ${({ active, theme }) => active ? theme.colors.primary : theme.colors.textPrimary};
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
    margin-bottom: 1rem;
`;

const AnalysisTitle = styled.h2`
    font-size: 1.75rem;
    margin: 0;
    background: linear-gradient(90deg, ${({ theme }) => theme.colors.primary} 0%, ${({ theme }) => theme.colors.primaryHover} 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
`;

const AnalysisSource = styled.div`
    font-size: 0.9rem;
    color: ${({ theme }) => theme.colors.textSecondary};
    display: flex;
    align-items: center;
    gap: 0.5rem;

    span {
        font-weight: 500;
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
    padding: 0.5rem 0;
    margin-bottom: 1.5rem;
    cursor: pointer;
    transition: color 0.2s;

    &:hover {
        color: ${({ theme }) => theme.colors.primary};
    }

    &:before {
        content: '←';
        font-size: 1.1rem;
    }
`;

const ErrorMessage = styled.div`
    background: rgba(255, 92, 92, 0.1);
    border-left: 3px solid ${({ theme }) => theme.colors.error};
    padding: 1rem;
    border-radius: 4px;
    color: ${({ theme }) => theme.colors.textPrimary};
    margin: 1rem 0;
`;

const ReportWrapper = styled.div`
    display: flex;
    flex-direction: column;
    gap: 1.5rem;
`;

const ReportSectionCard = styled.div`
    background: ${({ type, theme }) =>
            type === 'summary' ? 'rgba(123, 104, 238, 0.1)' :
                    type === 'disclaimer' ? 'rgba(255, 172, 51, 0.1)' :
                            theme.colors.backgroundAlt};
    border-radius: 8px;
    padding: 1.25rem;
    border-left: 4px solid ${({ type, theme }) =>
            type === 'summary' ? theme.colors.primary :
                    type === 'disclaimer' ? theme.colors.warning :
                            theme.colors.border};
`;

const ReportSectionTitle = styled.h3`
    font-size: 1.1rem;
    margin: 0 0 0.75rem 0;
    color: ${({ theme }) => theme.colors.textPrimary};
`;

const ReportText = styled.p`
    margin: 0;
    line-height: 1.6;
    color: ${({ type, theme }) =>
            type === 'disclaimer' ? theme.colors.warning : theme.colors.textPrimary};
    font-size: ${({ type }) => type === 'summary' ? '1.05rem' : '0.95rem'};
    font-weight: ${({ type }) => type === 'summary' ? '500' : '400'};
`;

const KeyValueText = styled.div`
  display: flex;
  margin-bottom: 0.75rem;
  
  strong {
    width: 100px;
    flex-shrink: 0;
    color: ${({ theme }) => theme.colors.textSecondary};
    margin-right: 1rem;
  }
`;

// Main Component
const ResultsPage = () => {
    const navigate = useNavigate();
    const { imageData, analysisResult, loading, setLoading, setAnalysisResult, setImageData } = useContext(ImageContext);
    const [externalAnalysisResult, setExternalAnalysisResult] = useState(null);
    const [reportData, setReportData] = useState(null);
    const [activeView, setActiveView] = useState('ourAI');
    const [error, setError] = useState('');

    useEffect(() => {
        if (!imageData) {
            setExternalAnalysisResult(null);
            setReportData(null);
            setActiveView('ourAI');
            setError('');
        } else if (analysisResult && activeView !== 'ourAI' && !externalAnalysisResult && !reportData) {
            setActiveView('ourAI');
        }
    }, [imageData, analysisResult]);

    const handleExternalAIAnalysis = async () => {
        if (externalAnalysisResult && !error) {
            setActiveView('externalAI');
            return;
        }

        if (!imageData) return;

        setLoading(true);
        setError('');
        setExternalAnalysisResult(null);
        setActiveView('externalAI');

        try {
            const result = await analyzeWithExternalAI(imageData);
            if (result.success) {
                setExternalAnalysisResult(result);
            } else {
                throw new Error(result.message || 'External analysis returned failure flag.');
            }
        } catch (err) {
            setError(`External analysis failed: ${err.message}`);
        } finally {
            setLoading(false);
        }
    };

    const handleGenerateReport = async () => {
        if (reportData && !error) {
            setActiveView('report');
            return;
        }

        if (!imageData) {
            setError("Image data missing.");
            return;
        }

        setLoading(true);
        setError('');
        setReportData(null);
        setActiveView('report');

        try {
            const result = await generateReport(imageData, analysisResult, externalAnalysisResult);
            if (result.success) {
                setReportData(result);
            } else {
                throw new Error(result.message || 'Report generation returned failure flag.');
            }
        } catch (err) {
            setError(`Report generation failed: ${err.message}`);
        } finally {
            setLoading(false);
        }
    };

    const handleGoBack = () => {
        navigate('/');
    };

    const renderAnalysisContent = () => {
        switch (activeView) {
            case 'ourAI':
                if (analysisResult?.status === 'backend_unavailable') {
                    return <ErrorMessage>{analysisResult.message || 'Backend unavailable.'}</ErrorMessage>;
                }
                if (!analysisResult?.success && analysisResult?.message) {
                    return <ErrorMessage>{analysisResult.message}</ErrorMessage>;
                }
                return analysisResult?.description
                    ? <ReportText>{analysisResult.description}</ReportText>
                    : <ReportText>Analysis status unknown.</ReportText>;

            case 'externalAI':
                if (!externalAnalysisResult)
                    return <ReportText>External analysis not yet performed or data unavailable.</ReportText>;

                return (
                    <div>
                        {externalAnalysisResult.keyFinding &&
                            <KeyValueText>
                                <strong>Finding:</strong>
                                {externalAnalysisResult.keyFinding}
                            </KeyValueText>
                        }
                        {externalAnalysisResult.supportingEvidence &&
                            <KeyValueText>
                                <strong>Evidence:</strong>
                                {externalAnalysisResult.supportingEvidence}
                            </KeyValueText>
                        }
                    </div>
                );

            case 'report':
                if (!reportData?.sections)
                    return <ReportText>Report not yet generated or data unavailable.</ReportText>;

                return (
                    <ReportWrapper>
                        {reportData.sections.map((section, index) => (
                            <ReportSectionCard key={index} type={section.type}>
                                <ReportSectionTitle>{section.title}</ReportSectionTitle>
                                <ReportText type={section.type}>{section.content}</ReportText>
                            </ReportSectionCard>
                        ))}
                    </ReportWrapper>
                );

            default:
                return <ReportText>Select an analysis type.</ReportText>;
        }
    };

    const getAnalysisSource = () => {
        switch (activeView) {
            case 'ourAI':
                return analysisResult?.source || 'Our AI Engine';
            case 'externalAI':
                return externalAnalysisResult?.source || 'External AI Provider';
            case 'report':
                return reportData?.source || 'Combined Analysis';
            default:
                return 'Unknown Source';
        }
    };

    return (
        <ResultsContainer>
            <BackButton onClick={handleGoBack}>Back to Upload</BackButton>
            <ResultsGrid>
                <ImageSection>
                    <ImagePreview imageData={imageData} />
                </ImageSection>

                <AnalysisSection>
                    <AnalysisHeader>
                        <AnalysisTitle>
                            {activeView === 'ourAI' ? 'Our AI Analysis' :
                                activeView === 'externalAI' ? 'External AI Opinion' :
                                    'Generated Report'}
                        </AnalysisTitle>
                    </AnalysisHeader>

                    <AnalysisTabsContainer>
                        <AnalysisTab
                            onClick={() => { setError(''); setActiveView('ourAI'); }}
                            active={activeView === 'ourAI'}
                            disabled={loading}
                        >
                            Our AI
                        </AnalysisTab>
                        <AnalysisTab
                            onClick={handleExternalAIAnalysis}
                            active={activeView === 'externalAI'}
                            disabled={loading || !imageData}
                        >
                            External AI
                        </AnalysisTab>
                        <AnalysisTab
                            onClick={handleGenerateReport}
                            active={activeView === 'report'}
                            disabled={loading || !imageData}
                        >
                            Report
                        </AnalysisTab>
                    </AnalysisTabsContainer>

                    <AnalysisSource>
                        Source: <span>{getAnalysisSource()}</span>
                        {activeView === 'ourAI' && analysisResult?.success &&
                            analysisResult?.confidence && !loading &&
                            ` (Confidence: ${(analysisResult.confidence * 100).toFixed(0)}%)`}
                    </AnalysisSource>

                    {/* Show error specific to the view */}
                    {error && activeView !== 'ourAI' && <ErrorMessage>{error}</ErrorMessage>}

                    {/* Show Spinner or Content Area */}
                    {loading ? (
                        <LoadingSpinner text={
                            activeView === 'ourAI' ? 'Checking AI Status...' :
                                activeView === 'externalAI' ? 'Querying External AI...' :
                                    activeView === 'report' ? 'Generating Report...' : 'Processing...'
                        } />
                    ) : (
                        <AnalysisContent>
                            {renderAnalysisContent()}
                        </AnalysisContent>
                    )}
                </AnalysisSection>
            </ResultsGrid>
        </ResultsContainer>
    );
};

export default ResultsPage;