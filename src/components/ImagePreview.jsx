import React, { useState, useRef, useEffect } from 'react';
import styled, { keyframes } from 'styled-components';

const fadeIn = keyframes`
    from { opacity: 0; }
    to { opacity: 1; }
`;

const PreviewContainer = styled.div`
    position: relative;
    width: 100%;
    height: 100%;
    display: flex;
    justify-content: center;
    align-items: center;
    background: ${({ theme }) => theme.colors.backgroundDarker};
    overflow: hidden;
    border-radius: 12px;
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.12);
    animation: ${fadeIn} 0.5s ease-out;
`;

const Image = styled.img`
    max-width: 100%;
    max-height: 500px;
    object-fit: contain;
    transform: scale(${({ zoom }) => zoom});
    transition: transform 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
    opacity: ${({ isLoading }) => (isLoading ? 0.7 : 1)};
`;

const overlayFade = keyframes`
    0% { opacity: 0; }
    100% { opacity: 1; }
`;

const Controls = styled.div`
    position: absolute;
    bottom: 1.5rem;
    left: 0;
    right: 0;
    display: flex;
    justify-content: center;
    gap: 1.25rem;
    opacity: 0;
    transition: opacity 0.3s ease, transform 0.3s ease;
    transform: translateY(10px);
    animation: ${overlayFade} 0.5s ease-out forwards;
    animation-delay: 0.5s;

    ${PreviewContainer}:hover & {
        opacity: 1;
        transform: translateY(0);
    }
`;

const buttonPulse = keyframes`
  0% { transform: scale(1); }
  50% { transform: scale(1.05); }
  100% { transform: scale(1); }
`;

const ControlButton = styled.button`
    background: rgba(0, 0, 0, 0.7);
    color: white;
    border: none;
    border-radius: 50%;
    width: 40px;
    height: 40px;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    transition: all 0.3s ease;
    font-size: 1.2rem;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);

    &:hover {
        background: rgba(0, 0, 0, 0.9);
        transform: translateY(-3px);
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3);
    }
    
    &:active {
        transform: translateY(-1px);
        animation: ${buttonPulse} 0.3s ease-in-out;
    }
`;

const ZoomDisplay = styled.div`
    position: absolute;
    top: 1rem;
    right: 1rem;
    background: rgba(0, 0, 0, 0.6);
    color: white;
    padding: 0.4rem 0.8rem;
    border-radius: 20px;
    font-size: 0.85rem;
    opacity: 0;
    transition: opacity 0.3s ease;
    
    ${PreviewContainer}:hover & {
        opacity: 0.8;
    }
`;

const FileName = styled.div`
    position: absolute;
    top: 1rem;
    left: 1rem;
    background: linear-gradient(90deg, rgba(0, 0, 0, 0.7) 0%, rgba(0, 0, 0, 0.5) 100%);
    color: white;
    padding: 0.4rem 0.9rem;
    border-radius: 20px;
    font-size: 0.85rem;
    max-width: 80%;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    backdrop-filter: blur(4px);
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.15);
    opacity: 0;
    transition: opacity 0.3s ease, transform 0.3s ease;
    transform: translateY(-5px);
    animation: ${overlayFade} 0.5s ease-out forwards;
    animation-delay: 0.3s;
    
    ${PreviewContainer}:hover & {
        opacity: 1;
        transform: translateY(0);
    }
`;

const ScaleIndicator = styled.div`
    position: absolute;
    bottom: 5rem;
    left: 0;
    right: 0;
    display: flex;
    justify-content: center;
    gap: 2px;
    opacity: 0;
    transition: opacity 0.3s ease;
    
    ${PreviewContainer}:hover & {
        opacity: 0.6;
    }
`;

const ScaleBar = styled.div`
    width: 5px;
    height: 20px;
    background: white;
    border-radius: 3px;
    opacity: ${({ active }) => (active ? 1 : 0.3)};
    transform: scaleY(${({ active }) => (active ? 1 : 0.7)});
    transition: transform 0.3s ease, opacity 0.3s ease;
`;

const ImagePreview = ({ imageData }) => {
    const [zoom, setZoom] = useState(1);
    const [isLoading, setIsLoading] = useState(true);
    const imageRef = useRef(null);

    useEffect(() => {
        if (imageData) {
            setIsLoading(true);
        }
    }, [imageData]);

    const handleImageLoad = () => {
        setIsLoading(false);
    };

    const handleZoomIn = () => {
        if (zoom < 2) {
            setZoom(prev => Math.min(prev + 0.1, 2));
        }
    };

    const handleZoomOut = () => {
        if (zoom > 0.5) {
            setZoom(prev => Math.max(prev - 0.1, 0.5));
        }
    };

    const handleReset = () => {
        setZoom(1);
    };

    if (!imageData) return null;

    // Calculate active scale bars based on zoom level
    const scaleLevel = Math.round((zoom - 0.5) * 10);
    const totalBars = 15; // 0.5 to 2 in steps of 0.1

    return (
        <PreviewContainer>
            <Image
                ref={imageRef}
                src={imageData.preview}
                alt="Preview"
                zoom={zoom}
                isLoading={isLoading}
                onLoad={handleImageLoad}
            />

            <FileName>{imageData.name}</FileName>
            <ZoomDisplay>{Math.round(zoom * 100)}%</ZoomDisplay>

            <ScaleIndicator>
                {[...Array(totalBars)].map((_, i) => (
                    <ScaleBar key={i} active={i <= scaleLevel} />
                ))}
            </ScaleIndicator>

            <Controls>
                <ControlButton onClick={handleZoomOut} title="Zoom Out">
                    <span>−</span>
                </ControlButton>
                <ControlButton onClick={handleReset} title="Reset Zoom">
                    <span>↺</span>
                </ControlButton>
                <ControlButton onClick={handleZoomIn} title="Zoom In">
                    <span>+</span>
                </ControlButton>
            </Controls>
        </PreviewContainer>
    );
};

export default ImagePreview;