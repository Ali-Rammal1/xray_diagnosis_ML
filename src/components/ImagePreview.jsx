import React, { useState } from 'react';
import styled from 'styled-components';

const PreviewContainer = styled.div`
    position: relative;
    width: 100%;
    height: 100%;
    display: flex;
    justify-content: center;
    align-items: center;
    background: ${({ theme }) => theme.colors.backgroundDarker};
    overflow: hidden;
`;

const Image = styled.img`
    max-width: 100%;
    max-height: 500px;
    object-fit: contain;
    transform: scale(${({ zoom }) => zoom});
    transition: transform 0.2s ease;
`;

const Controls = styled.div`
    position: absolute;
    bottom: 1rem;
    left: 0;
    right: 0;
    display: flex;
    justify-content: center;
    gap: 1rem;
    opacity: 0.6;
    transition: opacity 0.2s;

    &:hover {
        opacity: 1;
    }
`;

const ControlButton = styled.button`
    background: rgba(0, 0, 0, 0.7);
    color: white;
    border: none;
    border-radius: 50%;
    width: 36px;
    height: 36px;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    transition: background 0.2s;
    font-size: 1.2rem;

    &:hover {
        background: rgba(0, 0, 0, 0.9);
    }
`;

const FileName = styled.div`
  position: absolute;
  top: 1rem;
  left: 1rem;
  background: rgba(0, 0, 0, 0.6);
  color: white;
  padding: 0.3rem 0.7rem;
  border-radius: 4px;
  font-size: 0.8rem;
  max-width: 80%;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
`;

const ImagePreview = ({ imageData }) => {
    const [zoom, setZoom] = useState(1);

    const handleZoomIn = () => {
        if (zoom < 2) {
            setZoom(zoom + 0.1);
        }
    };

    const handleZoomOut = () => {
        if (zoom > 0.5) {
            setZoom(zoom - 0.1);
        }
    };

    const handleReset = () => {
        setZoom(1);
    };

    if (!imageData) return null;

    return (
        <PreviewContainer>
            <Image src={imageData.preview} alt="Preview" zoom={zoom} />

            <FileName>{imageData.name}</FileName>

            <Controls>
                <ControlButton onClick={handleZoomOut} title="Zoom Out">-</ControlButton>
                <ControlButton onClick={handleReset} title="Reset Zoom">↺</ControlButton>
                <ControlButton onClick={handleZoomIn} title="Zoom In">+</ControlButton>
            </Controls>
        </PreviewContainer>
    );
};

export default ImagePreview;