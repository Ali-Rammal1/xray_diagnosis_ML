import React, { useContext, useState, useRef } from 'react';
import styled from 'styled-components';
import { ImageContext } from '../App';
import { validateImage } from '../services/imageUtils';

const UploaderContainer = styled.div`
  width: 100%;
  max-width: 600px;
  margin: 0 auto;
`;

const DropZone = styled.div`
  border: 2px dashed ${({ theme, isDragActive, hasError }) =>
    hasError ? theme.colors.error :
        isDragActive ? theme.colors.primary :
            theme.colors.border};
  border-radius: 12px;
  padding: 2rem;
  text-align: center;
  cursor: pointer;
  transition: all 0.2s;
  background: ${({ theme, isDragActive }) =>
    isDragActive ? theme.colors.backgroundHover : 'transparent'};
  
  &:hover {
    border-color: ${({ theme, hasError }) =>
    hasError ? theme.colors.error : theme.colors.primary};
    background: ${({ theme }) => theme.colors.backgroundHover};
  }
`;

const UploadIcon = styled.div`
  font-size: 3rem;
  margin-bottom: 1rem;
  color: ${({ theme }) => theme.colors.textSecondary};
`;

const UploadText = styled.p`
  margin: 0;
  font-size: 1.1rem;
  color: ${({ theme }) => theme.colors.textSecondary};
`;

const FileInput = styled.input`
  display: none;
`;

const SubText = styled.p`
  color: ${({ theme }) => theme.colors.textTertiary};
  font-size: 0.9rem;
  margin-top: 0.5rem;
`;

const ErrorText = styled.p`
  color: ${({ theme }) => theme.colors.error};
  font-size: 0.9rem;
  margin-top: 0.5rem;
`;

const PreviewContainer = styled.div`
  margin-top: 1.5rem;
  border-radius: 12px;
  overflow: hidden;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
  position: relative;
`;

const PreviewImage = styled.img`
  width: 100%;
  height: auto;
  display: block;
`;

const RemoveButton = styled.button`
  position: absolute;
  top: 0.75rem;
  right: 0.75rem;
  background: rgba(0, 0, 0, 0.6);
  color: white;
  border: none;
  border-radius: 50%;
  width: 2rem;
  height: 2rem;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition: background 0.2s;
  
  &:hover {
    background: rgba(0, 0, 0, 0.8);
  }
`;

const ImageUploader = () => {
    const { imageData, setImageData } = useContext(ImageContext);
    const [isDragActive, setIsDragActive] = useState(false);
    const [error, setError] = useState('');
    const fileInputRef = useRef(null);

    const handleDragEnter = (e) => {
        e.preventDefault();
        e.stopPropagation();
        setIsDragActive(true);
    };

    const handleDragLeave = (e) => {
        e.preventDefault();
        e.stopPropagation();
        setIsDragActive(false);
    };

    const handleDragOver = (e) => {
        e.preventDefault();
        e.stopPropagation();
        if (!isDragActive) {
            setIsDragActive(true);
        }
    };

    const processFile = async (file) => {
        try {
            const validationResult = await validateImage(file);

            if (!validationResult.valid) {
                setError(validationResult.error);
                return;
            }

            setError('');

            const reader = new FileReader();
            reader.onload = (e) => {
                setImageData({
                    file,
                    preview: e.target.result,
                    name: file.name
                });
            };
            reader.readAsDataURL(file);

        } catch (err) {
            console.error('File processing error:', err);
            setError('An error occurred while processing the image');
        }
    };

    const handleDrop = (e) => {
        e.preventDefault();
        e.stopPropagation();
        setIsDragActive(false);

        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            const file = e.dataTransfer.files[0];
            processFile(file);
        }
    };

    const handleFileSelect = (e) => {
        if (e.target.files && e.target.files.length > 0) {
            const file = e.target.files[0];
            processFile(file);
        }
    };

    const handleRemoveImage = () => {
        setImageData(null);
        setError('');
        if (fileInputRef.current) {
            fileInputRef.current.value = '';
        }
    };

    return (
        <UploaderContainer>
            {!imageData ? (
                <>
                    <DropZone
                        onClick={() => fileInputRef.current.click()}
                        onDragEnter={handleDragEnter}
                        onDragLeave={handleDragLeave}
                        onDragOver={handleDragOver}
                        onDrop={handleDrop}
                        isDragActive={isDragActive}
                        hasError={!!error}
                    >
                        <UploadIcon>📁</UploadIcon>
                        <UploadText>
                            {isDragActive
                                ? 'Drop your image here'
                                : 'Drag & drop your image here or click to browse'}
                        </UploadText>
                        <SubText>Supported formats: JPEG, PNG, WebP (Max: 10MB)</SubText>
                        <FileInput
                            type="file"
                            ref={fileInputRef}
                            onChange={handleFileSelect}
                            accept="image/jpeg,image/png,image/webp"
                        />
                    </DropZone>
                    {error && <ErrorText>{error}</ErrorText>}
                </>
            ) : (
                <PreviewContainer>
                    <PreviewImage src={imageData.preview} alt="Preview" />
                    <RemoveButton onClick={handleRemoveImage}>✕</RemoveButton>
                </PreviewContainer>
            )}
        </UploaderContainer>
    );
};

export default ImageUploader;