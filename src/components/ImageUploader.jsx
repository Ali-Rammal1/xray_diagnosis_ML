import React, { useContext, useState, useRef, useCallback } from 'react';
import styled, { keyframes, css } from 'styled-components';
import { ImageContext } from '../App';
import { validateImage } from '../services/imageUtils';
import { motion } from 'framer-motion'; // We'll use framer-motion for animations

const fadeIn = keyframes`
  from { opacity: 0; }
  to { opacity: 1; }
`;

const slideUp = keyframes`
  from { transform: translateY(20px); opacity: 0; }
  to { transform: translateY(0); opacity: 1; }
`;

const pulse = keyframes`
  0% { transform: scale(1); }
  50% { transform: scale(1.05); }
  100% { transform: scale(1); }
`;

const UploaderContainer = styled(motion.div)`
  width: 100%;
  max-width: 600px;
  margin: 0 auto;
  animation: ${fadeIn} 0.5s ease-out;
`;

const DropZone = styled(motion.div)`
  border: 2px dashed ${({ theme, isDragActive, hasError }) =>
    hasError ? theme.colors.error :
        isDragActive ? theme.colors.primary :
            theme.colors.border};
  border-radius: 16px;
  padding: 2.5rem;
  text-align: center;
  cursor: pointer;
  transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
  background: ${({ theme, isDragActive }) =>
    isDragActive ? `${theme.colors.backgroundHover}80` : 'transparent'};
  box-shadow: ${({ isDragActive }) =>
    isDragActive ? '0 8px 16px rgba(0, 0, 0, 0.1)' : '0 2px 8px rgba(0, 0, 0, 0.05)'};
  
  &:hover {
    border-color: ${({ theme, hasError }) =>
    hasError ? theme.colors.error : theme.colors.primary};
    background: ${({ theme }) => `${theme.colors.backgroundHover}40`};
    transform: translateY(-2px);
    box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1);
  }

  ${({ isDragActive }) => isDragActive && css`
    animation: ${pulse} 1.5s infinite ease-in-out;
  `}
`;

const UploadIconContainer = styled(motion.div)`
    margin-bottom: 1.5rem;
`;

const UploadIcon = styled.div`
    font-size: 3.5rem;
    color: ${({ theme }) => theme.colors.primary};
    margin: 0 auto;
    transition: transform 0.3s ease;

    ${DropZone}:hover & {
        transform: scale(1.1);
    }
`;

const UploadText = styled(motion.p)`
    margin: 0;
    font-size: 1.2rem;
    font-weight: 500;
    color: ${({ theme }) => theme.colors.textPrimary};
    transition: color 0.3s ease;

    ${DropZone}:hover & {
        color: ${({ theme }) => theme.colors.primary};
    }
`;

const FileInput = styled.input`
    display: none;
`;

const SubText = styled(motion.p)`
    color: ${({ theme }) => theme.colors.textTertiary};
    font-size: 0.9rem;
    margin-top: 0.75rem;
    transition: opacity 0.3s ease;

    ${DropZone}:hover & {
        opacity: 0.9;
    }
`;

const ErrorText = styled(motion.p)`
    color: ${({ theme }) => theme.colors.error};
    font-size: 0.95rem;
    font-weight: 500;
    margin-top: 0.75rem;
    animation: ${slideUp} 0.4s ease-out;
    display: flex;
    align-items: center;
    justify-content: center;

    &::before {
        content: "⚠️";
        margin-right: 8px;
    }
`;

const PreviewContainer = styled(motion.div)`
    margin-top: 1.5rem;
    border-radius: 16px;
    overflow: hidden;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12);
    position: relative;
    transform-origin: center;
`;

const PreviewImage = styled(motion.img)`
    width: 100%;
    height: auto;
    display: block;
    transition: filter 0.3s ease;

    ${PreviewContainer}:hover & {
        filter: brightness(0.9);
    }
`;

const RemoveButton = styled(motion.button)`
    position: absolute;
    top: 1rem;
    right: 1rem;
    background: rgba(0, 0, 0, 0.6);
    color: white;
    border: none;
    border-radius: 50%;
    width: 2.5rem;
    height: 2.5rem;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    transition: all 0.2s ease-out;
    opacity: 0;
    transform: scale(0.9);

    ${PreviewContainer}:hover & {
        opacity: 1;
        transform: scale(1);
    }

    &:hover {
        background: rgba(0, 0, 0, 0.8);
        transform: scale(1.1);
    }
`;

const ImageUploader = () => {
    const { imageData, setImageData } = useContext(ImageContext);
    const [isDragActive, setIsDragActive] = useState(false);
    const [error, setError] = useState('');
    const fileInputRef = useRef(null);

    const handleDragEnter = useCallback((e) => {
        e.preventDefault();
        e.stopPropagation();
        setIsDragActive(true);
    }, []);

    const handleDragLeave = useCallback((e) => {
        e.preventDefault();
        e.stopPropagation();
        setIsDragActive(false);
    }, []);

    const handleDragOver = useCallback((e) => {
        e.preventDefault();
        e.stopPropagation();
        if (!isDragActive) {
            setIsDragActive(true);
        }
    }, [isDragActive]);

    const processFile = useCallback(async (file) => {
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
    }, [setImageData]);

    const handleDrop = useCallback((e) => {
        e.preventDefault();
        e.stopPropagation();
        setIsDragActive(false);

        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            const file = e.dataTransfer.files[0];
            processFile(file);
        }
    }, [processFile]);

    const handleFileSelect = useCallback((e) => {
        if (e.target.files && e.target.files.length > 0) {
            const file = e.target.files[0];
            processFile(file);
        }
    }, [processFile]);

    const handleRemoveImage = useCallback(() => {
        setImageData(null);
        setError('');
        if (fileInputRef.current) {
            fileInputRef.current.value = '';
        }
    }, [setImageData]);

    const containerVariants = {
        hidden: { opacity: 0 },
        visible: { opacity: 1, transition: { duration: 0.5 } }
    };

    const dropzoneVariants = {
        hover: {
            y: -5,
            boxShadow: "0 10px 20px rgba(0, 0, 0, 0.15)",
            transition: { duration: 0.3 }
        }
    };

    const previewVariants = {
        initial: { scale: 0.95, opacity: 0 },
        animate: {
            scale: 1,
            opacity: 1,
            transition: {
                type: "spring",
                stiffness: 300,
                damping: 20
            }
        },
        exit: {
            scale: 0.95,
            opacity: 0,
            transition: { duration: 0.3 }
        }
    };

    return (
        <UploaderContainer
            initial="hidden"
            animate="visible"
            variants={containerVariants}
        >
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
                        whileHover="hover"
                        variants={dropzoneVariants}
                    >
                        <UploadIconContainer
                            animate={isDragActive ? { scale: 1.1 } : { scale: 1 }}
                            transition={{ duration: 0.3 }}
                        >
                            <UploadIcon>
                                {isDragActive ? '📥' : '📁'}
                            </UploadIcon>
                        </UploadIconContainer>
                        <UploadText
                            animate={isDragActive ? { scale: 1.05 } : { scale: 1 }}
                            transition={{ duration: 0.3 }}
                        >
                            {isDragActive
                                ? 'Drop your image here'
                                : 'Drag & drop your image here or click to browse'}
                        </UploadText>
                        <SubText>
                            Supported formats: JPEG, PNG, WebP (Max: 10MB)
                        </SubText>
                        <FileInput
                            type="file"
                            ref={fileInputRef}
                            onChange={handleFileSelect}
                            accept="image/jpeg,image/png,image/webp"
                        />
                    </DropZone>
                    {error && (
                        <ErrorText
                            initial={{ opacity: 0, y: -10 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ duration: 0.3 }}
                        >
                            {error}
                        </ErrorText>
                    )}
                </>
            ) : (
                <PreviewContainer
                    variants={previewVariants}
                    initial="initial"
                    animate="animate"
                    exit="exit"
                >
                    <PreviewImage
                        src={imageData.preview}
                        alt="Preview"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        transition={{ duration: 0.5 }}
                    />
                    <RemoveButton
                        onClick={handleRemoveImage}
                        whileHover={{ scale: 1.1 }}
                        whileTap={{ scale: 0.9 }}
                    >
                        ✕
                    </RemoveButton>
                </PreviewContainer>
            )}
        </UploaderContainer>
    );
};

export default ImageUploader;