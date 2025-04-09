import React from 'react';
import styled, { keyframes } from 'styled-components';

const spin = keyframes`
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
`;

const SpinnerContainer = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100%;
  padding: 2rem;
`;

const Spinner = styled.div`
  width: 50px;
  height: 50px;
  border: 4px solid ${({ theme }) => theme.colors.backgroundHover};
  border-radius: 50%;
  border-top-color: ${({ theme }) => theme.colors.primary};
  animation: ${spin} 1s ease-in-out infinite;
  margin-bottom: 1rem;
`;

const LoadingText = styled.p`
  color: ${({ theme }) => theme.colors.textSecondary};
  font-size: 1rem;
  margin: 0;
`;

const LoadingSpinner = ({ text = 'Processing...' }) => {
    return (
        <SpinnerContainer>
            <Spinner />
            <LoadingText>{text}</LoadingText>
        </SpinnerContainer>
    );
};

export default LoadingSpinner;