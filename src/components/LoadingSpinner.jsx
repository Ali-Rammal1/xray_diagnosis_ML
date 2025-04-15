import React from 'react';
import styled, { keyframes } from 'styled-components';

// Create a more sophisticated pulse animation to accompany the spin
const pulse = keyframes`
    0% { transform: scale(1); opacity: 1; }
    50% { transform: scale(1.05); opacity: 0.8; }
    100% { transform: scale(1); opacity: 1; }
`;

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
    transition: all 0.3s ease;
`;

// More modern dual spinner design with gradient
const Spinner = styled.div`
  position: relative;
  width: 60px;
  height: 60px;
  margin-bottom: 1.5rem;
  
  &::before, &::after {
    content: '';
    position: absolute;
    border-radius: 50%;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
  }
  
  &::before {
    border: 3px solid transparent;
    border-top-color: ${({ theme }) => theme.colors.primary};
    border-right-color: ${({ theme }) => theme.colors.primary};
    animation: ${spin} 1s cubic-bezier(0.68, -0.55, 0.27, 1.55) infinite;
  }
  
  &::after {
    border: 3px solid transparent;
    border-bottom-color: ${({ theme }) => `${theme.colors.primary}80`};
    border-left-color: ${({ theme }) => `${theme.colors.primary}80`};
    animation: ${spin} 1.5s cubic-bezier(0.68, -0.55, 0.27, 1.55) infinite reverse;
  }
`;

const LoadingText = styled.p`
  color: ${({ theme }) => theme.colors.textSecondary};
  font-size: 1rem;
  font-weight: 500;
  margin: 0;
  position: relative;
  animation: ${pulse} 2s ease-in-out infinite;
  
  &::after {
    content: '';
    position: absolute;
    bottom: -6px;
    left: 50%;
    width: 0;
    height: 2px;
    background: ${({ theme }) => theme.colors.primary};
    transform: translateX(-50%);
    transition: width 0.3s ease;
  }
  
  &:hover::after {
    width: 100%;
  }
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