import React from 'react';
import styled, { css, keyframes } from 'styled-components';

const spin = keyframes`
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
`;

const pulse = keyframes`
    0% { box-shadow: 0 0 0 0 rgba(110, 72, 170, 0.4); }
    70% { box-shadow: 0 0 0 10px rgba(110, 72, 170, 0); }
    100% { box-shadow: 0 0 0 0 rgba(110, 72, 170, 0); }
`;

const ButtonContainer = styled.button`
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 0.9rem 1.8rem;
    border-radius: 10px;
    font-weight: 600;
    font-size: 1rem;
    cursor: pointer;
    transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
    border: none;
    outline: none;
    position: relative;
    flex: 1;
    overflow: hidden;
    letter-spacing: 0.5px;

    // Create button ripple effect
    &::after {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 5px;
        height: 5px;
        background: rgba(255, 255, 255, 0.5);
        opacity: 0;
        border-radius: 100%;
        transform: scale(1, 1) translate(-50%, -50%);
        transform-origin: 50% 50%;
    }

    &:focus:not(:active)::after {
        animation: ripple 1s ease-out;
    }

    @keyframes ripple {
        0% {
            transform: scale(0, 0) translate(-50%, -50%);
            opacity: 0.5;
        }
        100% {
            transform: scale(20, 20) translate(-50%, -50%);
            opacity: 0;
        }
    }

    ${({ primary, active, theme }) => {
        if (primary) {
            return css`
        background: linear-gradient(135deg, #9d50bb 0%, #6e48aa 100%);
        color: white;
        box-shadow: 0 4px 6px rgba(110, 72, 170, 0.15);
        
        &:hover:not(:disabled) {
          transform: translateY(-3px);
          box-shadow: 0 7px 14px rgba(110, 72, 170, 0.25);
          background: linear-gradient(135deg, #a55bc0 0%, #7652b1 100%);
        }
        
        &:active:not(:disabled) {
          transform: translateY(-1px);
          box-shadow: 0 3px 8px rgba(110, 72, 170, 0.2);
          background: linear-gradient(135deg, #8d46a8 0%, #5e3e92 100%);
        }
        
        &:focus {
          animation: ${pulse} 1.5s infinite;
        }
      `;
        }

        if (active) {
            return css`
        background: ${theme.colors.backgroundActive};
        color: ${theme.colors.primary};
        border: 2px solid ${theme.colors.primary};
        
        &:hover:not(:disabled) {
          background: ${theme.colors.backgroundHover};
          transform: translateY(-2px);
          box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        
        &:active:not(:disabled) {
          transform: translateY(0);
        }
      `;
        }

        return css`
      background: ${theme.colors.backgroundAlt};
      color: ${theme.colors.textPrimary};
      border: 1px solid transparent;
      
      &:hover:not(:disabled) {
        background: ${theme.colors.backgroundHover};
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.06);
        border: 1px solid ${theme.colors.border};
      }
      
      &:active:not(:disabled) {
        transform: translateY(0);
      }
    `;
    }}

    &:disabled {
        opacity: 0.7;
        cursor: not-allowed;
        filter: grayscale(30%);
    }
`;

const spinnerFade = keyframes`
  0%, 100% { opacity: 0.3; }
  50% { opacity: 1; }
`;

const Spinner = styled.div`
  width: 20px;
  height: 20px;
  border: 2px solid rgba(255, 255, 255, 0.3);
  border-radius: 50%;
  border-top-color: white;
  animation: ${spin} 0.8s linear infinite, ${spinnerFade} 1.5s ease-in-out infinite;
  margin-right: 0.75rem;
`;

const AnalysisButton = ({
                            children,
                            onClick,
                            disabled,
                            loading,
                            primary,
                            active
                        }) => {
    return (
        <ButtonContainer
            onClick={onClick}
            disabled={disabled || loading}
            primary={primary}
            active={active}
        >
            {loading && <Spinner />}
            {children}
        </ButtonContainer>
    );
};

export default AnalysisButton;