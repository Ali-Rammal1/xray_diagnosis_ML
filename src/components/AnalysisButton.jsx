import React from 'react';
import styled, { css, keyframes } from 'styled-components';

const spin = keyframes`
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
`;

const ButtonContainer = styled.button`
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 0.8rem 1.5rem;
  border-radius: 8px;
  font-weight: 600;
  font-size: 1rem;
  cursor: pointer;
  transition: all 0.2s;
  border: none;
  outline: none;
  position: relative;
  flex: 1;

  ${({ primary, active, theme }) => {
    if (primary) {
        return css`
        background: linear-gradient(90deg, #9d50bb 0%, #6e48aa 100%);
        color: white;
        
        &:hover:not(:disabled) {
          transform: translateY(-2px);
          box-shadow: 0 4px 12px rgba(110, 72, 170, 0.4);
        }
        
        &:active:not(:disabled) {
          transform: translateY(0);
        }
      `;
    }

    if (active) {
        return css`
        background: ${theme.colors.backgroundActive};
        color: ${theme.colors.primary};
        border: 1px solid ${theme.colors.primary};
      `;
    }

    return css`
      background: ${theme.colors.backgroundAlt};
      color: ${theme.colors.textPrimary};
      
      &:hover:not(:disabled) {
        background: ${theme.colors.backgroundHover};
      }
    `;
}}
  
  &:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }
`;

const Spinner = styled.div`
  width: 18px;
  height: 18px;
  border: 2px solid rgba(255, 255, 255, 0.3);
  border-radius: 50%;
  border-top-color: white;
  animation: ${spin} 1s ease-in-out infinite;
  margin-right: 0.5rem;
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