import React from 'react';
import { Link } from 'react-router-dom';
import styled from 'styled-components';

const NotFoundContainer = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100%;
  padding: 2rem;
  text-align: center;
`;

const ErrorCode = styled.h1`
  font-size: 8rem;
  margin: 0;
  background: linear-gradient(90deg, #9d50bb 0%, #6e48aa 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
`;

const ErrorMessage = styled.h2`
  font-size: 2rem;
  margin-bottom: 2rem;
  color: ${({ theme }) => theme.colors.textPrimary};
`;

const StyledLink = styled(Link)`
  display: inline-block;
  padding: 0.8rem 1.5rem;
  background: ${({ theme }) => theme.colors.primary};
  color: white;
  text-decoration: none;
  border-radius: 8px;
  font-weight: 600;
  transition: transform 0.2s, background-color 0.2s;
  
  &:hover {
    background: ${({ theme }) => theme.colors.primaryHover};
    transform: translateY(-2px);
  }
`;

const NotFoundPage = () => {
    return (
        <NotFoundContainer>
            <ErrorCode>404</ErrorCode>
            <ErrorMessage>Page Not Found</ErrorMessage>
            <StyledLink to="/">Return to Home</StyledLink>
        </NotFoundContainer>
    );
};

export default NotFoundPage;