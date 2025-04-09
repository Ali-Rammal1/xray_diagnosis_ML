// src/components/Footer.jsx

import React from 'react';
import styled from 'styled-components';

const FooterContainer = styled.footer`
    background: ${({ theme }) => theme.colors.backgroundAlt};
    padding: 2rem;
    margin-top: auto;
`;

const FooterContent = styled.div`
    display: flex;
    justify-content: space-between;
    align-items: center;
    max-width: 1200px;
    margin: 0 auto;

    @media (max-width: 768px) {
        flex-direction: column;
        gap: 1rem;
        text-align: center;
    }
`;

const Copyright = styled.p`
    margin: 0;
    color: ${({ theme }) => theme.colors.textSecondary};
    font-size: 0.9rem;
`;

const FooterLinks = styled.div`
  display: flex;
  gap: 1.5rem;
`;

const FooterLink = styled.a`
  color: ${({ theme }) => theme.colors.textSecondary};
  text-decoration: none;
  font-size: 0.9rem;
  transition: color 0.2s;
  
  &:hover {
    color: ${({ theme }) => theme.colors.primary};
  }
`;

const Footer = () => {
    const currentYear = new Date().getFullYear();

    return (
        <FooterContainer>
            <FooterContent>
                <Copyright>
                    © {currentYear} AI Image Analyzer | All Rights Reserved
                </Copyright>
                <FooterLinks>
                </FooterLinks>
            </FooterContent>
        </FooterContainer>
    );
};

export default Footer;