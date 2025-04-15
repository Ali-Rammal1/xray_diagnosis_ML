// src/components/Footer.jsx

import React from 'react';
import styled from 'styled-components';

const FooterContainer = styled.footer`
    background: ${({ theme }) => theme.colors.backgroundAlt};
    padding: 2.5rem 2rem;
    margin-top: auto;
    border-top: 1px solid ${({ theme }) => `${theme.colors.border}80`};
    transition: all 0.3s ease;

    &:hover {
        background: ${({ theme }) => `${theme.colors.backgroundAlt}F2`};
    }
`;

const FooterContent = styled.div`
    display: flex;
    justify-content: space-between;
    align-items: center;
    max-width: 1200px;
    margin: 0 auto;
    position: relative;

    @media (max-width: 768px) {
        flex-direction: column;
        gap: 1.5rem;
        text-align: center;
    }

    &::before {
        content: '';
        position: absolute;
        bottom: -15px;
        left: 0;
        width: 50px;
        height: 3px;
        background: linear-gradient(90deg, #9d50bb 0%, #6e48aa 100%);
        border-radius: 3px;
        transition: width 0.3s ease;

        @media (max-width: 768px) {
            left: 50%;
            transform: translateX(-50%);
        }
    }

    &:hover::before {
        width: 80px;
    }
`;

const Copyright = styled.p`
    margin: 0;
    color: ${({ theme }) => theme.colors.textSecondary};
    font-size: 0.95rem;
    transition: color 0.3s ease;

    &:hover {
        color: ${({ theme }) => theme.colors.textPrimary};
    }
`;

const FooterLinks = styled.div`
    display: flex;
    gap: 2rem;
`;

const FooterLink = styled.a`
    color: ${({ theme }) => theme.colors.textSecondary};
    text-decoration: none;
    font-size: 0.95rem;
    transition: all 0.3s ease;
    position: relative;
    padding: 0.25rem 0;

    &::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 0;
        width: 0;
        height: 2px;
        background: ${({ theme }) => theme.colors.primary};
        transition: width 0.3s ease;
    }

    &:hover {
        color: ${({ theme }) => theme.colors.primary};
        transform: translateY(-2px);

        &::after {
            width: 100%;
        }
    }
`;

const SocialIcons = styled.div`
  display: flex;
  gap: 1rem;
  
  @media (max-width: 768px) {
    margin-top: 1rem;
  }
`;

const SocialIcon = styled.a`
  color: ${({ theme }) => theme.colors.textSecondary};
  font-size: 1.2rem;
  transition: all 0.3s ease;
  
  &:hover {
    color: ${({ theme }) => theme.colors.primary};
    transform: translateY(-3px) scale(1.1);
  }
`;

const Footer = () => {
    const currentYear = new Date().getFullYear();

    return (
        <FooterContainer>
            <FooterContent>
                <Copyright>
                    © {currentYear} Lungify | All Rights Reserved
                </Copyright>
            </FooterContent>
        </FooterContainer>
    );
};

export default Footer;