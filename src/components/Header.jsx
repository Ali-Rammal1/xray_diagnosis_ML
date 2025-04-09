// src/components/Header.jsx

import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import styled from 'styled-components';

const HeaderContainer = styled.header`
    background: ${({ theme }) => theme.colors.backgroundAlt};
    padding: 1rem 2rem;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    position: sticky;
    top: 0;
    z-index: 100;
`;

const HeaderContent = styled.div`
    display: flex;
    justify-content: space-between;
    align-items: center;
    max-width: 1200px;
    margin: 0 auto;
`;

const Logo = styled(Link)`
    display: flex;
    align-items: center;
    gap: 0.75rem;
    text-decoration: none;
    font-weight: 700;
    font-size: 1.5rem;
    color: ${({ theme }) => theme.colors.textPrimary};

    &:hover {
        color: ${({ theme }) => theme.colors.primary};
    }
`;

const LogoIcon = styled.div`
    width: 36px;
    height: 36px;
    border-radius: 8px;
    background: linear-gradient(135deg, #9d50bb, #6e48aa);
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-weight: bold;
    font-size: 1rem;
`;

const NavLinks = styled.nav`
    display: flex;
    gap: 2rem;
`;

const NavLink = styled(Link)`
  color: ${({ active, theme }) =>
    active ? theme.colors.primary : theme.colors.textSecondary};
  text-decoration: none;
  font-weight: 500;
  font-size: 1rem;
  padding: 0.5rem 0;
  position: relative;
  transition: color 0.2s;
  
  &:after {
    content: '';
    position: absolute;
    bottom: 0;
    left: 0;
    width: ${({ active }) => active ? '100%' : '0'};
    height: 2px;
    background: ${({ theme }) => theme.colors.primary};
    transition: width 0.3s ease;
  }
  
  &:hover {
    color: ${({ theme }) => theme.colors.primary};
    
    &:after {
      width: 100%;
    }
  }
`;

const Header = () => {
    const location = useLocation();

    return (
        <HeaderContainer>
            <HeaderContent>
                <Logo to="/">
                    <LogoIcon>AI</LogoIcon>
                    ImageAnalyzer
                </Logo>
                <NavLinks>
                    <NavLink to="/" active={location.pathname === '/' ? 1 : 0}>
                        Home
                    </NavLink>
                </NavLinks>
            </HeaderContent>
        </HeaderContainer>
    );
};

export default Header;