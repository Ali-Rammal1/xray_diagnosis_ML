// src/components/Header.jsx

import React, { useState, useEffect } from 'react';
import { Link, NavLink as RouterNavLink } from 'react-router-dom';
import styled, { keyframes } from 'styled-components';

const fadeIn = keyframes`
    from { opacity: 0; transform: translateY(-10px); }
    to { opacity: 1; transform: translateY(0); }
`;

const HeaderContainer = styled.header`
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1.25rem 2.5rem;
    background-color: ${({ theme, scrolled }) =>
            scrolled ? theme.colors.backgroundAlt : 'transparent'};
    box-shadow: ${({ scrolled }) =>
            scrolled ? '0 4px 20px rgba(0, 0, 0, 0.08)' : 'none'};
    position: sticky;
    top: 0;
    z-index: 100;
    transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
    animation: ${fadeIn} 0.5s ease-out;
`;

const Logo = styled(Link)`
    display: flex;
    align-items: center;
    text-decoration: none;
    color: ${({ theme }) => theme.colors.primary};
    font-weight: 700;
    font-size: 1.6rem;
    transition: all 0.3s ease;

    &:hover {
        transform: scale(1.03);
    }
`;

const LogoImage = styled.img`
    width: 38px;
    height: 38px;
    margin-right: 0.75rem;
    vertical-align: middle;
    filter: drop-shadow(0 2px 4px rgba(0, 0, 0, 0.1));
    transition: all 0.3s ease;

    ${Logo}:hover & {
        transform: rotate(5deg);
    }
`;

const LogoText = styled.span`
    position: relative;

    &::after {
        content: '';
        position: absolute;
        bottom: -4px;
        left: 0;
        width: 0;
        height: 2px;
        background: linear-gradient(90deg, #9d50bb 0%, #6e48aa 100%);
        transition: width 0.3s ease;
    }

    ${Logo}:hover &::after {
        width: 100%;
    }
`;

const Nav = styled.nav`
    display: flex;
    gap: 2rem;
`;

const slideUnderline = keyframes`
  from { transform: scaleX(0); }
  to { transform: scaleX(1); }
`;

const NavLink = styled(RouterNavLink)`
    color: ${({ theme }) => theme.colors.textSecondary};
    text-decoration: none;
    font-weight: 500;
    transition: all 0.3s ease;
    padding: 0.5rem 0;
    position: relative;
    
    &::after {
      content: '';
      position: absolute;
      width: 100%;
      transform: scaleX(0);
      height: 2px;
      bottom: 0;
      left: 0;
      background: ${({ theme }) => theme.colors.primary};
      transform-origin: bottom right;
      transition: transform 0.3s ease-out;
    }

    &:hover {
        color: ${({ theme }) => theme.colors.primary};
        
        &::after {
          transform: scaleX(1);
          transform-origin: bottom left;
        }
    }

    &.active {
        color: ${({ theme }) => theme.colors.primary};
        font-weight: 600;
        
        &::after {
          transform: scaleX(1);
          transform-origin: bottom left;
          animation: ${slideUnderline} 0.3s ease-out;
        }
    }
`;

const Header = () => {
    const [scrolled, setScrolled] = useState(false);

    useEffect(() => {
        const handleScroll = () => {
            const offset = window.scrollY;
            if (offset > 50) {
                setScrolled(true);
            } else {
                setScrolled(false);
            }
        };

        window.addEventListener('scroll', handleScroll);
        return () => {
            window.removeEventListener('scroll', handleScroll);
        };
    }, []);

    return (
        <HeaderContainer scrolled={scrolled}>
            <Logo to="/">
                <LogoImage src="/ai-xray-logo.png" alt="Lungify Logo" />
                <LogoText>Lungify</LogoText>
            </Logo>

            <Nav>
                <NavLink to="/" end>Home</NavLink>
                <NavLink to="/about-us">About Us</NavLink>
                <NavLink to="/about-model">About Model</NavLink>
            </Nav>
        </HeaderContainer>
    );
};

export default Header;