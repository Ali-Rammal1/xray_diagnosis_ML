import styled from 'styled-components';

// App container
export const AppContainer = styled.div`
  display: flex;
  flex-direction: column;
  min-height: 100vh;
`;

// Main content area
export const MainContent = styled.main`
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: ${({ theme }) => theme.space.md} ${({ theme }) => theme.space.sm};

  @media (min-width: ${({ theme }) => theme.breakpoints.md}) {
    padding: ${({ theme }) => theme.space.lg} ${({ theme }) => theme.space.md};
  }
`;

// Card component
export const Card = styled.div`
  background-color: ${({ theme }) => theme.colors.cardBackground};
  border-radius: ${({ theme }) => theme.borderRadius.md};
  padding: ${({ theme }) => theme.space.lg};
  box-shadow: ${({ theme }) => theme.shadows.md};
  width: 100%;
  max-width: ${props => props.maxWidth || '100%'};
`;

// Section with padding
export const Section = styled.section`
  padding: ${({ theme }) => theme.space.lg} 0;
  width: 100%;
`;

// Container with max width
export const Container = styled.div`
  width: 100%;
  max-width: ${props => props.maxWidth || '1200px'};
  margin: 0 auto;
  padding: 0 ${({ theme }) => theme.space.md};
`;

// Flex container
export const Flex = styled.div`
  display: flex;
  flex-direction: ${props => props.direction || 'row'};
  align-items: ${props => props.align || 'center'};
  justify-content: ${props => props.justify || 'flex-start'};
  gap: ${props => props.gap || '0'};
  flex-wrap: ${props => props.wrap || 'nowrap'};
`;

// Grid container --- CORRECTED: Only one definition ---
export const Grid = styled.div`
  display: grid;
  grid-template-columns: ${props => props.columns || 'repeat(1, 1fr)'};
  grid-gap: ${props => props.gap || '1rem'};

  // Optional: Responsive columns based on theme breakpoints
  @media (max-width: ${({ theme }) => theme.breakpoints.md}) {
    grid-template-columns: ${props => props.mobileColumns || 'repeat(1, 1fr)'};
  }
`;

// Divider
export const Divider = styled.hr`
  border: none;
  height: 1px;
  background-color: ${({ theme }) => theme.colors.divider};
  margin: ${({ theme }) => theme.space.md} 0;
`;

// Button base styles
export const ButtonBase = styled.button`
  display: inline-flex;
  align-items: center;
  justify-content: center;
  padding: 0.5rem 1rem;
  font-weight: 500;
  border-radius: ${({ theme }) => theme.borderRadius.md};
  transition: all ${({ theme }) => theme.transitions.normal};
  font-size: 1rem;
  border: none;

  &:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }
`;

// Text styles
export const Text = styled.p`
  font-size: ${props => props.size || '1rem'};
  color: ${props => props.color ? ({ theme }) => theme.colors[props.color] || props.color : ({ theme }) => theme.colors.textPrimary};
  font-weight: ${props => props.weight || 'normal'};
  margin-bottom: ${props => props.mb || '0'};
  text-align: ${props => props.align || 'left'};
`;

// Headings
export const Heading = styled.h2`
  font-size: ${props => props.size || '1.5rem'};
  color: ${props => props.color ? ({ theme }) => theme.colors[props.color] || props.color : ({ theme }) => theme.colors.textPrimary};
  font-weight: ${props => props.weight || '600'};
  margin-bottom: ${props => props.mb || '1rem'};
  text-align: ${props => props.align || 'left'};
`;