export const theme = {
    colors: {
        // Primary colors
        primary: '#7b68ee',         // Bright purple
        primaryHover: '#6a5acd',    // Slightly darker purple

        // UI Background colors
        background: '#121212',       // Dark base
        backgroundAlt: '#1e1e1e',    // Slightly lighter than base
        backgroundHover: '#2d2d2d',  // For hover states
        backgroundActive: '#2a2438', // Purple-tinted dark for active states
        backgroundDarker: '#0a0a0a', // Even darker for contrast
        cardBackground: '#1e1e1e',   // For card components

        // Text colors
        textPrimary: '#ffffff',      // White for primary text
        textSecondary: '#a0a0a0',    // Light gray for secondary text
        textTertiary: '#6c6c6c',     // Darker gray for less important text

        // Border and divider
        border: '#2e2e2e',          // Subtle border color
        divider: '#2e2e2e',         // For dividers and separators

        // Utility colors
        error: '#ff5c5c',           // Red for errors
        success: '#4caf50',         // Green for success
        warning: '#ffac33',         // Orange for warnings
        info: '#2196f3',            // Blue for information
    },

    // Font sizes
    fontSizes: {
        xs: '0.75rem',    // 12px
        sm: '0.875rem',   // 14px
        md: '1rem',       // 16px
        lg: '1.125rem',   // 18px
        xl: '1.25rem',    // 20px
        '2xl': '1.5rem',  // 24px
        '3xl': '1.875rem', // 30px
        '4xl': '2.25rem',  // 36px
    },

    // Spacing scale
    space: {
        xs: '0.25rem',    // 4px
        sm: '0.5rem',     // 8px
        md: '1rem',       // 16px
        lg: '1.5rem',     // 24px
        xl: '2rem',       // 32px
        '2xl': '3rem',    // 48px
        '3xl': '4rem',    // 64px
    },

    // Border radius
    borderRadius: {
        sm: '0.25rem',    // 4px
        md: '0.5rem',     // 8px
        lg: '0.75rem',    // 12px
        xl: '1rem',       // 16px
        round: '50%',     // For circular elements
    },

    // Media queries breakpoints
    breakpoints: {
        xs: '480px',      // Mobile
        sm: '640px',      // Small tablets
        md: '768px',      // Tablets
        lg: '1024px',     // Desktop
        xl: '1280px',     // Large desktop
    },

    // Box shadows
    shadows: {
        sm: '0 1px 3px rgba(0, 0, 0, 0.12), 0 1px 2px rgba(0, 0, 0, 0.24)',
        md: '0 4px 6px rgba(0, 0, 0, 0.12), 0 1px 3px rgba(0, 0, 0, 0.24)',
        lg: '0 10px 20px rgba(0, 0, 0, 0.19), 0 6px 6px rgba(0, 0, 0, 0.23)',
        xl: '0 14px 28px rgba(0, 0, 0, 0.25), 0 10px 10px rgba(0, 0, 0, 0.22)',
    },

    // Animation speeds
    transitions: {
        fast: '0.15s',
        normal: '0.3s',
        slow: '0.5s',
    }
};