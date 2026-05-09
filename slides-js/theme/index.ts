import type { Theme } from 'theme-ui';
import { colors, lightColors } from './colors';
import { fontSizes, fontWeights, fonts, lineHeights } from './fonts';

const theme: Theme = {
  config: {
    initialColorModeName: 'dark',
    useColorSchemeMediaQuery: false,
  },

  // --- Design Tokens ---
  colors: {
    text: colors.text,
    background: colors.background,
    primary: colors.primary,
    secondary: colors.textSecondary,
    muted: colors.surface,
    accent: colors.accent,
    highlight: colors.highlight,
    // Custom tokens (accessible via sx={{ color: 'blue' }})
    blue: colors.blue,
    surface: colors.surface,
    surfaceLight: colors.surfaceLight,
    border: colors.border,
    subtle: colors.subtle,
    textSecondary: colors.textSecondary,
    textMuted: colors.textMuted,
    success: colors.success,
    warning: colors.warning,
    error: colors.error,

    modes: {
      light: {
        text: lightColors.text,
        background: lightColors.background,
        primary: lightColors.primary,
        secondary: lightColors.textSecondary,
        muted: lightColors.surface,
        highlight: lightColors.highlight,
        surface: lightColors.surface,
        surfaceLight: lightColors.surfaceLight,
        border: lightColors.border,
        subtle: lightColors.subtle,
        textSecondary: lightColors.textSecondary,
        textMuted: lightColors.textMuted,
      },
    },
  },

  fonts: {
    body: fonts.body,
    heading: fonts.heading,
    monospace: fonts.monospace,
  },

  fontSizes: [...fontSizes],
  fontWeights: {
    ...fontWeights,
    body: fontWeights.regular,
    heading: fontWeights.bold,
  },
  lineHeights,

  space: [0, 4, 8, 16, 24, 32, 48, 64, 96, 128],

  radii: {
    none: 0,
    sm: 4,
    md: 8,
    lg: 12,
    xl: 16,
    full: 9999,
  },

  // Use CSS vars so borders respond to color mode changes
  borders: {
    thin: '1px solid var(--theme-ui-colors-border)',
    thick: '2px solid var(--theme-ui-colors-border)',
    accent: '2px solid var(--theme-ui-colors-accent)',
  },

  // --- Base HTML Element Styles ---
  styles: {
    root: {
      fontFamily: 'body',
      fontWeight: 'body',
      lineHeight: 'normal',
      color: 'text',
      bg: 'background',
      WebkitFontSmoothing: 'antialiased',
      MozOsxFontSmoothing: 'grayscale',
    },
    h1: {
      fontFamily: 'heading',
      fontWeight: 'heading',
      fontSize: [7, 8],
      lineHeight: 'tight',
      mt: 0,
      mb: 4,
      letterSpacing: '-0.02em',
    },
    h2: {
      fontFamily: 'heading',
      fontWeight: 'heading',
      fontSize: [6, 7],
      lineHeight: 'tight',
      mt: 0,
      mb: 3,
      letterSpacing: '-0.01em',
    },
    h3: {
      fontFamily: 'heading',
      fontWeight: 'medium',
      fontSize: 5,
      lineHeight: 'snug',
      mt: 0,
      mb: 3,
    },
    h4: {
      fontFamily: 'heading',
      fontWeight: 'medium',
      fontSize: 4,
      lineHeight: 'snug',
      mt: 0,
      mb: 2,
    },
    p: {
      fontSize: 6,
      lineHeight: 'normal',
      mt: 0,
      mb: 3,
    },
    a: {
      color: 'accent',
      textDecoration: 'none',
      '&:hover': {
        textDecoration: 'underline',
      },
    },
    ul: {
      fontSize: 6,
      lineHeight: 'relaxed',
      pl: 5,
      mt: 0,
      mb: 3,
      listStyleType: 'none',
      '& > li': {
        position: 'relative',
        pl: 4,
        '&::before': {
          content: '""',
          position: 'absolute',
          left: 0,
          top: '0.65em',
          width: '6px',
          height: '1.5px',
          bg: 'textMuted',
          borderRadius: 'full',
        },
      },
    },
    ol: {
      fontSize: 6,
      lineHeight: 'relaxed',
      pl: 5,
      mt: 0,
      mb: 3,
      counterReset: 'ol-counter',
      listStyleType: 'none',
      '& > li': {
        position: 'relative',
        pl: 5,
        counterIncrement: 'ol-counter',
        '&::before': {
          content: 'counter(ol-counter) "."',
          position: 'absolute',
          left: 0,
          top: 0,
          fontFamily: 'monospace',
          fontSize: '0.85em',
          fontWeight: 'medium',
          color: 'textMuted',
        },
      },
    },
    li: {
      mb: 2,
    },
    blockquote: {
      borderLeft: '3px solid var(--theme-ui-colors-accent)',
      pl: 4,
      py: 3,
      ml: 0,
      my: 4,
      bg: 'surface',
      borderRadius: 'sm',
      color: 'textSecondary',
      fontStyle: 'italic',
      '& p': { m: 0 },
    },
    code: {
      fontFamily: 'monospace',
      fontSize: '0.9em',
      bg: 'surface',
      color: 'accent',
      px: 1,
      py: '2px',
      borderRadius: 'sm',
    },
    pre: {
      fontFamily: 'monospace',
      fontSize: 3,
      lineHeight: 'relaxed',
      bg: 'surface',
      color: 'text',
      p: 4,
      borderRadius: 'md',
      border: 'thin',
      overflow: 'auto',
    },
    hr: {
      display: 'none',
    },
    img: {
      maxWidth: '100%',
      height: 'auto',
    },
    table: {
      borderCollapse: 'collapse',
      fontSize: 3,
      mb: 3,
      borderRadius: 'md',
      overflow: 'hidden',
    },
    th: {
      borderBottom: '2px solid var(--theme-ui-colors-border)',
      p: 3,
      textAlign: 'left',
      fontWeight: 'bold',
      fontSize: 2,
      color: 'text',
    },
    td: {
      borderBottom: 'thin',
      p: 3,
    },
    tr: {
      '&:nth-of-type(even)': {
        bg: 'surface',
      },
    },
  },
};

export default theme;
