/**
 * Taylor Geospatial brand color palette.
 * Dark + light modes built from the TG Brand Guide.
 */
export const colors = {
  // Core brand
  primary: '#80a0d8', // Periwinkle
  accent: '#ff4f2c', // Red
  blue: '#a7d0dc', // Light Blue
  brown: '#3b1e1c', // Brown

  // Gradient (used for display headings)
  gradient: 'linear-gradient(261deg, #ff4f2c 0%, #80a0d8 100%)',

  // Background scale (dark theme — based on TG Brown)
  background: '#1a0f0e',
  surface: '#261816',
  surfaceLight: '#33211f',

  // Text scale (based on TG Ivory)
  text: '#f4f4eb',
  textSecondary: '#c4c4b8',
  textMuted: '#8a8a7e',

  // Borders
  border: '#3d2a28',
  subtle: '#4a3634',

  // Semantic
  highlight: '#ff4f2c',
  success: '#cff29e', // TG Green
  warning: '#fbbf24',
  error: '#ff4f2c', // TG Red
} as const;

/**
 * Light mode overrides.
 * Ivory-based background, brown text.
 */
export const lightColors = {
  background: '#f4f4eb',
  surface: '#eaeade',
  surfaceLight: '#dfdfd4',

  text: '#3b1e1c',
  textSecondary: '#5c3d3a',
  textMuted: '#8a6e6b',

  border: '#ccc8ba',
  subtle: '#d9d5c8',

  // Adjust primary/accent for light bg contrast
  primary: '#5a7ab8',
  highlight: '#e0401f',
} as const;

export type ColorToken = keyof typeof colors;
