/**
 * Font family definitions.
 * Space Grotesk for body/heading, JetBrains Mono for code.
 */
export const fonts = {
  body: '"Space Grotesk", system-ui, -apple-system, sans-serif',
  heading: '"Space Grotesk", system-ui, -apple-system, sans-serif',
  monospace: '"JetBrains Mono", "Fira Code", "Cascadia Code", monospace',
} as const;

export const fontSizes = [
  '0.875rem', // 0: 14px (caption)
  '1rem', // 1: 16px (small / code)
  '1.25rem', // 2: 20px (body)
  '1.5rem', // 3: 24px (large body / bullets)
  '1.75rem', // 4: 28px (h4)
  '2.25rem', // 5: 36px (h3)
  '3rem', // 6: 48px (h2)
  '4rem', // 7: 64px (h1)
  '5.5rem', // 8: 88px (display)
] as const;

export const fontWeights = {
  regular: 400,
  medium: 500,
  bold: 700,
} as const;

export const lineHeights = {
  tight: 1.1,
  snug: 1.25,
  normal: 1.5,
  relaxed: 1.75,
} as const;
