/** @jsxImportSource theme-ui */
import type { ReactNode } from 'react';

export interface DiagramProps {
  /** SVG content as children (JSX SVG elements) */
  children: ReactNode;
  /** Accessible title for the diagram */
  title?: string;
  /** Width constraint. Default: '100%' */
  width?: string | number;
  /** Height constraint. Default: 'auto' */
  height?: string | number;
}

/**
 * Wrapper for inline SVG diagrams that applies theme-aware colors.
 * Sets CSS custom properties so SVG `currentColor` and `var(--diagram-*)` work.
 */
export function Diagram({ children, title, width = '100%', height = 'auto' }: DiagramProps) {
  return (
    <figure
      sx={{
        width,
        height,
        m: 0,
        mx: 'auto',
        // Expose theme colors as CSS custom properties for SVG use
        '--diagram-primary': 'var(--theme-ui-colors-primary)',
        '--diagram-accent': 'var(--theme-ui-colors-accent)',
        '--diagram-blue': 'var(--theme-ui-colors-blue)',
        '--diagram-text': 'var(--theme-ui-colors-text)',
        '--diagram-muted': 'var(--theme-ui-colors-textMuted)',
        '--diagram-border': 'var(--theme-ui-colors-border)',
        '--diagram-surface': 'var(--theme-ui-colors-surface)',
        // SVG styling defaults
        '& svg': {
          width: '100%',
          height: 'auto',
          display: 'block',
        },
        '& text': {
          fontFamily: 'body',
          fill: 'currentColor',
        },
      }}
    >
      {title && (
        <figcaption sx={{ textAlign: 'center', mb: 2, color: 'textMuted', fontSize: 1 }}>
          {title}
        </figcaption>
      )}
      {children}
    </figure>
  );
}
