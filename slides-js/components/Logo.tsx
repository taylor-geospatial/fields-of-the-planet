/** @jsxImportSource theme-ui */
import { useColorMode } from 'theme-ui';

export interface LogoProps {
  /** Height of the logo in pixels. Width scales proportionally. Default: 40 */
  height?: number;
  /** Show the wordmark text alongside the logo. Default: true */
  wordmark?: boolean;
  /** Override color (defaults to current text color) */
  color?: string;
}

/**
 * Taylor Geospatial logo component.
 * Renders the TG brandmark with optional wordmark text.
 * SVG fill adapts to the current color mode.
 */
export function Logo({ height = 40, wordmark = true, color }: LogoProps) {
  const [colorMode] = useColorMode();
  const fill = color ?? (colorMode === 'light' ? '#3b1e1c' : '#f4f4eb');

  return (
    <div sx={{ display: 'flex', alignItems: 'center', gap: 2, height }}>
      {/* TG Brandmark */}
      <svg
        width={height}
        height={height}
        viewBox="0 0 216 216"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        role="img"
        aria-label="Taylor Geospatial"
      >
        <g fill={fill}>
          <path d="M87.9,98.6l-8.8,8.8c-.3.3-.3.8,0,1.1l8.8,8.8c.6.6,1.6,0,1.3-.8-2.1-5.5-2.1-11.6,0-17,.3-.8-.7-1.4-1.3-.8Z" />
          <path d="M98.6,87.9l8.8-8.8c.3-.3.8-.3,1.1,0l8.8,8.8c.6.6,0,1.6-.8,1.3-5.5-2.1-11.6-2.1-17,0-.8.3-1.4-.7-.8-1.3Z" />
          <path d="M128.1,117.4l8.8-8.8c.3-.3.3-.8,0-1.1l-8.8-8.8c-.6-.6-1.6,0-1.3.8,2.1,5.5,2.1,11.6,0,17-.3.8.7,1.4,1.3.8Z" />
          <path d="M117.4,128.1l-8.8,8.8c-.3.3-.8.3-1.1,0l-8.8-8.8c-.6-.6,0-1.6.8-1.3,5.5,2.1,11.6,2.1,17,0,.8-.3,1.4.7.8,1.3Z" />
          <path d="M145.6,88v-16.5c0-.5-.4-1-1-1h-16.1c-1,0-1.4,1.4-.4,1.9,7,3.2,12.6,8.9,15.6,16,.4,1,1.9.7,1.9-.4Z" />
          <path d="M70.4,128v16.5c0,.5.4,1,1,1h16.1c1,0,1.4-1.4.4-1.9-7-3.2-12.6-8.9-15.6-16-.4-1-1.9-.7-1.9.4Z" />
          <path d="M128.1,145.4h16.5c.5,0,1-.4,1-1v-16.1c0-1-1.4-1.4-1.9-.4-3.2,7-8.9,12.6-16,15.6-1,.4-.7,1.9.4,1.9Z" />
          <path d="M87.9,70.6h-16.5c-.5,0-1,.4-1,1v16.1c0,1,1.4,1.4,1.9.4,3.2-7,8.9-12.6,16-15.6,1-.4.7-1.9-.4-1.9Z" />
        </g>
      </svg>

      {/* Wordmark */}
      {wordmark && (
        <span
          sx={{
            fontFamily: 'heading',
            fontWeight: 'bold',
            fontSize: `${height * 0.45}px`,
            color: color ?? 'text',
            letterSpacing: '-0.02em',
          }}
        >
          Taylor Geospatial
        </span>
      )}
    </div>
  );
}
