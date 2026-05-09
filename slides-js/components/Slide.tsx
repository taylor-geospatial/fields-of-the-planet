/** @jsxImportSource theme-ui */
import type { ReactNode } from 'react';

export interface SlideProps {
  /** The slide content (one group from splitChildren) */
  children: ReactNode;
  /** Zero-indexed slide number */
  index: number;
  /** Total number of slides */
  total: number;
  /** Whether to show speaker notes */
  showNotes: boolean;
}

/**
 * Individual slide wrapper that provides consistent full-viewport layout.
 * Centers content vertically and provides responsive padding.
 * Includes a subtle radial gradient background for depth.
 */
export function Slide({ children, index, total }: SlideProps) {
  return (
    <div
      sx={{
        width: '100vw',
        height: '100vh',
        display: 'flex',
        flexDirection: 'column',
        justifyContent: ['flex-start', 'center'],
        px: [3, 5, 7],
        py: [3, 5, 6],
        boxSizing: 'border-box',
        position: 'relative',
        overflowX: 'hidden',
        overflowY: ['auto', 'hidden'],
      }}
    >
      {/* Subtle radial gradient for depth */}
      <div
        aria-hidden="true"
        sx={{
          position: 'absolute',
          inset: 0,
          pointerEvents: 'none',
          background:
            'radial-gradient(ellipse 80% 60% at 70% 40%, rgba(128,160,216,0.04) 0%, transparent 70%), radial-gradient(ellipse 60% 50% at 20% 80%, rgba(255,79,44,0.03) 0%, transparent 60%)',
          zIndex: 0,
        }}
      />

      {/* Content */}
      <div
        sx={{
          flex: 1,
          minWidth: 0,
          minHeight: 0,
          width: '100%',
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'center',
          position: 'relative',
          zIndex: 1,
        }}
      >
        {children}
      </div>

      {/* Progress bar */}
      <div
        aria-hidden="true"
        sx={{
          position: 'absolute',
          bottom: 0,
          left: 0,
          right: 0,
          height: '3px',
          bg: 'accent',
          width: `${((index + 1) / total) * 100}%`,
          transition: 'width 0.3s ease',
          zIndex: 1,
        }}
      />
    </div>
  );
}
