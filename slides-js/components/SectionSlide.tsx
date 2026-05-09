/** @jsxImportSource theme-ui */
import type { ReactNode } from 'react';

export interface SectionSlideProps {
  /** The big text to display */
  children: ReactNode;
  /** Optional background image URL. A dark scrim is applied for readability. */
  backgroundImage?: string;
  /** Optional kicker text above the heading (e.g. section number) */
  kicker?: string;
}

/**
 * Big centered text section divider.
 * Use for dramatic statements or topic transitions.
 * Supports an optional satellite/background image with a dark overlay.
 */
export function SectionSlide({ children, backgroundImage, kicker }: SectionSlideProps) {
  const prefixedBg = backgroundImage
    ? `${process.env.NEXT_PUBLIC_BASE_PATH ?? ''}${backgroundImage}`
    : undefined;
  return (
    <div
      data-slide-type="section"
      sx={{
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'center',
        alignItems: 'center',
        textAlign: 'center',
        height: '100%',
        position: 'relative',
        '& > *:not([aria-hidden]):not([data-kicker])': {
          fontSize: [4, 6, 8],
          lineHeight: 'tight',
          fontWeight: 'bold',
          maxWidth: '95%',
          wordBreak: 'break-word',
          overflowWrap: 'break-word',
          position: 'relative',
          zIndex: 2,
          ...(backgroundImage && { color: '#f4f4eb' }),
        },
      }}
    >
      {backgroundImage && (
        <div
          aria-hidden="true"
          sx={{
            position: 'fixed',
            inset: 0,
            backgroundImage: `url(${prefixedBg})`,
            backgroundSize: 'cover',
            backgroundPosition: 'center',
            zIndex: 0,
            '&::after': {
              content: '""',
              position: 'absolute',
              inset: 0,
              bg: 'rgba(26, 15, 14, 0.5)',
            },
          }}
        />
      )}
      {kicker && (
        <span
          data-kicker="true"
          sx={{
            fontSize: 1,
            fontWeight: 'medium',
            letterSpacing: '0.12em',
            textTransform: 'uppercase',
            color: backgroundImage ? 'rgba(244, 244, 235, 0.6)' : 'textMuted',
            position: 'relative',
            zIndex: 2,
            mb: 3,
          }}
        >
          {kicker}
        </span>
      )}
      {children}
    </div>
  );
}
