/** @jsxImportSource theme-ui */
import type { ReactNode } from 'react';

export interface SpeakerNotesProps {
  /** The notes content — can be Markdown/JSX */
  children: ReactNode;
}

/**
 * Speaker notes component. Content is hidden by default.
 * Shown when ?notes=true or when the N key toggles notes mode.
 *
 * The Deck component sets `data-show-notes="true"` on its container
 * when notes should be visible.
 */
export function SpeakerNotes({ children }: SpeakerNotesProps) {
  return (
    <div
      className="speaker-notes"
      sx={{
        display: 'none',
        '[data-show-notes="true"] &': {
          display: 'block',
          position: 'fixed',
          bottom: '48px', // Above SlideNav
          left: 0,
          right: 0,
          maxHeight: '30vh',
          overflow: 'auto',
          bg: 'rgba(0, 0, 0, 0.95)',
          color: 'textSecondary',
          p: 4,
          fontSize: 1,
          lineHeight: 'normal',
          borderTop: '2px solid',
          borderColor: 'accent',
          fontFamily: 'body',
          zIndex: 99,
        },
        // Always hidden in print mode
        '@media print': {
          display: 'none !important',
        },
      }}
    >
      {children}
    </div>
  );
}
