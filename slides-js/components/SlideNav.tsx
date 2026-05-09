/** @jsxImportSource theme-ui */
import { useColorMode } from 'theme-ui';

export interface SlideNavProps {
  /** Current slide number (1-indexed for display) */
  current: number;
  /** Total number of slides */
  total: number;
  /** When true, nav is completely hidden (cover/section/close slides) */
  hidden?: boolean;
}

/**
 * Minimal slide counter fixed to the bottom-right.
 * Hidden on cover, section, and closing slides.
 */
export function SlideNav({ current, total, hidden = false }: SlideNavProps) {
  const [colorMode, setColorMode] = useColorMode();

  const toggleMode = () => {
    setColorMode(colorMode === 'light' ? 'dark' : 'light');
  };

  if (hidden) return null;

  return (
    <div
      sx={{
        position: 'fixed',
        bottom: 0,
        right: 0,
        display: 'flex',
        alignItems: 'center',
        gap: 3,
        px: 4,
        py: 3,
        fontFamily: 'monospace',
        fontSize: '12px',
        color: 'textMuted',
        zIndex: 100,
        letterSpacing: '0.05em',
        opacity: 0.5,
        transition: 'opacity 0.2s ease',
        '&:hover': {
          opacity: 1,
        },
        '@media print': {
          display: 'none',
        },
      }}
    >
      <span sx={{ fontVariantNumeric: 'tabular-nums' }}>
        {current}/{total}
      </span>

      <button
        type="button"
        onClick={toggleMode}
        aria-label={`Switch to ${colorMode === 'light' ? 'dark' : 'light'} mode`}
        sx={{
          appearance: 'none',
          bg: 'transparent',
          border: 'none',
          cursor: 'pointer',
          p: 0,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          color: 'textMuted',
          transition: 'color 0.2s ease',
          '&:hover': {
            color: 'text',
          },
        }}
      >
        {colorMode === 'light' ? (
          <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
            <path d="M21.53 15.93c-.16-.27-.61-.69-1.73-.49a8.46 8.46 0 01-1.88.13 8.41 8.41 0 01-5.91-2.82 8.21 8.21 0 01-1.9-4.49 8.56 8.56 0 01.33-3.48c.26-.85-.19-1.22-.41-1.35a1.27 1.27 0 00-1.36.1A10.03 10.03 0 004 13.6c.21 5.49 4.72 9.95 10.22 10.12a10.04 10.04 0 007.31-3.79z" />
          </svg>
        ) : (
          <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
            <path d="M12 18a6 6 0 110-12 6 6 0 010 12zM11 1h2v3h-2V1zm0 19h2v3h-2v-3zM3.515 4.929l1.414-1.414L7.05 5.636 5.636 7.05 3.515 4.93zM16.95 18.364l1.414-1.414 2.121 2.121-1.414 1.414-2.121-2.121zm2.121-13.435l1.414 1.414-2.121 2.121-1.414-1.414 2.121-2.121zM5.636 16.95l1.414 1.414-2.121 2.121-1.414 1.414 2.121-2.121zM23 11v2h-3v-2h3zM4 11v2H1v-2h3z" />
          </svg>
        )}
      </button>
    </div>
  );
}
