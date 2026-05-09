/** @jsxImportSource theme-ui */
import { useColorMode } from 'theme-ui';

export interface TitleSlideProps {
  /** Presentation title — large display text */
  title: string;
  /** Subtitle or event name */
  subtitle?: string;
  /** Speaker name(s) */
  author?: string;
  /** Presentation date */
  date?: string;
}

const GRADIENT_DARK = 'linear-gradient(261deg, #ff4f2c 0%, #80a0d8 100%)';
const GRADIENT_LIGHT = 'linear-gradient(261deg, #e0401f 0%, #5a7ab8 100%)';

/**
 * Cover/title slide with logo, title, subtitle, author, and date.
 * The title text uses the TG brand gradient as a text fill.
 */
export function TitleSlide({ title, subtitle, author, date }: TitleSlideProps) {
  const [colorMode] = useColorMode();
  const gradient = colorMode === 'light' ? GRADIENT_LIGHT : GRADIENT_DARK;

  return (
    <div
      data-slide-type="title"
      sx={{
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'center',
        alignItems: 'flex-start',
        height: '100%',
        gap: 4,
      }}
    >
      <h1
        sx={{
          fontSize: [5, 7, 8],
          fontWeight: 'bold',
          lineHeight: 'tight',
          m: 0,
          background: gradient,
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent',
          backgroundClip: 'text',
        }}
      >
        {title}
      </h1>

      {subtitle && (
        <p sx={{ fontSize: [2, 4], color: 'textSecondary', m: 0, fontWeight: 'medium' }}>{subtitle}</p>
      )}

      <div sx={{ display: 'flex', gap: 3, alignItems: 'center', color: 'textMuted', fontSize: 2 }}>
        {author && <span>{author}</span>}
        {author && date && <span sx={{ color: 'border' }}>|</span>}
        {date && <span>{date}</span>}
      </div>
    </div>
  );
}
