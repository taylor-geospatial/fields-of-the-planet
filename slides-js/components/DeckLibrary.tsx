/** @jsxImportSource theme-ui */
import Link from 'next/link';
import { useColorMode } from 'theme-ui';
import { Logo } from './Logo';

// ---------------------------------------------------------------------------
// Deck manifest — add new entries here when creating new decks
// ---------------------------------------------------------------------------

export interface DeckEntry {
  slug: string;
  title: string;
  description: string;
  author: string;
  date: string;
  tags: string[];
}

export const deckManifest: DeckEntry[] = [
  {
    slug: 'introducing-research-decks',
    title: 'Slide Template',
    description: 'A branded slide deck template for Taylor Geospatial presentations.',
    author: 'Taylor Geospatial',
    date: 'April 2026',
    tags: ['template'],
  },
];

// ---------------------------------------------------------------------------
// DeckCard component
// ---------------------------------------------------------------------------

function DeckCard({ deck }: { deck: DeckEntry }) {
  const [colorMode] = useColorMode();
  const gradient =
    colorMode === 'light'
      ? 'linear-gradient(261deg, #e0401f 0%, #5a7ab8 100%)'
      : 'linear-gradient(261deg, #ff4f2c 0%, #80a0d8 100%)';

  return (
    <Link
      href={`/decks/${deck.slug}`}
      sx={{
        display: 'flex',
        flexDirection: 'column',
        bg: 'surface',
        border: 'thin',
        borderRadius: 'lg',
        p: 5,
        textDecoration: 'none',
        color: 'text',
        transition: 'all 0.2s ease',
        cursor: 'pointer',
        position: 'relative',
        overflow: 'hidden',
        '&:hover, &:focus-visible': {
          borderColor: 'accent',
          transform: 'translateY(-2px)',
          boxShadow: '0 8px 24px rgba(0,0,0,0.15)',
        },
        '&:focus-visible': {
          outline: '2px solid var(--theme-ui-colors-accent)',
          outlineOffset: '2px',
        },
        '&::before': {
          content: '""',
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          height: '3px',
          background: gradient,
          opacity: 0,
          transition: 'opacity 0.2s ease',
        },
        '&:hover::before, &:focus-visible::before': {
          opacity: 1,
        },
      }}
    >
      <h3
        sx={{
          fontFamily: 'heading',
          fontWeight: 'bold',
          fontSize: 4,
          lineHeight: 'snug',
          m: 0,
          mb: 2,
        }}
      >
        {deck.title}
      </h3>

      <p
        sx={{
          fontSize: 2,
          lineHeight: 'normal',
          color: 'textSecondary',
          m: 0,
          mb: 4,
          flex: 1,
        }}
      >
        {deck.description}
      </p>

      <div sx={{ display: 'flex', flexWrap: 'wrap', gap: 2, mb: 4 }}>
        {deck.tags.map((tag) => (
          <span
            key={tag}
            sx={{
              fontSize: 0,
              fontFamily: 'monospace',
              color: 'accent',
              bg: 'surfaceLight',
              px: 2,
              py: '2px',
              borderRadius: 'sm',
              textTransform: 'uppercase',
              letterSpacing: '0.05em',
            }}
          >
            {tag}
          </span>
        ))}
      </div>

      <div
        sx={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          borderTop: 'thin',
          pt: 3,
          fontSize: 1,
          color: 'textMuted',
        }}
      >
        <span>{deck.author}</span>
        <span>{deck.date}</span>
      </div>
    </Link>
  );
}

// ---------------------------------------------------------------------------
// DeckLibrary component
// ---------------------------------------------------------------------------

export function DeckLibrary() {
  const [colorMode] = useColorMode();
  const gradient =
    colorMode === 'light'
      ? 'linear-gradient(261deg, #e0401f 0%, #5a7ab8 100%)'
      : 'linear-gradient(261deg, #ff4f2c 0%, #80a0d8 100%)';

  return (
    <div
      sx={{
        minHeight: '100vh',
        bg: 'background',
        display: 'flex',
        flexDirection: 'column',
      }}
    >
      <header
        sx={{
          px: [5, 6, 7],
          pt: [6, 7],
          pb: [5, 6],
        }}
      >
        <Logo height={40} />

        <h1
          sx={{
            fontFamily: 'heading',
            fontWeight: 'bold',
            fontSize: [7, 8],
            lineHeight: 'tight',
            m: 0,
            mt: 5,
            background: gradient,
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            backgroundClip: 'text',
          }}
        >
          Slide Decks
        </h1>

        <p
          sx={{
            fontFamily: 'body',
            fontSize: 3,
            color: 'textSecondary',
            m: 0,
            mt: 3,
            maxWidth: '640px',
            lineHeight: 'normal',
          }}
        >
          Taylor Geospatial presentation decks — version controlled, on-brand, and
          developer-friendly.
        </p>
      </header>

      <main
        sx={{
          px: [5, 6, 7],
          pb: [6, 7],
          flex: 1,
        }}
      >
        <ul
          aria-label="Available decks"
          sx={{
            display: 'grid',
            gridTemplateColumns: ['1fr', 'repeat(2, 1fr)', 'repeat(3, 1fr)'],
            gap: 5,
            mt: 5,
            listStyle: 'none',
            p: 0,
            m: 0,
          }}
        >
          {deckManifest.map((deck) => (
            <li key={deck.slug}>
              <DeckCard deck={deck} />
            </li>
          ))}
        </ul>

        {deckManifest.length <= 1 && (
          <p
            sx={{
              mt: 6,
              fontSize: 2,
              color: 'textMuted',
              fontFamily: 'body',
              textAlign: 'center',
              lineHeight: 'normal',
            }}
          >
            Add {deckManifest.length === 0 ? '' : 'more '}decks by creating MDX files in{' '}
            <code
              sx={{
                fontFamily: 'monospace',
                fontSize: '0.9em',
                bg: 'surface',
                color: 'accent',
                px: 1,
                py: '2px',
                borderRadius: 'sm',
              }}
            >
              pages/decks/
            </code>{' '}
            and registering them in the deck manifest.
          </p>
        )}
      </main>

      <footer
        sx={{
          px: [5, 6, 7],
          py: 4,
          borderTop: 'thin',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          fontSize: 0,
          color: 'textMuted',
          fontFamily: 'body',
          letterSpacing: '0.05em',
          textTransform: 'uppercase',
        }}
      >
        <span>Taylor Geospatial</span>
        <span>
          {deckManifest.length} {deckManifest.length === 1 ? 'deck' : 'decks'} available
        </span>
      </footer>
    </div>
  );
}
