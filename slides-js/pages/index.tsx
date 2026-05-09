/** @jsxImportSource theme-ui */
import { useRouter } from 'next/router';
import { useEffect } from 'react';

/**
 * Index page — redirects to the single slide deck.
 * When more decks are added, this can be swapped back to the DeckLibrary.
 */
export default function IndexPage() {
  const router = useRouter();

  useEffect(() => {
    router.replace('/presentation');
  }, [router]);

  return (
    <div
      sx={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        height: '100vh',
        bg: 'background',
        color: 'textMuted',
        fontFamily: 'body',
        fontSize: 2,
      }}
    >
      Loading...
    </div>
  );
}
