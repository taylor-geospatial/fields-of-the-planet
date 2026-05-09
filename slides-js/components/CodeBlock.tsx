/** @jsxImportSource theme-ui */
import { useEffect, useState } from 'react';
import { highlightCode } from '../theme/syntax';

export interface CodeBlockProps {
  /** The code string (from MDX code fences, this comes as children) */
  children: string;
  /** Language for syntax highlighting */
  className?: string; // MDX passes "language-python" etc. as className
  /** Optional title shown above the code block */
  title?: string;
  /** Line numbers to highlight (e.g., "3,5-7") */
  highlight?: string;
  /** Font size index in theme scale. Default: 2 (20px). Use 1 for compact, 0 for tiny. */
  fontSize?: number;
}

/**
 * Syntax-highlighted code block using shiki with the custom theme.
 * Used as the MDX override for fenced code blocks.
 */
export function CodeBlock({ children, className, title, fontSize = 2 }: CodeBlockProps) {
  const responsiveFontSize = typeof fontSize === 'number' ? [0, fontSize] : fontSize;
  const [html, setHtml] = useState<string | null>(null);

  // Extract language from className (MDX passes "language-python")
  const lang = className?.replace('language-', '') ?? 'text';

  // The children from MDX code blocks is the raw code string
  const code = typeof children === 'string' ? children.trim() : '';

  useEffect(() => {
    let cancelled = false;
    highlightCode(code, lang).then((result) => {
      if (!cancelled) {
        setHtml(result);
      }
    });
    return () => {
      cancelled = true;
    };
  }, [code, lang]);

  return (
    <div
      sx={{
        bg: 'surface',
        borderRadius: 'md',
        border: '1px solid var(--theme-ui-colors-subtle)',
        overflow: 'hidden',
        my: 3,
        fontSize: responsiveFontSize,
        position: 'relative',
        maxWidth: '100%',
        '&::before': {
          content: '""',
          position: 'absolute',
          left: 0,
          top: 0,
          bottom: 0,
          width: '3px',
          background: 'linear-gradient(180deg, #80a0d8, #ff4f2c)',
          borderRadius: '3px 0 0 3px',
          zIndex: 3,
        },
      }}
    >
      {title && (
        <div
          sx={{
            px: 3,
            py: 2,
            borderBottom: 'thin',
            color: 'textMuted',
            fontSize: 0,
            fontFamily: 'monospace',
          }}
        >
          {title}
        </div>
      )}
      {/* Scroll wrapper with right-edge fade on mobile */}
      <div sx={{ position: 'relative' }}>
        <div
          sx={{
            p: 3,
            overflowX: 'auto',
            overflowY: 'hidden',
            WebkitOverflowScrolling: 'touch',
            '& pre': { m: 0, bg: 'transparent', border: 'none', p: 0 },
            '& code': { fontFamily: 'monospace', bg: 'transparent', p: 0, fontSize: 'inherit' },
          }}
        >
          {html ? (
            // biome-ignore lint/security/noDangerouslySetInnerHtml: shiki returns pre-sanitized HTML
            <div dangerouslySetInnerHTML={{ __html: html }} />
          ) : (
            <pre sx={{ m: 0 }}>
              <code>{code}</code>
            </pre>
          )}
        </div>
        {/* Right-edge fade hint — visible on mobile, hidden on larger screens */}
        <div
          aria-hidden="true"
          sx={{
            position: 'absolute',
            top: 0,
            right: 0,
            bottom: 0,
            width: '40px',
            background: 'linear-gradient(to right, transparent, var(--theme-ui-colors-surface))',
            pointerEvents: 'none',
            zIndex: 2,
            '@media (min-width: 40em)': { display: 'none' },
          }}
        />
      </div>
    </div>
  );
}
