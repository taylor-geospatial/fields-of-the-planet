/** @jsxImportSource theme-ui */
import type { MDXComponents } from 'mdx/types';
import type React from 'react';
import { isValidElement } from 'react';
import { CodeBlock } from './components/CodeBlock';
import { Columns } from './components/Columns';
import { Diagram } from './components/Diagram';
import { Embed } from './components/Embed';
import { Logo } from './components/Logo';
import { SectionSlide } from './components/SectionSlide';
import { SpeakerNotes } from './components/SpeakerNotes';
import { TitleSlide } from './components/TitleSlide';

/**
 * Img wrapper that prepends the GitHub Pages base path to absolute src URLs.
 * Use <Img> instead of <img> in MDX — explicit JSX bypasses MDX component overrides.
 */
function Img({ src, ...props }: React.ImgHTMLAttributes<HTMLImageElement>) {
  const prefixed = src ? `${process.env.NEXT_PUBLIC_BASE_PATH ?? ''}${src}` : src;
  // biome-ignore lint/a11y/useAltText: alt is spread via props
  return <img src={prefixed} {...props} />;
}

/**
 * MDX component overrides for Next.js.
 *
 * Critical: `hr` → `<hr data-slide-separator />` for slide splitting.
 *
 * We also apply theme-aware sx styles to all prose elements here because
 * Theme-UI's `styles.*` config only affects `Styled.*` components, not
 * native HTML elements. Without these overrides, MDX renders at browser
 * default sizes (16px) regardless of the theme.
 */
export function useMDXComponents(components: MDXComponents): MDXComponents {
  return {
    ...components,

    // --- Slide separator ---
    hr: (props: React.DetailedHTMLProps<React.HTMLAttributes<HTMLHRElement>, HTMLHRElement>) => (
      <hr {...props} data-slide-separator="" />
    ),

    // --- Prose elements with theme-ui font sizes ---
    h1: (props) => (
      <h1
        sx={{
          fontFamily: 'heading',
          fontWeight: 'bold',
          fontSize: [4, 6, 7],
          lineHeight: 'tight',
          mt: 0,
          mb: [3, 4],
          letterSpacing: '-0.02em',
          color: 'text',
          maxWidth: '100%',
          overflowWrap: 'break-word',
          wordBreak: 'break-word',
        }}
        {...props}
      />
    ),
    h2: (props) => (
      <h2
        sx={{
          fontFamily: 'heading',
          fontWeight: 'bold',
          fontSize: [3, 5, 6],
          lineHeight: 'tight',
          mt: 0,
          mb: [2, 3],
          letterSpacing: '-0.01em',
          color: 'text',
        }}
        {...props}
      />
    ),
    h3: (props) => (
      <h3
        sx={{
          fontFamily: 'heading',
          fontWeight: 'medium',
          fontSize: [2, 3, 4],
          lineHeight: 'snug',
          mt: 0,
          mb: [2, 3],
          color: 'text',
        }}
        {...props}
      />
    ),
    p: (props) => (
      <p
        sx={{
          fontSize: [1, 3, 4],
          lineHeight: 'relaxed',
          mt: 0,
          mb: [2, 3],
          color: 'text',
          maxWidth: '100%',
          overflowWrap: 'break-word',
          wordBreak: 'break-word',
        }}
        {...props}
      />
    ),
    ul: (props) => (
      <ul
        sx={{
          fontSize: [1, 3, 4],
          lineHeight: 'relaxed',
          pl: [3, 5],
          mt: 0,
          mb: [2, 3],
          maxWidth: '100%',
          listStyleType: 'none',
          '& > li': {
            position: 'relative',
            pl: 4,
            '&::before': {
              content: '""',
              position: 'absolute',
              left: 0,
              top: '0.65em',
              width: '6px',
              height: '1.5px',
              bg: 'textMuted',
              borderRadius: 'full',
            },
          },
        }}
        {...props}
      />
    ),
    ol: (props) => (
      <ol
        sx={{
          fontSize: [1, 3, 4],
          lineHeight: 'relaxed',
          pl: [3, 5],
          mt: 0,
          mb: [2, 3],
          maxWidth: '100%',
          counterReset: 'ol-counter',
          listStyleType: 'none',
          '& > li': {
            position: 'relative',
            pl: 5,
            counterIncrement: 'ol-counter',
            '&::before': {
              content: 'counter(ol-counter) "."',
              position: 'absolute',
              left: 0,
              top: 0,
              fontFamily: 'monospace',
              fontSize: '0.85em',
              fontWeight: 'medium',
              color: 'textMuted',
            },
          },
        }}
        {...props}
      />
    ),
    li: (props) => <li sx={{ mb: 2, maxWidth: '100%', overflowWrap: 'break-word', wordBreak: 'break-word' }} {...props} />,
    blockquote: (props) => (
      <blockquote
        sx={{
          borderLeft: '3px solid var(--theme-ui-colors-accent)',
          pl: 4,
          py: 3,
          ml: 0,
          my: 4,
          bg: 'surface',
          borderRadius: 'sm',
          color: 'textSecondary',
          fontStyle: 'italic',
          fontSize: [1, 3, 4],
          '& p': { m: 0, fontSize: 'inherit' },
        }}
        {...props}
      />
    ),
    strong: (props) => <strong sx={{ fontWeight: 'bold', color: 'text' }} {...props} />,
    a: (props) => (
      <a
        sx={{
          color: 'primary',
          textDecoration: 'none',
          '&:hover': { textDecoration: 'underline' },
        }}
        {...props}
      />
    ),
    code: (props) => (
      <code
        sx={{
          fontFamily: 'monospace',
          fontSize: '0.85em',
          bg: 'surface',
          color: 'accent',
          px: 1,
          py: '2px',
          borderRadius: 'sm',
        }}
        {...props}
      />
    ),

    // --- Code block (fenced) ---
    // Note: the `code` override below transforms <code> elements, so children.type is no longer
    // the string 'code' — we check for string children instead to detect fenced code blocks.
    pre: ({
      children,
      ...props
    }: React.DetailedHTMLProps<React.HTMLAttributes<HTMLPreElement>, HTMLPreElement>) => {
      if (isValidElement(children)) {
        const { className, children: codeContent } = children.props as {
          className?: string;
          children?: string;
        };
        if (typeof codeContent === 'string') {
          return <CodeBlock className={className}>{codeContent}</CodeBlock>;
        }
      }
      return <pre {...props}>{children}</pre>;
    },

    // --- Custom slide components ---
    Img,
    TitleSlide,
    SectionSlide,
    Embed,
    Columns,
    CodeBlock,
    Logo,
    SpeakerNotes,
    Diagram,
  };
}
