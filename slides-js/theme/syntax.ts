import type { ThemeRegistration } from 'shiki';

/**
 * Custom syntax highlighting theme for shiki.
 * Taylor Geospatial palette — warm dark theme.
 */
export const syntaxTheme: ThemeRegistration = {
  name: 'tg-slides',
  type: 'dark',
  colors: {
    'editor.background': '#261816',
    'editor.foreground': '#f4f4eb',
    'editorLineNumber.foreground': '#5a4440',
    'editorCursor.foreground': '#ff4f2c',
    'editor.selectionBackground': '#80a0d840',
    'editor.lineHighlightBackground': '#f4f4eb08',
  },
  tokenColors: [
    // Comments
    {
      scope: ['comment', 'punctuation.definition.comment'],
      settings: {
        foreground: '#6a5753',
        fontStyle: 'italic',
      },
    },
    // Strings
    {
      scope: ['string', 'string.quoted', 'string.template'],
      settings: {
        foreground: '#cff29e', // TG Green
      },
    },
    // Numbers
    {
      scope: ['constant.numeric'],
      settings: {
        foreground: '#fbbf24',
      },
    },
    // Keywords
    {
      scope: [
        'keyword',
        'keyword.control',
        'keyword.operator.new',
        'storage.type',
        'storage.modifier',
      ],
      settings: {
        foreground: '#ff4f2c', // TG Red
      },
    },
    // Built-in types and language constants
    {
      scope: ['constant.language', 'support.type.builtin', 'variable.language'],
      settings: {
        foreground: '#ff4f2c',
      },
    },
    // Functions and methods
    {
      scope: ['entity.name.function', 'support.function', 'meta.function-call'],
      settings: {
        foreground: '#a7d0dc', // TG Light Blue
      },
    },
    // Classes and types
    {
      scope: [
        'entity.name.type',
        'entity.name.class',
        'support.class',
        'entity.other.inherited-class',
      ],
      settings: {
        foreground: '#80a0d8', // TG Periwinkle
      },
    },
    // Variables and parameters
    {
      scope: ['variable', 'variable.parameter', 'variable.other'],
      settings: {
        foreground: '#f4f4eb', // TG Ivory
      },
    },
    // Operators and punctuation
    {
      scope: ['keyword.operator', 'punctuation', 'punctuation.separator', 'punctuation.terminator'],
      settings: {
        foreground: '#c4c4b8',
      },
    },
    // Decorators / annotations
    {
      scope: ['meta.decorator', 'punctuation.decorator'],
      settings: {
        foreground: '#ff4f2c',
      },
    },
    // SQL keywords
    {
      scope: ['keyword.other.DML.sql', 'keyword.other.DDL.sql', 'keyword.other.sql'],
      settings: {
        foreground: '#ff4f2c',
        fontStyle: 'bold',
      },
    },
    // SQL functions
    {
      scope: ['support.function.sql'],
      settings: {
        foreground: '#a7d0dc',
      },
    },
    // Tags (JSX/HTML)
    {
      scope: ['entity.name.tag', 'punctuation.definition.tag'],
      settings: {
        foreground: '#ff4f2c',
      },
    },
    // Attributes (JSX/HTML)
    {
      scope: ['entity.other.attribute-name'],
      settings: {
        foreground: '#a7d0dc',
      },
    },
    // Regex
    {
      scope: ['string.regexp'],
      settings: {
        foreground: '#ff4f2c',
      },
    },
    // Markdown headings (for MDX)
    {
      scope: ['markup.heading', 'entity.name.section'],
      settings: {
        foreground: '#80a0d8',
        fontStyle: 'bold',
      },
    },
    // Markdown bold/italic
    {
      scope: ['markup.bold'],
      settings: {
        fontStyle: 'bold',
      },
    },
    {
      scope: ['markup.italic'],
      settings: {
        fontStyle: 'italic',
      },
    },
  ],
};

/**
 * Highlight code using shiki with the custom theme.
 * Dynamically imports shiki to avoid SSR bundle bloat.
 */
export async function highlightCode(code: string, lang: string): Promise<string> {
  const { codeToHtml } = await import('shiki');
  return codeToHtml(code, {
    lang: lang || 'text',
    theme: syntaxTheme,
  });
}
