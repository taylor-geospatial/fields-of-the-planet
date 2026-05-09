# Slides Template — AI Instructions

## Project Overview

**Next.js + MDX slide deck system** with a warm dark theme (brown/ivory/periwinkle/red).

- Index page redirects to the single template deck
- Deck presentations are MDX files under `pages/decks/`
- Deck manifest in `components/DeckLibrary.tsx` registers each deck
- Styled with **Theme-UI** (`sx` prop)
- Static export via `output: 'export'` in `next.config.mjs`

## Commands

```bash
make install    # bun install
make serve      # bun run dev
make check      # lint:fix + typecheck
bun run build   # static export to out/
```

## How Slides Work

Single MDX file per deck. Slides separated by `---` (horizontal rule). The `Deck` component splits on `<hr data-slide-separator>` at runtime.

Every deck must export metadata:

```mdx
export const meta = {
  title: 'Presentation Title',
}
```

Navigation: arrow keys, Space, Enter (next), N (notes), F (fullscreen). URL hash `#3` = slide 3.

## Brand Colors

From `theme/colors.ts`:

| Token | Hex | Role |
|-------|-----|------|
| Brown | `#3b1e1c` | Background base |
| Ivory | `#f4f4eb` | Text |
| Periwinkle | `#80a0d8` | Primary |
| Red | `#ff4f2c` | Accent |
| Light Blue | `#a7d0dc` | Secondary |
| Green | `#cff29e` | Success / strings |

Gradient: `linear-gradient(261deg, #ff4f2c 0%, #80a0d8 100%)`

## Components

### TitleSlide

Cover/closing slide with logo + gradient title.

| Prop | Type | Default |
|------|------|---------|
| `title` | `string` | required |
| `subtitle` | `string` | — |
| `author` | `string` | — |
| `date` | `string` | — |
| `showLogo` | `boolean` | `true` |

### SectionSlide

Big centered text for section dividers. Use `##` headings. Blank lines required after opening and before closing tag. Background images render **full-bleed** via `position: fixed` covering the entire viewport with a 50% dark scrim. The `Slide` component intentionally omits `overflow: hidden` to allow this — do not re-add it.

### Columns

Multi-column grid. Each child must be wrapped in `<div>` with blank lines for MDX processing. Thin vertical separator line rendered between columns automatically.

| Prop | Type | Default |
|------|------|---------|
| `cols` | `2 \| 3` | `2` |
| `gap` | `number` | `6` |
| `align` | `'start' \| 'center' \| 'end' \| 'stretch'` | `'start'` |

### Embed

Full-viewport iframe. Must be the **only content** on its slide.

| Prop | Type | Default |
|------|------|---------|
| `src` | `string` | required |
| `title` | `string` | `"Embedded content"` |
| `clip` | `boolean` | `false` |

### Code Blocks

Standard fenced Markdown code blocks — `CodeBlock` is applied automatically. Syntax theme uses brand colors. Blocks have a gradient accent stripe (periwinkle→red) on the left edge.

### SpeakerNotes

Hidden presenter notes. Place at the end of a slide before `---`.

### Logo

Brandmark SVG + wordmark. Included automatically in `TitleSlide`. Update `components/Logo.tsx` to swap in your own brand mark.

### Diagram

SVG wrapper with CSS custom properties:
- `--diagram-primary` → `#80a0d8`
- `--diagram-accent` → `#ff4f2c`
- `--diagram-blue` → `#a7d0dc`
- `--diagram-text` → `#f4f4eb`
- `--diagram-muted` → `#8a8a7e`
- `--diagram-border` → `#3d2a28`
- `--diagram-surface` → `#261816`

## Component Decision Tree

1. First/last slide → `<TitleSlide>`
2. Section divider / big statement → `<SectionSlide>`
3. Live demo / map → `<Embed>` (own slide, add lead-in)
4. Comparing items → `<Columns cols={2}>`
5. Three parallel items → `<Columns cols={3}>`
6. Code + explanation → `<Columns>` with text + code
7. Architecture diagram → `<Diagram>` with inline SVG
8. Just text → Plain Markdown
9. Presenter reminders → `<SpeakerNotes>` at end of slide

## Scaffold Workflow

1. Create `pages/decks/<slug>.mdx` with `export const meta`
2. Start with `<TitleSlide>`, end with closing `<TitleSlide>`
3. Use `<SectionSlide>` between major topics
4. Add `<SpeakerNotes>` on key slides
5. Register in `deckManifest` in `components/DeckLibrary.tsx`
6. Verify: `bun run build`

## Iterate Workflow

1. Read the MDX file
2. Count `---` to locate target slide
3. Make changes
4. Run `make check`

## Visual Design Notes

- **Slide backgrounds** have subtle radial gradients (periwinkle + red tints) and SVG noise texture for depth — see `Slide.tsx`
- **Nav bar** is glassmorphic (translucent + backdrop blur) with a gradient progress bar across top. Light/dark mode aware — see `SlideNav.tsx`
- **Lists** use custom markers: `ul` gets accent-colored dashes, `ol` gets numbered pills — see `theme/index.ts`
- **Blockquotes** render as callout containers (surface bg + 3px accent left border)
- **Tables** have accent-colored header underlines and alternating row stripes
- **Headings** are responsive: `h1` = `[7,8]` (64/88px), `h2` = `[6,7]` (48/64px) with tight letter-spacing
- Do NOT add `overflow: hidden` to `Slide.tsx` — it breaks `SectionSlide` full-bleed backgrounds

## Key Files

| File | Purpose |
|------|---------|
| `theme/colors.ts` | Brand color palette |
| `theme/fonts.ts` | Font families + sizes |
| `theme/index.ts` | Theme-UI config (lists, tables, blockquotes, headings) |
| `theme/syntax.ts` | Code syntax highlighting |
| `components/Logo.tsx` | Brandmark + wordmark |
| `components/Slide.tsx` | Slide wrapper with bg gradients + noise |
| `components/SlideNav.tsx` | Glassmorphic nav bar + progress bar |
| `components/TitleSlide.tsx` | Cover slide with gradient |
| `components/SectionSlide.tsx` | Full-bleed section dividers |
| `components/CodeBlock.tsx` | Syntax highlighting + accent stripe |
| `components/Columns.tsx` | Multi-column grid with separators |
| `components/DeckLibrary.tsx` | Deck manifest + library UI |
| `components/Deck.tsx` | Slide splitting + navigation engine |
| `pages/index.tsx` | Redirect to single deck |
| `mdx-components.tsx` | MDX component registry |
