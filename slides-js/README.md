# Slides Template

A slide deck template built with Next.js + MDX. Warm dark theme — customizable to any brand.

## Quick Start

```bash
make install
make serve
```

Opens at [http://localhost:3000](http://localhost:3000) — redirects straight to the template deck.

## How It Works

Each presentation is a single MDX file under `pages/decks/`. Slides are separated by `---`. Mix Markdown with React components for structured layouts.

```mdx
export const meta = {
  title: 'My Presentation',
}

<TitleSlide
  title="My Presentation"
  subtitle="Event Name"
  author="Your Name"
  date="April 2026"
/>

---

# Content Slide

- Point one
- Point two

---

<TitleSlide title="Thank You" subtitle="Questions?" />
```

## Adding a New Deck

1. Create `pages/decks/my-talk.mdx`
2. Register in `deckManifest` in `components/DeckLibrary.tsx`
3. Run `make check` to verify

## Components

| Component | Purpose | Key Props |
|-----------|---------|-----------|
| `TitleSlide` | Cover/closing slide with logo + gradient title | `title`, `subtitle`, `author`, `date` |
| `SectionSlide` | Big centered section divider | `children` |
| `Columns` | 2 or 3 column grid | `cols`, `gap`, `align` |
| `Embed` | Full-viewport iframe for demos | `src`, `title`, `clip` |
| `Diagram` | Theme-aware SVG wrapper | `children`, `title`, `width` |
| `SpeakerNotes` | Hidden notes (press **N**) | `children` |
| `Logo` | Brandmark + wordmark | `height`, `wordmark` |
| Code blocks | Syntax-highlighted via fenced Markdown | Language tag (e.g. ` ```python `) |

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `→` / `↓` / `Space` / `Enter` | Next slide |
| `←` / `↑` | Previous slide |
| `N` | Toggle speaker notes |
| `F` | Toggle fullscreen |

URL hash tracks current slide — `#3` = slide 3.

## Theme

Colors defined in `theme/colors.ts`:

| Token | Hex | Usage |
|-------|-----|-------|
| Brown | `#3b1e1c` | Dark background base |
| Ivory | `#f4f4eb` | Text |
| Periwinkle | `#80a0d8` | Primary / links |
| Red | `#ff4f2c` | Accent / highlights |
| Light Blue | `#a7d0dc` | Secondary |
| Green | `#cff29e` | Success / code strings |

Theme files: `theme/colors.ts`, `theme/fonts.ts`, `theme/syntax.ts`.

## Scripts

| Command | Description |
|---------|-------------|
| `make install` | Install dependencies |
| `make serve` | Dev server with hot reload |
| `make check` | Lint + fix + typecheck |
| `bun run build` | Static export to `out/` |

## Deployment

Pushes to `main` deploy to GitHub Pages via `.github/workflows/deploy.yml`. First-time: enable **Settings → Pages → Source → GitHub Actions**.

## Project Structure

```
slides-js/
├── pages/
│   ├── index.tsx               # Redirects to single deck
│   ├── decks/*.mdx             # Slide decks
│   ├── _app.tsx                # Theme + Deck wrapper
│   └── _document.tsx           # Font loading
├── components/                 # Slide components
├── theme/                      # Colors, fonts, syntax highlighting
├── styles/globals.css          # Reset + print styles
├── public/fonts/               # Self-hosted fonts
├── Makefile                    # install / serve / check
└── AGENTS.md                   # AI agent instructions
```

## Tooling

- **Runtime:** [Bun](https://bun.sh)
- **Lint/Format:** [Biome](https://biomejs.dev)
- **Types:** TypeScript strict mode

## License

MIT
