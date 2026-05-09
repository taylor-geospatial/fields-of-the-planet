export interface EmbedProps {
  /** URL to embed in the iframe */
  src: string;
  /** Title for accessibility */
  title?: string;
  /**
   * Whether to clip the bottom of the iframe to hide the SlideNav bar.
   * Useful when the embedded page has its own navigation that conflicts.
   * Default: false
   */
  clip?: boolean;
}

/**
 * Full-viewport iframe embed component.
 * Use for live demos, maps, and interactive visualizations.
 * Uses position absolute to break out of the Slide padding.
 */
export function Embed({ src, title = 'Embedded content', clip = false }: EmbedProps) {
  return (
    <div
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        width: '100vw',
        height: clip ? 'calc(100vh - 48px)' : '100vh',
        zIndex: 999,
        overflow: 'hidden',
      }}
    >
      <iframe
        src={src}
        title={title}
        style={{
          width: '100%',
          height: '100%',
          border: 'none',
          display: 'block',
        }}
        sandbox="allow-scripts allow-same-origin allow-popups allow-forms"
        loading="lazy"
      />
      {/* Fallback link if iframe fails to load */}
      <noscript>
        <a href={src} target="_blank" rel="noopener noreferrer">
          Open {title} in a new tab
        </a>
      </noscript>
    </div>
  );
}
