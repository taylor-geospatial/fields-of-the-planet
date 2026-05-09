/** @jsxImportSource theme-ui */
import type { ReactNode } from 'react';

export interface ColumnsProps {
  /** Number of columns. Default: 2 */
  cols?: 2 | 3;
  /** Custom grid-template-columns value (overrides `cols`). E.g. "1fr 2fr" */
  template?: string;
  /** Gap between columns in theme space units. Default: 5 (32px) */
  gap?: number;
  /** Column content — each direct child becomes one column */
  children: ReactNode;
  /** Vertical alignment. Default: 'start' */
  align?: 'start' | 'center' | 'end' | 'stretch';
}

/**
 * Multi-column grid layout component.
 * Each direct child becomes one column.
 * In MDX, wrap each column's content in a <div>.
 */
export function Columns({ cols = 2, template, gap = 6, children, align = 'start' }: ColumnsProps) {
  return (
    <div
      sx={{
        display: 'grid',
        gridTemplateColumns: ['1fr', template ?? `repeat(${cols}, 1fr)`],
        gap: [3, gap],
        alignItems: align,
        width: '100%',
        '& > *': {
          position: 'relative',
        },
      }}
    >
      {children}
    </div>
  );
}
