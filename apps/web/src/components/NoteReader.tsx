import { useMemo } from 'react';

import type { NoteSpan } from '@pneumovision/shared';

import { percent } from '../lib/format.js';
import { sequentialBlue } from './charts.jsx';

/**
 * F2 (notes): high/low-confidence span highlighting.
 *
 * Spans above θ get a tinted background *and* an underline; spans below get
 * neither. Confidence is a magnitude, so the tint uses the sequential ramp —
 * and because the tint alone would be invisible to some readers, the
 * underline carries the same information, with the value in the title.
 */
export function NoteReader({
  text,
  spans,
  theta,
}: {
  text: string;
  spans: NoteSpan[];
  theta: number;
}) {
  const segments = useMemo(() => {
    if (spans.length === 0) return [{ text, confidence: null as number | null }];

    // Spans arrive sorted per note, but never assume it; clamp to the text
    // length so a stale span from an edited note cannot slice out of range.
    const ordered = [...spans]
      .map((s) => ({
        start: Math.max(0, Math.min(text.length, s.start)),
        end: Math.max(0, Math.min(text.length, s.end)),
        confidence: s.confidence,
      }))
      .filter((s) => s.end > s.start)
      .sort((a, b) => a.start - b.start);

    const out: { text: string; confidence: number | null }[] = [];
    let cursor = 0;
    for (const span of ordered) {
      if (span.start > cursor) {
        out.push({ text: text.slice(cursor, span.start), confidence: null });
      }
      out.push({ text: text.slice(span.start, span.end), confidence: span.confidence });
      cursor = Math.max(cursor, span.end);
    }
    if (cursor < text.length) out.push({ text: text.slice(cursor), confidence: null });
    return out;
  }, [text, spans]);

  return (
    <div className="whitespace-pre-wrap break-words font-mono text-[13px] leading-relaxed text-ink-secondary">
      {segments.map((segment, i) => {
        if (segment.confidence === null) return <span key={i}>{segment.text}</span>;

        const above = segment.confidence >= theta;
        return (
          <mark
            key={i}
            title={`Token confidence ${percent(segment.confidence)} (${above ? 'above' : 'below'} θ ${theta.toFixed(2)})`}
            className="rounded-sm bg-transparent text-ink"
            style={
              above
                ? {
                    backgroundColor: sequentialBlue(segment.confidence),
                    color: segment.confidence > 0.6 ? '#ffffff' : 'inherit',
                    textDecoration: 'underline',
                    textDecorationStyle: 'solid',
                    textDecorationThickness: '2px',
                    textUnderlineOffset: '2px',
                  }
                : {
                    textDecoration: 'underline',
                    textDecorationStyle: 'dotted',
                    textDecorationColor: 'var(--axis)',
                    textUnderlineOffset: '2px',
                  }
            }
          >
            {segment.text}
          </mark>
        );
      })}
    </div>
  );
}
