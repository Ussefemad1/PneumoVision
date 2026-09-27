import type { Image, Prediction } from '@pneumovision/shared';
import { EHR_VARIABLES, EHR_WINDOW_HOURS } from '@pneumovision/shared';
import { useQuery } from '@tanstack/react-query';

import { getData } from '../../lib/api.js';
import { Card } from '../../components/ui.jsx';

const NOTE_LABEL: Record<string, string> = {
  radiology: 'Radiology',
  progress: 'Progress',
  nursing: 'Nursing',
  discharge: 'Discharge',
};

/**
 * "Inputs the model received": what the inference service reports it decoded
 * from the signed request — the image hash and size, EHR coverage, and per-note
 * token counts — plus anything F8 withheld. This is the evidence that an
 * upload actually reached the model, not the API's belief that it sent it.
 */
export function ReceivedInputs({ prediction }: { prediction: Prediction }) {
  const received = prediction.result?.received ?? null;

  // The thumbnail comes from the stay's images, matched by hash.
  const { data: images } = useQuery({
    queryKey: ['stay-images', prediction.stayId],
    queryFn: () => getData<Image[]>(`/stays/${prediction.stayId}/images`),
    enabled: Boolean(received?.cxr),
  });
  const image = received?.cxr ? images?.find((i) => i.sha256 === received.cxr!.sha256) : undefined;

  return (
    <Card
      title="Inputs the model received"
      subtitle="Reported by the inference service from the decoded request"
      actions={
        received?.mode === 'mock' ? (
          <span className="inline-flex items-center gap-1 rounded-full border border-status-warning/60 bg-status-warning/15 px-2 py-0.5 text-[11px] font-semibold text-ink">
            <span aria-hidden="true">◆</span> MOCK MODEL — not a clinical result
          </span>
        ) : null
      }
    >
      {!received ? (
        <p className="text-sm text-ink-secondary">
          Not recorded for this prediction (made before input receipts were stored).
        </p>
      ) : (
        <div className="grid gap-4 md:grid-cols-3">
          <section aria-label="Chest X-ray received">
            <h3 className="text-xs font-semibold uppercase tracking-wide text-ink-muted">
              Chest X-ray
            </h3>
            {received.cxr ? (
              <div className="mt-2 flex items-start gap-3">
                {image && (
                  <img
                    src={image.url}
                    alt="Chest X-ray the model received"
                    className="h-20 w-20 rounded border border-hairline bg-black object-contain"
                  />
                )}
                <dl className="text-xs">
                  <dt className="text-ink-muted">sha256</dt>
                  <dd>
                    <code title={received.cxr.sha256}>{received.cxr.sha256.slice(0, 12)}…</code>
                  </dd>
                  <dt className="mt-1 text-ink-muted">Dimensions</dt>
                  <dd>
                    {received.cxr.width}×{received.cxr.height}px
                  </dd>
                </dl>
              </div>
            ) : (
              <p className="mt-2 text-xs text-ink-muted">— none received</p>
            )}
          </section>

          <section aria-label="EHR received">
            <h3 className="text-xs font-semibold uppercase tracking-wide text-ink-muted">EHR</h3>
            {received.ehr ? (
              <p className="mt-2 text-xs text-ink">
                {received.ehr.hours} of {EHR_WINDOW_HOURS} hours charted ·{' '}
                {received.ehr.variablesPresent.length} of {EHR_VARIABLES.length} variables
                <span className="mt-1 block text-ink-muted">
                  {received.ehr.variablesPresent.join(', ')}
                </span>
              </p>
            ) : (
              <p className="mt-2 text-xs text-ink-muted">— none received</p>
            )}
          </section>

          <section aria-label="Notes received">
            <h3 className="text-xs font-semibold uppercase tracking-wide text-ink-muted">Notes</h3>
            {received.notes.length > 0 ? (
              <ul className="mt-2 space-y-1 text-xs">
                {received.notes.map((note, i) => (
                  <li key={note.id} className="text-ink">
                    {i + 1}. {NOTE_LABEL[note.type] ?? note.type} ·{' '}
                    <span className="tnum">
                      {note.tokens} tokens · {note.chunks} chunk{note.chunks === 1 ? '' : 's'}
                    </span>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="mt-2 text-xs text-ink-muted">— none received</p>
            )}
            {prediction.excludedInputs.length > 0 && (
              <div className="mt-2 text-xs text-ink-secondary">
                <span aria-hidden="true">⊘ </span>
                Excluded:{' '}
                {prediction.excludedInputs
                  .map((e) => `${NOTE_LABEL[e.type] ?? e.type} note`)
                  .join(', ')}{' '}
                — withheld to prevent outcome leakage
              </div>
            )}
          </section>
        </div>
      )}
    </Card>
  );
}
