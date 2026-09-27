import {
  ANALYZE_LIMITS,
  NOTE_TYPES,
  NOTE_TYPES_BY_TASK,
  chunkCount,
  estimateTokens,
  type NoteType,
  type Task,
} from '@pneumovision/shared';

import { cn } from '../../lib/cn.js';
import { readText } from '../../lib/files.js';
import { Button } from '../../components/ui.jsx';

export interface DraftNote {
  id: string;
  type: NoteType;
  text: string;
  /** Set when a .txt could not be loaded; blocks submit until fixed. */
  error: string | null;
}

let nextId = 0;
export const newNote = (type: NoteType = 'radiology'): DraftNote => ({
  id: `note-${++nextId}`,
  type,
  text: '',
  error: null,
});

export const noteExcluded = (task: Task, note: DraftNote) =>
  !NOTE_TYPES_BY_TASK[task].includes(note.type);

/** Notes that will actually reach the model. */
export const usableNotes = (task: Task, notes: DraftNote[]) =>
  notes.filter((n) => n.text.trim() !== '' && !noteExcluded(task, n));

export const noteTooLong = (note: DraftNote) =>
  new Blob([note.text]).size > ANALYZE_LIMITS.noteFileBytes;

const TYPE_LABEL: Record<NoteType, string> = {
  radiology: 'Radiology',
  progress: 'Progress',
  nursing: 'Nursing',
  discharge: 'Discharge',
};

export function NotesEditor({
  task,
  notes,
  onNotes,
}: {
  task: Task;
  notes: DraftNote[];
  onNotes: (notes: DraftNote[]) => void;
}) {
  const update = (id: string, patch: Partial<DraftNote>) =>
    onNotes(notes.map((n) => (n.id === id ? { ...n, ...patch } : n)));

  const loadFile = async (id: string, file: File) => {
    if (file.size > ANALYZE_LIMITS.noteFileBytes) {
      update(id, { error: `${file.name} is larger than 200 KB` });
      return;
    }
    update(id, { text: await readText(file), error: null });
  };

  return (
    <div className="space-y-3">
      {notes.length === 0 && (
        <p className="text-sm text-ink-secondary">No notes — the text branches will be missing.</p>
      )}

      {notes.map((note, i) => {
        const n = i + 1;
        const excluded = noteExcluded(task, note);
        const tokens = estimateTokens(note.text);
        const chunks = chunkCount(tokens);
        const tooLong = noteTooLong(note);
        return (
          <div
            key={note.id}
            className={cn(
              'rounded-lg border border-hairline p-3',
              excluded && 'border-dashed bg-surface-2 opacity-70',
            )}
          >
            <div className="flex flex-wrap items-center gap-2">
              <label className="text-xs text-ink-secondary">
                <span className="sr-only">{`Note ${n} type`}</span>
                <select
                  aria-label={`Note ${n} type`}
                  value={note.type}
                  onChange={(e) => update(note.id, { type: e.target.value as NoteType })}
                  className="rounded-md border border-hairline bg-surface-1 px-2 py-1 text-xs text-ink"
                >
                  {NOTE_TYPES.map((t) => (
                    <option key={t} value={t}>
                      {TYPE_LABEL[t]}
                    </option>
                  ))}
                </select>
              </label>
              <span className="text-[11px] text-ink-muted">
                → {note.type === 'discharge' ? 'DN (discharge) encoder' : 'RR encoder'}
              </span>
              <label className="ml-auto cursor-pointer text-xs text-ink-secondary hover:text-ink">
                Load .txt
                <input
                  aria-label={`Note ${n} file`}
                  type="file"
                  accept=".txt,text/plain"
                  className="sr-only"
                  onChange={(e) => {
                    const file = e.target.files?.[0];
                    e.target.value = '';
                    if (file) void loadFile(note.id, file);
                  }}
                />
              </label>
              <Button
                size="sm"
                variant="ghost"
                aria-label={`Remove note ${n}`}
                onClick={() => onNotes(notes.filter((x) => x.id !== note.id))}
              >
                ✕
              </Button>
            </div>

            {excluded && (
              <p className="mt-2 flex items-center gap-1 text-xs font-medium text-ink-secondary">
                <span aria-hidden="true">⊘</span>
                Excluded to prevent outcome leakage — discharge notes describe the outcome a
                mortality prediction is trying to forecast. It will not be stored or sent.
              </p>
            )}

            <textarea
              aria-label={`Note ${n} text`}
              value={note.text}
              disabled={excluded}
              onChange={(e) => update(note.id, { text: e.target.value, error: null })}
              rows={5}
              placeholder="Paste or type a synthetic clinical note…"
              className="mt-2 w-full rounded-md border border-hairline bg-surface-2 p-2 text-sm text-ink disabled:cursor-not-allowed"
            />
            <div className="mt-1 flex flex-wrap gap-3 text-[11px] text-ink-muted">
              <span className="tnum">
                ≈ {tokens} tokens · {chunks} chunk{chunks === 1 ? '' : 's'} of 512
              </span>
              {chunks > 1 && <span>Long note — the text encoder will split and pool it.</span>}
              {tooLong && (
                <span className="text-status-critical">✕ Longer than the 200 KB note limit</span>
              )}
              {note.error && <span className="text-status-critical">✕ {note.error}</span>}
            </div>
          </div>
        );
      })}

      <Button
        size="sm"
        onClick={() => onNotes([...notes, newNote()])}
        disabled={notes.length >= ANALYZE_LIMITS.noteFiles}
      >
        Add note
      </Button>
    </div>
  );
}
