import { ANALYZE_LIMITS, CXR_CONTENT_TYPES } from '@pneumovision/shared';
import { useEffect, useState } from 'react';

import { cn } from '../../lib/cn.js';
import { Button } from '../../components/ui.jsx';

/** Returns a user-facing problem with the file, or null if it may be sent. */
export function cxrProblem(file: File): string | null {
  if (!(CXR_CONTENT_TYPES as readonly string[]).includes(file.type)) {
    return 'Only PNG or JPEG images are accepted.';
  }
  if (file.size > ANALYZE_LIMITS.cxrBytes) return 'The image is larger than 20 MB.';
  return null;
}

export function CxrDrop({
  file,
  onFile,
}: {
  file: File | null;
  onFile: (file: File | null) => void;
}) {
  const [dragging, setDragging] = useState(false);
  const [preview, setPreview] = useState<string | null>(null);
  const problem = file ? cxrProblem(file) : null;

  useEffect(() => {
    if (!file || problem || typeof URL.createObjectURL !== 'function') {
      setPreview(null);
      return;
    }
    const url = URL.createObjectURL(file);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file, problem]);

  return (
    <div>
      <label
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          const dropped = e.dataTransfer.files[0];
          if (dropped) onFile(dropped);
        }}
        className={cn(
          'flex cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed p-4 text-center',
          dragging ? 'border-[var(--series-mortality)] bg-surface-2' : 'border-hairline',
        )}
      >
        <span className="text-sm text-ink">Upload X-ray</span>
        <input
          aria-label="Chest X-ray file"
          type="file"
          accept="image/png,image/jpeg"
          className="sr-only"
          onChange={(e) => {
            onFile(e.target.files?.[0] ?? null);
            e.target.value = '';
          }}
        />
      </label>

      {file && (
        <div className="mt-3 flex items-start gap-3">
          {preview && (
            <img
              src={preview}
              alt="Chest X-ray preview"
              className="h-24 w-24 rounded border border-hairline bg-black object-contain"
            />
          )}
          <div className="min-w-0 flex-1 text-xs">
            <div className="truncate font-medium text-ink">{file.name}</div>
            {problem && <p className="mt-1 text-status-critical">✕ {problem}</p>}
            <Button size="sm" variant="ghost" className="mt-1 -ml-2" onClick={() => onFile(null)}>
              Remove
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}
