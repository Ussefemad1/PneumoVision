/**
 * Reads a File/Blob as UTF-8 text. FileReader rather than `Blob.text()` so it
 * also runs under jsdom, which the component tests use.
 */
export function readText(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(typeof reader.result === 'string' ? reader.result : '');
    reader.onerror = () => reject(reader.error ?? new Error('could not read file'));
    reader.readAsText(blob);
  });
}
