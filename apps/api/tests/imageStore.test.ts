import { describe, expect, it } from 'vitest';

import { encodeGreyscalePng } from '../src/demo/syntheticXray.js';
import { resolveImagePath, sniffImage } from '../src/lib/imageStore.js';

/** Minimal JPEG header: SOI, an APP0 segment, then SOF0 with the dimensions. */
function jpegHeader(width: number, height: number): Buffer {
  const app0 = Buffer.from([0xff, 0xe0, 0x00, 0x04, 0x00, 0x00]);
  const sof0 = Buffer.alloc(19);
  sof0.set([0xff, 0xc0, 0x00, 0x11, 0x08]);
  sof0.writeUInt16BE(height, 5);
  sof0.writeUInt16BE(width, 7);
  return Buffer.concat([Buffer.from([0xff, 0xd8]), app0, sof0]);
}

describe('sniffImage', () => {
  it('reads PNG dimensions', () => {
    const png = encodeGreyscalePng(new Uint8Array(30 * 20), 30, 20);
    expect(sniffImage(png)).toEqual({ contentType: 'image/png', width: 30, height: 20 });
  });

  it('reads JPEG dimensions past leading segments', () => {
    expect(sniffImage(jpegHeader(640, 480))).toEqual({
      contentType: 'image/jpeg',
      width: 640,
      height: 480,
    });
  });

  it('rejects other bytes and truncated headers', () => {
    expect(sniffImage(Buffer.from('GIF89a......'))).toBeNull();
    expect(sniffImage(jpegHeader(640, 480).subarray(0, 10))).toBeNull();
  });
});

describe('resolveImagePath', () => {
  it('refuses paths that escape the images directory', () => {
    const env = { DEMO_DATA_DIR: '.demo-data' };
    expect(() => resolveImagePath(env, '../secret.png')).toThrow();
    expect(resolveImagePath(env, 'ok.png')).toMatch(/images[\\/]ok\.png$/);
  });
});
