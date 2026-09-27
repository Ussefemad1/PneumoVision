import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { isAbsolute, join, resolve, sep } from 'node:path';

import type { CxrContentType } from '@pneumovision/shared';

import type { Env } from '../config/env.js';

/**
 * Local-disk storage for radiographs (demo and single-container deploys).
 *
 * TODO(phase-4): replaced by a private MinIO bucket. Callers go through these
 * functions, so that swap stays inside this file. Paths always come from the
 * database record, never from user input, and are re-checked to stay inside
 * the images directory.
 */

export function imagesDir(env: Pick<Env, 'DEMO_DATA_DIR'>): string {
  // `resolve`, not `join`: DEMO_DATA_DIR may be absolute (tests use a temp dir).
  return resolve(process.cwd(), env.DEMO_DATA_DIR, 'images');
}

/** Resolves a stored relative path, refusing anything that escapes the directory. */
export function resolveImagePath(env: Pick<Env, 'DEMO_DATA_DIR'>, filePath: string): string {
  const root = imagesDir(env);
  const resolved = resolve(root, filePath);
  if (isAbsolute(filePath) || !resolved.startsWith(root + sep)) {
    throw new Error('image path escapes the images directory');
  }
  return resolved;
}

export async function writeImage(
  env: Pick<Env, 'DEMO_DATA_DIR'>,
  fileName: string,
  bytes: Buffer,
): Promise<void> {
  await mkdir(imagesDir(env), { recursive: true });
  await writeFile(join(imagesDir(env), fileName), bytes);
}

export function sha256Hex(bytes: Buffer): string {
  return createHash('sha256').update(bytes).digest('hex');
}

export interface SniffedImage {
  contentType: CxrContentType;
  width: number;
  height: number;
}

const PNG_SIGNATURE = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);

/** JPEG start-of-frame markers that carry the image dimensions. */
const JPEG_SOF = new Set([
  0xc0, 0xc1, 0xc2, 0xc3, 0xc5, 0xc6, 0xc7, 0xc9, 0xca, 0xcb, 0xcd, 0xce, 0xcf,
]);

function sniffPng(buf: Buffer): SniffedImage | null {
  // Signature, then the IHDR chunk: length(4) "IHDR"(4) width(4) height(4) …
  if (buf.length < 33 || !buf.subarray(0, 8).equals(PNG_SIGNATURE)) return null;
  if (buf.toString('latin1', 12, 16) !== 'IHDR') return null;
  const width = buf.readUInt32BE(16);
  const height = buf.readUInt32BE(20);
  return width > 0 && height > 0 ? { contentType: 'image/png', width, height } : null;
}

function sniffJpeg(buf: Buffer): SniffedImage | null {
  if (buf.length < 4 || buf[0] !== 0xff || buf[1] !== 0xd8 || buf[2] !== 0xff) return null;
  let i = 2;
  while (i + 3 < buf.length) {
    if (buf[i] !== 0xff) return null;
    const marker = buf[i + 1]!;
    if (marker === 0xff) {
      i++; // fill byte
      continue;
    }
    // Standalone markers carry no length.
    if (marker === 0x01 || (marker >= 0xd0 && marker <= 0xd9)) {
      i += 2;
      continue;
    }
    const length = buf.readUInt16BE(i + 2);
    if (JPEG_SOF.has(marker)) {
      if (i + 9 > buf.length) return null;
      const height = buf.readUInt16BE(i + 5);
      const width = buf.readUInt16BE(i + 7);
      return width > 0 && height > 0 ? { contentType: 'image/jpeg', width, height } : null;
    }
    i += 2 + length;
  }
  return null;
}

/**
 * Identifies a PNG or JPEG by its bytes and reads its dimensions. Returns null
 * for anything else — including a truncated file with valid magic bytes.
 */
export function sniffImage(buf: Buffer): SniffedImage | null {
  return sniffPng(buf) ?? sniffJpeg(buf);
}

export interface StoredImageBytes {
  bytes: Buffer;
  contentType: CxrContentType;
  sha256: string;
}

/**
 * Reads a stored radiograph and verifies it against the hash recorded at
 * upload, so a prediction can never silently run on a different file.
 */
export async function readStoredImage(
  env: Pick<Env, 'DEMO_DATA_DIR'>,
  image: { filePath: string; sha256: string },
): Promise<StoredImageBytes> {
  const bytes = await readFile(resolveImagePath(env, image.filePath));
  const sha256 = sha256Hex(bytes);
  if (sha256 !== image.sha256) {
    throw new Error('stored image does not match its recorded sha256');
  }
  const sniffed = sniffImage(bytes);
  if (!sniffed) throw new Error('stored image is not a readable PNG or JPEG');
  return { bytes, contentType: sniffed.contentType, sha256 };
}
