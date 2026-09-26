import { createHash } from 'node:crypto';
import { deflateSync } from 'node:zlib';

/**
 * Generates synthetic chest-radiograph-like PNGs.
 *
 * **No MIMIC-CXR image ever enters this repo.** These are procedurally drawn
 * from a seed: a dark field, two brighter lung zones, a central mediastinum
 * band, a rib suggestion and film grain. They are obviously synthetic on
 * inspection, which is the point — they exist so the imaging tab and the
 * patch-confidence overlay have something to render.
 *
 * The PNG is written by hand (zlib + CRC32) rather than pulling in an image
 * library, since the only requirement is a valid 8-bit greyscale file.
 */

const CRC_TABLE = (() => {
  const table = new Int32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    table[n] = c;
  }
  return table;
})();

function crc32(buf: Buffer): number {
  let c = 0xffffffff;
  for (const byte of buf) c = CRC_TABLE[(c ^ byte) & 0xff]! ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

function chunk(type: string, data: Buffer): Buffer {
  const length = Buffer.alloc(4);
  length.writeUInt32BE(data.length, 0);
  const typeAndData = Buffer.concat([Buffer.from(type, 'ascii'), data]);
  const crc = Buffer.alloc(4);
  crc.writeUInt32BE(crc32(typeAndData), 0);
  return Buffer.concat([length, typeAndData, crc]);
}

/** Encodes an 8-bit greyscale raster as a PNG. */
export function encodeGreyscalePng(pixels: Uint8Array, width: number, height: number): Buffer {
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(width, 0);
  ihdr.writeUInt32BE(height, 4);
  ihdr.writeUInt8(8, 8); // bit depth
  ihdr.writeUInt8(0, 9); // colour type 0 = greyscale
  ihdr.writeUInt8(0, 10); // deflate
  ihdr.writeUInt8(0, 11); // adaptive filtering
  ihdr.writeUInt8(0, 12); // no interlace

  // Each scanline is prefixed with its filter type byte (0 = None).
  const raw = Buffer.alloc((width + 1) * height);
  for (let y = 0; y < height; y++) {
    raw[y * (width + 1)] = 0;
    for (let x = 0; x < width; x++) {
      raw[y * (width + 1) + 1 + x] = pixels[y * width + x]!;
    }
  }

  return Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    chunk('IHDR', ihdr),
    chunk('IDAT', deflateSync(raw, { level: 6 })),
    chunk('IEND', Buffer.alloc(0)),
  ]);
}

/** Small deterministic PRNG so the same seed always draws the same film. */
function mulberry32(seed: number) {
  return () => {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export interface XrayOptions {
  /** Anything stable — the stay id and image index work well. */
  seed: string;
  size?: number;
  /**
   * 0 = clear lung fields, 1 = dense opacity at the bases. Drives how
   * "abnormal" the film looks, so a deteriorating patient can be given a
   * visibly worse radiograph.
   */
  opacity?: number;
}

/**
 * Draws one synthetic film. The anatomy is a crude approximation — elliptical
 * lung fields either side of a bright mediastinal column, a lighter
 * diaphragm below, faint rib arcs, and grain.
 */
export function generateSyntheticXray(options: XrayOptions): {
  png: Buffer;
  width: number;
  height: number;
  sha256: string;
} {
  const size = options.size ?? 384;
  const opacity = Math.min(1, Math.max(0, options.opacity ?? 0));
  const seedNum = [...createHash('sha256').update(options.seed).digest().subarray(0, 4)].reduce(
    (a, b) => (a << 8) | b,
    0,
  );
  const rand = mulberry32(seedNum);

  const pixels = new Uint8Array(size * size);
  const cx = size / 2;
  // Slight per-film variation so the set does not look stamped.
  const lungOffset = size * (0.22 + rand() * 0.02);
  const lungRx = size * (0.16 + rand() * 0.015);
  const lungRy = size * (0.26 + rand() * 0.02);
  const lungCy = size * 0.45;

  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      // Vignetted soft-tissue background.
      const dx = (x - cx) / size;
      const dy = (y - size * 0.5) / size;
      let v = 46 + 26 * Math.exp(-(dx * dx + dy * dy) * 5);

      // Mediastinum / spine: a bright central column.
      const spine = Math.exp(-Math.pow((x - cx) / (size * 0.045), 2));
      v += 92 * spine;

      // Lung fields: darker (more radiolucent) than the surrounding tissue.
      for (const side of [-1, 1]) {
        const lx = cx + side * lungOffset;
        const nx = (x - lx) / lungRx;
        const ny = (y - lungCy) / lungRy;
        const d = nx * nx + ny * ny;
        if (d < 1) {
          const depth = Math.sqrt(1 - d);
          v -= 34 * depth;
          // Basal opacity: consolidation sits low in the lung field.
          const basal = Math.max(0, (y - lungCy) / lungRy);
          v += opacity * 74 * depth * basal * basal;
        }
      }

      // Diaphragm and abdomen below the lung fields.
      const belowDiaphragm = (y - size * 0.72) / (size * 0.1);
      if (belowDiaphragm > -1) v += 40 * Math.min(1, belowDiaphragm + 1);

      // Faint rib arcs.
      v += 7 * Math.sin((y / size) * 34 + Math.cos((x - cx) / size) * 2.4);

      // Film grain.
      v += (rand() - 0.5) * 9;

      pixels[y * size + x] = Math.max(0, Math.min(255, Math.round(v)));
    }
  }

  const png = encodeGreyscalePng(pixels, size, size);
  return { png, width: size, height: size, sha256: createHash('sha256').update(png).digest('hex') };
}
