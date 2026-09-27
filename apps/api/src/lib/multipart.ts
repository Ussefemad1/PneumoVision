import busboy from 'busboy';
import type { Request } from 'express';

import { ApiError } from '../middleware/errorHandler.js';

/**
 * Buffered multipart/form-data parsing on top of busboy, with per-field-name
 * file rules.
 *
 * Every limit maps to a 413 with a specific code, so the client can tell the
 * user exactly which input was too big. On a violation the rest of the body is
 * still drained (bounded by the overall size cap) before rejecting, so the
 * client always receives the error response instead of a reset connection.
 */

export interface FileRule {
  maxBytes: number;
  maxCount: number;
  /** 413 code when one file exceeds `maxBytes`. */
  tooLargeCode: string;
  /** 413 code when more than `maxCount` files arrive under this name. */
  tooManyCode: string;
}

export interface MultipartSpec {
  files: Record<string, FileRule>;
  /** Longest accepted value for a plain field. */
  fieldBytes: number;
  maxFields: number;
  /** Hard cap on the whole body, checked against Content-Length up front. */
  totalBytes: number;
}

export interface MultipartFile {
  fieldName: string;
  fileName: string;
  mimeType: string;
  buffer: Buffer;
}

export interface MultipartBody {
  fields: Record<string, string>;
  files: MultipartFile[];
}

const tooLarge = (code: string, message: string) => new ApiError(413, code, message);

export function readMultipart(req: Request, spec: MultipartSpec): Promise<MultipartBody> {
  const contentType = req.headers['content-type'] ?? '';
  if (!/^multipart\/form-data/i.test(contentType)) {
    return Promise.reject(ApiError.badRequest('INVALID_CONTENT_TYPE', 'Use multipart/form-data'));
  }
  const declared = Number(req.headers['content-length']);
  if (Number.isFinite(declared) && declared > spec.totalBytes) {
    return Promise.reject(tooLarge('PAYLOAD_TOO_LARGE', 'The upload exceeds the total size limit'));
  }

  const rules = Object.values(spec.files);
  let parser: busboy.Busboy;
  try {
    parser = busboy({
      headers: req.headers,
      limits: {
        // Per-name limits are enforced below; these are the outer bounds.
        fileSize: Math.max(...rules.map((r) => r.maxBytes)) + 1,
        files: rules.reduce((n, r) => n + r.maxCount, 0) + 1,
        fields: spec.maxFields,
        fieldSize: spec.fieldBytes,
      },
    });
  } catch {
    return Promise.reject(
      ApiError.badRequest('INVALID_CONTENT_TYPE', 'multipart/form-data needs a boundary'),
    );
  }

  return new Promise((resolve, reject) => {
    const fields: Record<string, string> = {};
    const pending: { file: Omit<MultipartFile, 'buffer'>; chunks: Buffer[] }[] = [];
    const counts: Record<string, number> = {};
    let received = 0;
    let failure: ApiError | null = null;
    const fail = (err: ApiError) => {
      failure ??= err;
    };

    // Chunked bodies carry no Content-Length; count as we go.
    req.on('data', (chunk: Buffer) => {
      received += chunk.length;
      if (received > spec.totalBytes) {
        fail(tooLarge('PAYLOAD_TOO_LARGE', 'The upload exceeds the total size limit'));
        req.unpipe(parser);
        req.resume();
        reject(failure!);
      }
    });

    parser.on('file', (name, stream, info) => {
      const rule = spec.files[name];
      if (!rule) {
        fail(ApiError.badRequest('UNEXPECTED_FILE', `Unexpected file field "${name}"`));
        stream.resume();
        return;
      }
      counts[name] = (counts[name] ?? 0) + 1;
      if (counts[name] > rule.maxCount) {
        fail(tooLarge(rule.tooManyCode, `At most ${rule.maxCount} "${name}" file(s) are accepted`));
        stream.resume();
        return;
      }

      const entry = {
        file: { fieldName: name, fileName: info.filename, mimeType: info.mimeType },
        chunks: [] as Buffer[],
      };
      pending.push(entry);
      let size = 0;
      const overLimit = () => {
        entry.chunks = [];
        fail(
          tooLarge(
            rule.tooLargeCode,
            `"${name}" file exceeds ${Math.floor(rule.maxBytes / 1024)} KB`,
          ),
        );
      };
      stream.on('data', (chunk: Buffer) => {
        size += chunk.length;
        if (size > rule.maxBytes) overLimit();
        else if (!failure) entry.chunks.push(chunk);
      });
      stream.on('limit', overLimit);
    });

    parser.on('field', (name, value, info) => {
      if (info.valueTruncated) {
        fail(tooLarge('FIELD_TOO_LARGE', `Field "${name}" exceeds the size limit`));
        return;
      }
      fields[name] = value;
    });

    parser.on('filesLimit', () => fail(tooLarge('TOO_MANY_FILES', 'Too many files')));
    parser.on('fieldsLimit', () => fail(tooLarge('TOO_MANY_FIELDS', 'Too many fields')));
    parser.on('partsLimit', () => fail(tooLarge('TOO_MANY_PARTS', 'Too many parts')));

    parser.on('error', () => {
      reject(failure ?? ApiError.badRequest('INVALID_MULTIPART', 'Malformed multipart body'));
    });

    parser.on('close', () => {
      if (failure) {
        reject(failure);
        return;
      }
      resolve({
        fields,
        files: pending.map(({ file, chunks }) => ({ ...file, buffer: Buffer.concat(chunks) })),
      });
    });

    req.pipe(parser);
  });
}
