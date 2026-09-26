import { mkdir, writeFile } from 'node:fs/promises';
import { join } from 'node:path';

import type { NoteType } from '@pneumovision/shared';

import {
  AlertModel,
  ImageModel,
  NoteModel,
  PatientModel,
  PredictionModel,
  StayModel,
  UserModel,
  VitalsModel,
} from '../db/models.js';
import type { Env } from '../config/env.js';
import { hashPassword } from '../lib/password.js';
import type { Logger } from '../lib/logger.js';
import { generateSyntheticXray } from './syntheticXray.js';

/**
 * Synthetic demo data.
 *
 * Everything here is fabricated. No MIMIC record, image or note text is used
 * or reproduced — the vitals come from a physiological random walk, the
 * radiographs are drawn procedurally, and the note text is assembled from
 * templates. The *shapes* match the real schema so the demo exercises the
 * same code paths the credentialed data eventually will.
 */

/** Deterministic PRNG so every run of the demo tells the same story. */
function mulberry32(seed: number) {
  return () => {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export interface StayPlan {
  index: number;
  ward: string;
  age: number;
  sex: 'M' | 'F';
  /** 'stable' | 'deteriorating' | 'recovering' — shapes the vitals walk. */
  course: 'stable' | 'deteriorating' | 'recovering';
  hasCxr: boolean;
  hasNotes: boolean;
  /** Hours of history to generate. */
  hours: number;
}

/**
 * Twelve stays with deliberately varied modality availability, so the F5
 * badges and the reduced-input banner are visible in the demo rather than
 * theoretical. Four deteriorate, which is what makes the alerting and the
 * risk trajectory worth looking at.
 */
const PLANS: StayPlan[] = [
  {
    index: 1,
    ward: 'MICU',
    age: 74,
    sex: 'M',
    course: 'deteriorating',
    hasCxr: true,
    hasNotes: true,
    hours: 60,
  },
  {
    index: 2,
    ward: 'MICU',
    age: 68,
    sex: 'F',
    course: 'deteriorating',
    hasCxr: true,
    hasNotes: true,
    hours: 72,
  },
  {
    index: 3,
    ward: 'SICU',
    age: 81,
    sex: 'F',
    course: 'deteriorating',
    hasCxr: true,
    hasNotes: false,
    hours: 54,
  },
  {
    index: 4,
    ward: 'CCU',
    age: 59,
    sex: 'M',
    course: 'deteriorating',
    hasCxr: false,
    hasNotes: true,
    hours: 66,
  },
  {
    index: 5,
    ward: 'MICU',
    age: 47,
    sex: 'F',
    course: 'stable',
    hasCxr: true,
    hasNotes: true,
    hours: 50,
  },
  {
    index: 6,
    ward: 'MICU',
    age: 63,
    sex: 'M',
    course: 'stable',
    hasCxr: false,
    hasNotes: true,
    hours: 58,
  },
  {
    index: 7,
    ward: 'SICU',
    age: 55,
    sex: 'M',
    course: 'recovering',
    hasCxr: true,
    hasNotes: true,
    hours: 70,
  },
  {
    index: 8,
    ward: 'SICU',
    age: 39,
    sex: 'F',
    course: 'stable',
    hasCxr: true,
    hasNotes: false,
    hours: 48,
  },
  {
    index: 9,
    ward: 'CCU',
    age: 77,
    sex: 'M',
    course: 'recovering',
    hasCxr: true,
    hasNotes: true,
    hours: 64,
  },
  {
    index: 10,
    ward: 'CCU',
    age: 70,
    sex: 'F',
    course: 'stable',
    hasCxr: false,
    hasNotes: false,
    hours: 52,
  },
  {
    index: 11,
    ward: 'MICU',
    age: 84,
    sex: 'F',
    course: 'stable',
    hasCxr: true,
    hasNotes: true,
    hours: 56,
  },
  {
    index: 12,
    ward: 'SICU',
    age: 51,
    sex: 'M',
    course: 'recovering',
    hasCxr: false,
    hasNotes: true,
    hours: 62,
  },
];

const GCS_EYE = ['4 Spontaneously', '3 To speech', '2 To pain', '1 No Response'];
const GCS_MOTOR = ['6 Obeys Commands', '5 Localizes Pain', '4 Flex-withdraws', '3 Abnorm flexion'];
const GCS_VERBAL = ['5 Oriented', '4 Confused', '3 Inapprop words', '2 Incomp sounds'];

/**
 * A physiological random walk. A deteriorating course drives heart and
 * respiratory rate up while oxygen saturation and blood pressure fall, which
 * is what makes the risk trajectory and the alert rules fire believably.
 */
function vitalsForHour(
  plan: StayPlan,
  hour: number,
  rand: () => number,
): Record<string, number | string | null> {
  const t = hour / plan.hours;
  const severity =
    plan.course === 'deteriorating'
      ? Math.pow(t, 1.6)
      : plan.course === 'recovering'
        ? Math.max(0, 0.55 - t * 0.5)
        : 0.12 + Math.sin(t * 6) * 0.04;

  const noise = (scale: number) => (rand() - 0.5) * scale;

  const hr = 78 + severity * 48 + noise(6);
  const rr = 16 + severity * 16 + noise(3);
  const spo2 = 98 - severity * 14 - Math.max(0, noise(2));
  const sbp = 124 - severity * 30 + noise(8);
  const dbp = 70 - severity * 16 + noise(5);
  const map = (sbp + 2 * dbp) / 3;
  const temp = 36.8 + severity * 1.6 + noise(0.3);
  const fio2 = Math.min(1, 0.21 + severity * 0.55);
  const ph = 7.42 - severity * 0.12 + noise(0.02);
  const glucose = 118 + severity * 55 + noise(20);

  // GCS worsens in steps, not continuously.
  const gcsBand = Math.min(3, Math.floor(severity * 3.4));
  const gcsTotal = String(Math.max(3, 15 - gcsBand * 3 - (rand() < 0.3 ? 1 : 0)));

  const round = (v: number, dp = 0) => Number(v.toFixed(dp));

  // Not everything is charted every hour — gaps are normal and the UI must
  // show them as gaps rather than interpolating.
  const charted = (p: number) => rand() < p;

  return {
    'Heart Rate': charted(0.95) ? round(hr) : null,
    'Respiratory rate': charted(0.92) ? round(rr) : null,
    'Oxygen saturation': charted(0.94) ? round(spo2) : null,
    'Systolic blood pressure': charted(0.85) ? round(sbp) : null,
    'Diastolic blood pressure': charted(0.85) ? round(dbp) : null,
    'Mean blood pressure': charted(0.8) ? round(map) : null,
    Temperature: charted(0.5) ? round(temp, 1) : null,
    'Fraction inspired oxygen': charted(0.45) ? round(fio2, 2) : null,
    pH: charted(0.25) ? round(ph, 2) : null,
    Glucose: charted(0.3) ? round(glucose) : null,
    Weight: hour === 0 ? round(70 + rand() * 25, 1) : null,
    Height: hour === 0 ? round(160 + rand() * 25, 1) : null,
    'Glascow coma scale total': charted(0.4) ? gcsTotal : null,
    'Glascow coma scale eye opening': charted(0.35) ? GCS_EYE[gcsBand]! : null,
    'Glascow coma scale motor response': charted(0.35) ? GCS_MOTOR[gcsBand]! : null,
    'Glascow coma scale verbal response': charted(0.35) ? GCS_VERBAL[gcsBand]! : null,
    'Capillary refill rate': charted(0.15) ? (severity > 0.6 ? '1.0' : '0.0') : null,
  };
}

/** Template-assembled note text. Nothing here is derived from a real report. */
function noteText(plan: StayPlan, type: NoteType, hour: number): string {
  const t = hour / plan.hours;
  const worsening = plan.course === 'deteriorating' && t > 0.4;

  if (type === 'radiology') {
    return [
      'SYNTHETIC RADIOLOGY REPORT — generated for demonstration. Not a real study.',
      '',
      'EXAMINATION: Chest radiograph, single AP portable view.',
      '',
      'COMPARISON: Prior portable study from the preceding day.',
      '',
      'FINDINGS:',
      worsening
        ? 'Interval increase in bibasilar airspace opacification, more confluent at the right base. Findings are compatible with progressive multifocal consolidation. Small layering right pleural effusion. Cardiomediastinal silhouette remains within normal limits for a portable technique. No pneumothorax.'
        : 'Lung volumes are adequate. No focal consolidation, pleural effusion or pneumothorax is identified. The cardiomediastinal silhouette is unremarkable. Support lines and tubes, where present, are in satisfactory position.',
      '',
      'IMPRESSION:',
      worsening
        ? '1. Progressive bibasilar consolidation, right greater than left, concerning for evolving pneumonia. 2. Small right pleural effusion.'
        : '1. No acute cardiopulmonary process.',
    ].join('\n');
  }

  if (type === 'nursing') {
    return [
      'SYNTHETIC NURSING NOTE — generated for demonstration.',
      '',
      worsening
        ? 'Patient increasingly tachypnoeic over the shift with rising oxygen requirement. Saturations dipping on current settings; FiO2 titrated upward. Appears fatigued, using accessory muscles intermittently. Medical team notified and attended.'
        : 'Patient settled overnight, tolerating current respiratory support without distress. Observations stable and within the parameters set. Repositioned two-hourly, pressure areas intact. No concerns raised this shift.',
    ].join('\n');
  }

  if (type === 'discharge') {
    // Present in the dataset specifically so the F8 filter has something to
    // exclude from mortality predictions.
    return [
      'SYNTHETIC DISCHARGE SUMMARY — generated for demonstration.',
      '',
      'This note exists to exercise the leakage-control rule: discharge notes',
      'describe the outcome and are never sent for a mortality prediction.',
      '',
      'HOSPITAL COURSE: The patient completed a course of treatment for the',
      'presenting respiratory illness and was stepped down from intensive care.',
    ].join('\n');
  }

  return [
    'SYNTHETIC PROGRESS NOTE — generated for demonstration.',
    '',
    worsening
      ? 'ASSESSMENT: Ongoing respiratory deterioration. Increasing work of breathing and oxygen requirement over the past 12 hours, with radiographic progression. Differential remains hospital-acquired pneumonia versus fluid overload, favouring the former given the fever trend.'
      : 'ASSESSMENT: Clinically stable on current support. Gas exchange adequate, haemodynamics unremarkable, no new organ dysfunction.',
    '',
    worsening
      ? 'PLAN: Escalate respiratory support, broaden antimicrobial cover pending cultures, repeat imaging in the morning, and discuss with the critical care consultant.'
      : 'PLAN: Continue current management, daily review, wean support as tolerated.',
  ].join('\n');
}

/** Rough token estimate, enough for the chunking warning (F9). */
function estimateTokens(text: string): number {
  return Math.ceil(text.length / 4);
}

export interface SeedResult {
  users: number;
  patients: number;
  stays: number;
  vitalsPoints: number;
  images: number;
  notes: number;
}

export async function seedDemoData(env: Env, logger: Logger): Promise<SeedResult> {
  const existing = await StayModel.estimatedDocumentCount();
  if (existing > 0) {
    logger.info({ stays: existing }, 'demo data already present, skipping seed');
    return { users: 0, patients: 0, stays: existing, vitalsPoints: 0, images: 0, notes: 0 };
  }

  const [adminHash, clinicianHash] = await Promise.all([
    hashPassword(env.SEED_ADMIN_PASSWORD),
    hashPassword(env.SEED_CLINICIAN_PASSWORD),
  ]);

  const [, clinician] = await Promise.all([
    UserModel.create({
      email: env.SEED_ADMIN_EMAIL,
      passwordHash: adminHash,
      name: 'Demo Administrator',
      role: 'admin',
    }),
    UserModel.create({
      email: env.SEED_CLINICIAN_EMAIL,
      passwordHash: clinicianHash,
      name: 'Dr Demo Clinician',
      role: 'clinician',
    }),
  ]);

  const imagesDir = join(process.cwd(), env.DEMO_DATA_DIR, 'images');
  await mkdir(imagesDir, { recursive: true });

  const now = new Date();
  // Align to the top of the hour so the 48-hour window lines up cleanly.
  now.setMinutes(0, 0, 0);

  const counts: SeedResult = {
    users: 2,
    patients: 0,
    stays: 0,
    vitalsPoints: 0,
    images: 0,
    notes: 0,
  };

  for (const plan of PLANS) {
    const rand = mulberry32(plan.index * 7919);
    const admittedAt = new Date(now.getTime() - plan.hours * 3_600_000);

    const patient = await PatientModel.create({
      pseudoId: `PV-${String(plan.index).padStart(6, '0')}`,
      demographics: { age: plan.age, sex: plan.sex },
      createdBy: clinician._id,
    });
    counts.patients++;

    const stay = await StayModel.create({
      patientId: patient._id,
      ward: plan.ward,
      bedLabel: `${plan.ward[0]}-${String(plan.index).padStart(2, '0')}`,
      admittedAt,
      status: 'active',
      availability: { ehr: true, cxr: plan.hasCxr, notes: plan.hasNotes },
    });
    counts.stays++;

    // ── Vitals: one document per hour, through the model so validation and
    //    the time-series collection are exercised exactly as in production.
    const vitalsDocs = [];
    for (let hour = 0; hour < plan.hours; hour++) {
      vitalsDocs.push({
        ts: new Date(admittedAt.getTime() + hour * 3_600_000),
        meta: { stayId: stay._id, source: 'seed' },
        values: vitalsForHour(plan, hour, rand),
      });
    }
    // insertMany still runs schema validation and is far faster than one
    // save per hour.
    await VitalsModel.insertMany(vitalsDocs, { ordered: true });
    counts.vitalsPoints += vitalsDocs.length;

    // ── Images
    if (plan.hasCxr) {
      const shots = plan.course === 'deteriorating' ? 3 : 2;
      for (let i = 0; i < shots; i++) {
        const progress = i / (shots - 1);
        const opacity =
          plan.course === 'deteriorating'
            ? 0.15 + progress * 0.75
            : plan.course === 'recovering'
              ? 0.6 - progress * 0.45
              : 0.1 + progress * 0.05;

        const xray = generateSyntheticXray({ seed: `${plan.index}-${i}`, opacity });
        const fileName = `${String(stay._id)}-${i}.png`;
        await writeFile(join(imagesDir, fileName), xray.png);

        await ImageModel.create({
          stayId: stay._id,
          filePath: fileName,
          takenAt: new Date(
            admittedAt.getTime() + Math.floor((plan.hours * (i + 1)) / (shots + 1)) * 3_600_000,
          ),
          view: i % 2 === 0 ? 'AP' : 'PA',
          width: xray.width,
          height: xray.height,
          sha256: xray.sha256,
          uploadedBy: clinician._id,
        });
        counts.images++;
      }
    }

    // ── Notes
    if (plan.hasNotes) {
      const schedule: { type: NoteType; atHour: number }[] = [
        { type: 'radiology', atHour: Math.floor(plan.hours * 0.25) },
        { type: 'nursing', atHour: Math.floor(plan.hours * 0.45) },
        { type: 'progress', atHour: Math.floor(plan.hours * 0.7) },
        { type: 'radiology', atHour: Math.floor(plan.hours * 0.9) },
      ];
      // Give a couple of stays a discharge note so the F8 exclusion is
      // demonstrable rather than hypothetical.
      if (plan.index % 5 === 0) {
        schedule.push({ type: 'discharge', atHour: plan.hours - 1 });
      }

      for (const entry of schedule) {
        const text = noteText(plan, entry.type, entry.atHour);
        await NoteModel.create({
          stayId: stay._id,
          type: entry.type,
          authoredAt: new Date(admittedAt.getTime() + entry.atHour * 3_600_000),
          text,
          tokenCount: estimateTokens(text),
        });
        counts.notes++;
      }
    }
  }

  logger.info(counts, 'demo data seeded');
  return counts;
}

/**
 * Clears everything the seed creates. Used by `npm run dev:demo -- --reset`
 * and by tests.
 */
export async function clearDemoData(): Promise<void> {
  await Promise.all([
    UserModel.deleteMany({}),
    PatientModel.deleteMany({}),
    StayModel.deleteMany({}),
    ImageModel.deleteMany({}),
    NoteModel.deleteMany({}),
    PredictionModel.deleteMany({}),
    AlertModel.deleteMany({}),
  ]);
}

export { PLANS as DEMO_STAY_PLANS };
