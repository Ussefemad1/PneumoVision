/**
 * Platform-wide constants shared by the API gateway and the web SPA.
 *
 * Everything here is pinned to what the training code in `medpatch/` actually
 * does. See CLAUDE.md § "Ground truth from the training code" before changing a
 * value — a mismatch here silently corrupts inference inputs.
 */

/** RBAC roles (Section 7: deny by default, object-level checks per role). */
export const ROLES = ['admin', 'clinician', 'radiologist', 'researcher'] as const;
export type Role = (typeof ROLES)[number];

/** Roles allowed to read identifiable clinical content. `researcher` is not one
 *  of them — it sees pseudonymised and aggregate data only, never note text. */
export const CLINICAL_ROLES: readonly Role[] = ['admin', 'clinician', 'radiologist'];

/**
 * Prediction tasks.
 *
 * `mortality`  -> medpatch task `in-hospital-mortality`, num_classes = 1,
 *                 modalities EHR-CXR-RR (discharge notes excluded upstream).
 * `pneumonia`  -> medpatch task `phenotyping`, num_classes = 25, we surface
 *                 class index 21 only. Modalities EHR-CXR-RR-DN.
 */
export const TASKS = ['mortality', 'pneumonia'] as const;
export type Task = (typeof TASKS)[number];

/** How a platform task maps onto a training run. */
export const TASK_SPEC = {
  mortality: {
    medpatchTask: 'in-hospital-mortality',
    fusionType: 'c-msma',
    numClasses: 1,
    classIndex: 0,
    modalities: ['EHR', 'CXR', 'RR'],
    label: '48-hour in-hospital mortality',
  },
  pneumonia: {
    medpatchTask: 'phenotyping',
    fusionType: 'c-e-msma',
    numClasses: 25,
    classIndex: 21,
    modalities: ['EHR', 'CXR', 'RR', 'DN'],
    label: 'Pneumonia (phenotype)',
  },
} as const satisfies Record<
  Task,
  {
    medpatchTask: string;
    fusionType: string;
    numClasses: number;
    classIndex: number;
    modalities: readonly string[];
    label: string;
  }
>;

/**
 * Modalities the platform tracks. The training code treats radiology reports
 * (RR) and discharge notes (DN) as two separate modalities with their own
 * encoders, confidence predictors and temperatures — not one "notes" input.
 */
export const MODALITIES = ['ehr', 'cxr', 'rr', 'dn'] as const;
export type Modality = (typeof MODALITIES)[number];

/** Note types. `discharge` maps to the DN modality; the rest map to RR. */
export const NOTE_TYPES = ['radiology', 'progress', 'nursing', 'discharge'] as const;
export type NoteType = (typeof NOTE_TYPES)[number];

/**
 * F8 leakage control. Discharge notes leak the outcome, so they are never sent
 * for mortality. This is the single source of truth the API enforces against.
 */
export const NOTE_TYPES_BY_TASK: Readonly<Record<Task, readonly NoteType[]>> = {
  mortality: ['radiology', 'progress', 'nursing'],
  pneumonia: ['radiology', 'progress', 'nursing', 'discharge'],
};

/**
 * The 25 phenotype labels, in the order the training code lists them.
 * Source: medpatch/datasets/DataFusion.py `CLASSES`. Index 21 is pneumonia.
 * Guarded by a parity test against the Python source.
 */
export const PHENOTYPE_CLASSES = [
  'Acute and unspecified renal failure',
  'Acute cerebrovascular disease',
  'Acute myocardial infarction',
  'Cardiac dysrhythmias',
  'Chronic kidney disease',
  'Chronic obstructive pulmonary disease and bronchiectasis',
  'Complications of surgical procedures or medical care',
  'Conduction disorders',
  'Congestive heart failure; nonhypertensive',
  'Coronary atherosclerosis and other heart disease',
  'Diabetes mellitus with complications',
  'Diabetes mellitus without complication',
  'Disorders of lipid metabolism',
  'Essential hypertension',
  'Fluid and electrolyte disorders',
  'Gastrointestinal hemorrhage',
  'Hypertension with complications and secondary hypertension',
  'Other liver diseases',
  'Other lower respiratory disease',
  'Other upper respiratory disease',
  'Pleurisy; pneumothorax; pulmonary collapse',
  'Pneumonia (except that caused by tuberculosis or sexually transmitted disease)',
  'Respiratory failure; insufficiency; arrest (adult)',
  'Septicemia (except in labor)',
  'Shock',
] as const;

export const PNEUMONIA_CLASS_INDEX = 21 as const;
export const RESPIRATORY_FAILURE_CLASS_INDEX = 22 as const;

/**
 * CXR encoder geometry: timm `vit_small_patch16_384` on 384x384 input gives a
 * 24x24 patch grid plus a CLS token. The confidence heatmap is drawn on the
 * 24x24 grid; the CLS token is not a spatial patch and is excluded.
 */
export const CXR_IMAGE_SIZE = 384 as const;
export const CXR_PATCH_SIZE = 16 as const;
export const CXR_PATCH_GRID = 24 as const;
export const CXR_TOKEN_COUNT = CXR_PATCH_GRID * CXR_PATCH_GRID + 1;

/** ImageNet normalisation, matching datasets/cxr_dataset.py get_transforms(). */
export const CXR_NORM_MEAN = [0.485, 0.456, 0.406] as const;
export const CXR_NORM_STD = [0.229, 0.224, 0.225] as const;

/** Text encoder: Bio_ClinicalBERT, 512-token chunks, mean-pooled across chunks. */
export const TEXT_MODEL_NAME = 'emilyalsentzer/Bio_ClinicalBERT';
export const TEXT_CHUNK_TOKENS = 512 as const;

/** Confidence-based patching threshold θ (MedPatch §3.3). */
export const DEFAULT_THETA = 0.75;

/** Prediction lifecycle. */
export const PREDICTION_STATUSES = ['queued', 'running', 'done', 'failed'] as const;
export type PredictionStatus = (typeof PREDICTION_STATUSES)[number];

export const ALERT_SEVERITIES = ['info', 'warning', 'critical'] as const;
export type AlertSeverity = (typeof ALERT_SEVERITIES)[number];

export const ALERT_STATUSES = ['open', 'acknowledged', 'resolved'] as const;
export type AlertStatus = (typeof ALERT_STATUSES)[number];

export const STAY_STATUSES = ['active', 'discharged'] as const;
export type StayStatus = (typeof STAY_STATUSES)[number];

export const CXR_VIEWS = ['AP', 'PA', 'LAT'] as const;
export type CxrView = (typeof CXR_VIEWS)[number];

/** Replay speeds offered by ICU Replay mode (P1). */
export const REPLAY_SPEEDS = [1, 10, 60] as const;
export type ReplaySpeed = (typeof REPLAY_SPEEDS)[number];

/** Pagination ceiling enforced on every list endpoint. */
export const MAX_PAGE_SIZE = 100 as const;
export const DEFAULT_PAGE_SIZE = 25 as const;

/** Shown on every prediction surface (P7). */
export const DISCLAIMER =
  'Decision support only. PneumoVision is a research prototype and is not a ' +
  'medical device. It must not be used for clinical decision-making.';

/**
 * Reference numbers from the MedPatch paper (Al Jorf & Shamout, MLHC 2025).
 * Displayed read-only beside our own metrics, always labelled as the paper's.
 */
export const MEDPATCH_REFERENCE_METRICS = {
  mortality: { auroc: 0.876, auprc: 0.558 },
  pneumonia: { auroc: 0.902 },
} as const;
