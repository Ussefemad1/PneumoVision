import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

import { expect, test, type Locator, type Page } from '@playwright/test';

/**
 * The Analyze flow against the real API and the mock inference service.
 *
 * These assert WHICH modalities reached the model — not merely that a report
 * rendered — because a report with a reduced-input banner renders even when
 * every upload was silently dropped.
 */

const credentials = {
  email: process.env.SEED_CLINICIAN_EMAIL ?? 'clinician@pneumovision.local',
  password: process.env.SEED_CLINICIAN_PASSWORD ?? '',
};

/** Procedurally drawn by fixtures/generate.ts — not a MIMIC image. */
const FIXTURE = fileURLToPath(new URL('./fixtures/synthetic-cxr.png', import.meta.url));
const SYNTHETIC_CXR_SHA256 = '1b2d312d1005bafd25b4709ae6a411a943635472e831cdf80151be4a110baff4';

const RADIOLOGY =
  'Synthetic radiology report: patchy opacity at the right lung base, no pleural effusion.';
const DISCHARGE =
  'Synthetic discharge summary: treated for community-acquired pneumonia, discharged home.';

test('the fixture is the one whose hash is recorded', () => {
  expect(createHash('sha256').update(readFileSync(FIXTURE)).digest('hex')).toBe(
    SYNTHETIC_CXR_SHA256,
  );
});

async function signIn(page: Page) {
  await page.goto('/login');
  await page.getByLabel('Email').fill(credentials.email);
  await page.getByLabel('Password').fill(credentials.password);
  await page.getByRole('button', { name: 'Sign in' }).click();
  await expect(page).toHaveURL(/\/$/);
}

async function chooseTask(page: Page, task: 'Pneumonia' | 'Mortality') {
  await page.getByRole('radio', { name: new RegExp(`^${task}`) }).check();
}

async function addNote(page: Page, text: string, type: 'radiology' | 'discharge') {
  await page.getByRole('button', { name: 'Add note' }).click();
  const n = await page.getByLabel(/^Note \d+ text$/).count();
  // Text first: a discharge note under mortality is disabled once typed as such.
  await page.getByLabel(`Note ${n} text`).fill(text);
  await page.getByLabel(`Note ${n} type`).selectOption(type);
}

async function attachXray(page: Page) {
  await page.getByLabel('Chest X-ray file').setInputFiles(FIXTURE);
  await expect(page.getByAltText('Chest X-ray preview')).toBeVisible();
}

async function loadDeterioratingEhr(page: Page) {
  await page.getByRole('tab', { name: 'Synthetic example' }).click();
  await page.getByRole('button', { name: /^Deteriorating/ }).click();
  await expect(page.getByRole('table', { name: 'EHR grid' })).toBeVisible();
}

async function runAndOpenReport(page: Page) {
  await page.getByRole('button', { name: 'Run analysis' }).click();
  await expect(page).toHaveURL(/\/predictions\/[0-9a-f]{24}$/);
  await expect(page.getByRole('heading', { name: 'Prediction report' })).toBeVisible();
}

const card = (page: Page, title: string) =>
  page.locator('section', { has: page.getByRole('heading', { name: title, exact: true }) });

const banner = (page: Page) =>
  page.getByRole('note').filter({ hasText: 'Reduced-input prediction' });

/** An F1 "evidence by source" row. */
const evidence = (page: Page, label: string): Locator =>
  card(page, '1 · Evidence by source').getByRole('listitem').filter({ hasText: label });

const CONTRIBUTED = 'Contributed to the fused score';

test('CXR + notes, EHR omitted: only EHR is reported missing', async ({ page }) => {
  await signIn(page);
  await page.goto('/analyze');
  await chooseTask(page, 'Pneumonia');
  await attachXray(page);
  await addNote(page, RADIOLOGY, 'radiology');
  await addNote(page, DISCHARGE, 'discharge');
  await runAndOpenReport(page);

  await expect(banner(page)).toBeVisible();
  await expect(banner(page)).toContainText('produced without EHR (vitals).');
  for (const present of ['Chest X-ray', 'Radiology reports', 'Discharge notes']) {
    await expect(banner(page)).not.toContainText(present);
    await expect(evidence(page, present)).toContainText(CONTRIBUTED);
  }
  await expect(evidence(page, 'EHR (vitals)')).toContainText('Not available');

  const received = card(page, 'Inputs the model received');
  await expect(received).toContainText(SYNTHETIC_CXR_SHA256.slice(0, 12));
  await expect(received).toContainText('96×96px');
  await expect(received).toContainText('MOCK MODEL — not a clinical result');
  await expect(received.getByRole('region', { name: 'EHR received' })).toContainText(
    'none received',
  );
});

test('EHR + CXR + notes: full multimodal report, no reduced-input banner', async ({ page }) => {
  await signIn(page);
  await page.goto('/analyze');
  await chooseTask(page, 'Pneumonia');
  await attachXray(page);
  await loadDeterioratingEhr(page);
  await addNote(page, RADIOLOGY, 'radiology');
  await addNote(page, DISCHARGE, 'discharge');
  await runAndOpenReport(page);

  await expect(page.getByText('Fused decision')).toBeVisible();
  await expect(banner(page)).toHaveCount(0);
  for (const present of ['EHR (vitals)', 'Chest X-ray', 'Radiology reports', 'Discharge notes']) {
    await expect(evidence(page, present)).toContainText(CONTRIBUTED);
  }

  const received = card(page, 'Inputs the model received');
  await expect(received.getByRole('region', { name: 'EHR received' })).toContainText(
    '12 of 48 hours charted',
  );
  await expect(received).toContainText(SYNTHETIC_CXR_SHA256.slice(0, 12));
});

test('mortality + discharge note: the note is excluded and DN never reaches the model', async ({
  page,
}) => {
  await signIn(page);
  await page.goto('/analyze');
  await chooseTask(page, 'Mortality');
  await attachXray(page);
  await addNote(page, RADIOLOGY, 'radiology');
  await addNote(page, DISCHARGE, 'discharge');

  await expect(page.getByLabel('Note 2 text')).toBeDisabled();
  await expect(
    page.getByText(/Excluded to prevent outcome leakage — discharge notes/),
  ).toBeVisible();

  await runAndOpenReport(page);

  // Mortality's model has no DN branch: not listed as evidence, not "missing".
  await expect(evidence(page, 'Discharge notes')).toHaveCount(0);
  await expect(banner(page)).toContainText('produced without EHR (vitals).');
  await expect(banner(page)).not.toContainText('Discharge');
  await expect(evidence(page, 'Radiology reports')).toContainText(CONTRIBUTED);

  const notes = card(page, 'Inputs the model received').getByRole('region', {
    name: 'Notes received',
  });
  await expect(notes.getByRole('listitem')).toHaveCount(1);
  await expect(notes.getByRole('listitem')).toContainText('Radiology');
  await expect(notes).toContainText('Excluded: Discharge note');
});
