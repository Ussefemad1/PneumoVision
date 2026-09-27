import { test, expect, type Page } from '@playwright/test';

const credentials = {
  email: process.env.SEED_CLINICIAN_EMAIL ?? 'clinician@pneumovision.local',
  password: process.env.SEED_CLINICIAN_PASSWORD ?? '',
};

const PNG_1X1 = Buffer.from(
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=',
  'base64',
);

async function signIn(page: Page) {
  await page.goto('/login');
  await page.getByLabel('Email').fill(credentials.email);
  await page.getByLabel('Password').fill(credentials.password);
  await page.getByRole('button', { name: 'Sign in' }).click();
  await expect(page).toHaveURL(/\/$/);
}

async function submitAnalyze(page: Page, includeEhr: boolean) {
  await page.goto('/analyze');
  await page.getByLabel('Chest X-ray').setInputFiles({
    name: 'demo-cxr.png',
    mimeType: 'image/png',
    buffer: PNG_1X1,
  });
  await page.getByLabel('Clinical notes').fill(
    'Synthetic radiology note: bibasilar opacity with increasing respiratory symptoms.',
  );
  if (includeEhr) await page.getByRole('switch', { name: /EHR/ }).click();

  await page.getByRole('button', { name: 'Run analysis' }).click();
  await expect(page).toHaveURL(/\/predictions\/[0-9a-f]{24}$/);
  await expect(page.getByRole('heading', { name: 'Prediction report' })).toBeVisible();
}

test('CXR + notes produces reduced-input completed report', async ({ page }) => {
  await signIn(page);
  await submitAnalyze(page, false);
  await expect(page.getByRole('note')).toContainText('Reduced-input prediction');
  await expect(page.getByText('Fused decision')).toBeVisible();
});

test('CXR + notes + EHR produces completed multimodal report', async ({ page }) => {
  await signIn(page);
  await submitAnalyze(page, true);
  await expect(page.getByText('Fused decision')).toBeVisible();
  await expect(page.getByText('EHR (vitals)')).toBeVisible();
});
