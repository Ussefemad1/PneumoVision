import { Card, EmptyState } from '../components/ui.jsx';

/**
 * What is running, and how it compares with the paper.
 *
 * Every number here is static reference data transcribed from Al Jorf &
 * Shamout, MedPatch (MLHC 2025) — it is not measured by this app. The
 * PneumoVision columns stay "Pending export" until the model track exports
 * test-split predictions for the active run; they are deliberately not
 * filled with anything derived or estimated.
 */

const PENDING = 'Pending export';

const HEADLINE: { task: string; setting: string; auroc: string; auprc: string }[] = [
  { task: 'Pneumonia (phenotype class)', setting: 'EHR + CXR', auroc: '0.820', auprc: '0.396' },
  {
    task: 'Pneumonia (phenotype class)',
    setting: 'All four sources',
    auroc: '0.902',
    auprc: '0.564',
  },
  { task: 'Respiratory failure', setting: 'All four sources', auroc: '0.902', auprc: '0.661' },
  { task: 'Clinical conditions (25)', setting: 'All four sources', auroc: '0.862', auprc: '0.614' },
  { task: '48 h mortality', setting: 'EHR + CXR + RR', auroc: '0.876', auprc: '0.558' },
];

const SINGLE_SOURCE: { source: string; encoder: string; mortality: string; conditions: string }[] =
  [
    { source: 'EHR', encoder: 'LSTM', mortality: '0.861', conditions: '0.764' },
    { source: 'CXR', encoder: 'ViT', mortality: '0.723', conditions: '0.692' },
    { source: 'Radiology reports', encoder: 'BioBERT', mortality: '0.742', conditions: '0.776' },
    { source: 'Discharge notes', encoder: 'BioBERT', mortality: '—', conditions: '0.856' },
  ];

const ABLATIONS: { removed: string; auroc: string; auprc: string }[] = [
  { removed: 'None (full MedPatch)', auroc: '0.876', auprc: '0.558' },
  { removed: 'Unimodal predictions', auroc: '0.850', auprc: '0.484' },
  { removed: 'Missingness module', auroc: '0.875', auprc: '0.552' },
  { removed: 'Calibration', auroc: '0.876', auprc: '0.557' },
  { removed: 'Confidence patching', auroc: '0.876', auprc: '0.549' },
];

export function ModelsPage() {
  return (
    <div className="flex flex-col gap-[22px]">
      <header>
        <h1 className="text-[28px] font-bold tracking-[-0.02em] text-ink">Models</h1>
        <p className="mt-1.5 text-ink-secondary">
          The active model, and how it compares with the MedPatch paper.
        </p>
      </header>

      <Card bodyClassName="flex flex-wrap items-center gap-8 p-6">
        <div className="min-w-[260px] flex-1">
          <div className="eyebrow">Active model</div>
          <h2 className="mt-1.5 text-[22px] font-bold tracking-[-0.02em] text-ink">
            MedPatch fusion · Round 3
          </h2>
          <p className="mt-1.5 text-sm text-ink-secondary">
            Run r3-fusion-002 · tasks: pneumonia phenotype, 48 h mortality
          </p>
        </div>
        <dl className="grid shrink-0 grid-cols-[auto_auto] gap-x-5 gap-y-2 text-[13px]">
          <dt className="text-ink-secondary">Confidence threshold θ</dt>
          <dd className="tnum text-ink">0.75</dd>
          <dt className="text-ink-secondary">Encoders</dt>
          <dd className="text-ink">LSTM · ViT-Small · BioBERT</dd>
          <dt className="text-ink-secondary">Calibration</dt>
          <dd className="text-ink">Temperature scaling</dd>
        </dl>
      </Card>

      <Card
        title="Performance"
        subtitle="AUROC / AUPRC on the test split. The paper columns are read-only reference data from Al Jorf & Shamout (MLHC 2025)."
        bodyClassName="p-0"
      >
        <div className="w-full min-w-0 overflow-x-auto">
          <table className="w-full min-w-[760px] text-sm">
            <thead>
              <tr className="table-head">
                <th>Task</th>
                <th>Setting</th>
                <th>Paper · AUROC</th>
                <th>Paper · AUPRC</th>
                <th>PneumoVision · AUROC</th>
                <th>PneumoVision · AUPRC</th>
              </tr>
            </thead>
            <tbody>
              {HEADLINE.map((row) => (
                <tr
                  key={`${row.task}-${row.setting}`}
                  className="border-t border-hairline hover:bg-softer"
                >
                  <td className="px-3.5 py-3 pl-5 font-medium text-ink">{row.task}</td>
                  <td className="px-3.5 py-3 text-ink-secondary">{row.setting}</td>
                  <td className="tnum px-3.5 py-3 text-ink">{row.auroc}</td>
                  <td className="tnum px-3.5 py-3 text-ink">{row.auprc}</td>
                  <td className="px-3.5 py-3 text-ink-muted">{PENDING}</td>
                  <td className="px-3.5 py-3 pr-5 text-ink-muted">{PENDING}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card title="Single-source baselines (paper)" bodyClassName="p-0">
          <div className="w-full min-w-0 overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="table-head">
                  <th>Source</th>
                  <th>Encoder</th>
                  <th>Mortality AUROC</th>
                  <th>Conditions AUROC</th>
                </tr>
              </thead>
              <tbody>
                {SINGLE_SOURCE.map((row) => (
                  <tr key={row.source} className="border-t border-hairline hover:bg-softer">
                    <td className="px-3.5 py-3 pl-5 font-medium text-ink">{row.source}</td>
                    <td className="px-3.5 py-3 text-ink-secondary">{row.encoder}</td>
                    <td className="tnum px-3.5 py-3 text-ink">{row.mortality}</td>
                    <td className="tnum px-3.5 py-3 pr-5 text-ink">{row.conditions}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>

        <Card title="Ablations · mortality (paper)" bodyClassName="p-0">
          <div className="w-full min-w-0 overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="table-head">
                  <th>Removed component</th>
                  <th>AUROC</th>
                  <th>AUPRC</th>
                </tr>
              </thead>
              <tbody>
                {ABLATIONS.map((row) => (
                  <tr key={row.removed} className="border-t border-hairline hover:bg-softer">
                    <td className="px-3.5 py-3 pl-5 font-medium text-ink">{row.removed}</td>
                    <td className="tnum px-3.5 py-3 text-ink">{row.auroc}</td>
                    <td className="tnum px-3.5 py-3 pr-5 text-ink">{row.auprc}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      </div>

      <Card>
        <EmptyState
          icon="◠"
          title="Reliability diagram and ECE"
          description="Appears here once the team exports validation-set predictions for the active model."
        />
      </Card>
    </div>
  );
}
