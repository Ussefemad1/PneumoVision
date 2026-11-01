/**
 * Brand mark and wordmark, ported from docs/design/pneumovision-demo.html.
 *
 * The mark is half a lung and half a data grid: the two larger dots are the
 * "hot" pair, the one spot of emphasis. Two palettes rather than CSS variables,
 * because the mark sits on the navy sidebar and on a white card in the same
 * theme — `onDark` picks the palette, not the current theme.
 *
 * Decorative by default: every placement in the app sits next to the word
 * "PneumoVision" already, so an accessible name here would just repeat it.
 */

const DOTS: readonly (readonly [number, number])[] = [
  [39, 24],
  [45, 24],
  [39, 30],
  [45, 30],
  [51, 30],
  [39, 36],
  [45, 36],
  [51, 36],
  [39, 42],
  [45, 42],
  [51, 42],
  [39, 48],
  [45, 48],
];

export function LogoMark({
  size = 32,
  onDark = false,
  label,
}: {
  size?: number | undefined;
  onDark?: boolean | undefined;
  /** Supply only where no adjacent text names the product. */
  label?: string | undefined;
}) {
  const ink = onDark ? '#FFFFFF' : '#2F4156';
  const fill = onDark ? '#C8D9E6' : '#567C8D';
  const hot = onDark ? '#9FE3F5' : '#2F4156';

  return (
    <svg
      {...(label ? { role: 'img', 'aria-label': label } : { 'aria-hidden': true })}
      width={size}
      height={size}
      viewBox="0 0 64 64"
    >
      <path d="M30 20C23 16 14 23 11.5 36C9.5 47 13 55 21 55C26.5 55 30 51.5 30 46Z" fill={fill} />
      <path
        d="M32 6V20M32 20L27.5 25.5M32 20L36.5 25.5"
        fill="none"
        stroke={ink}
        strokeWidth="3.2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      {DOTS.map(([cx, cy]) => {
        const isHot = cy === 42 && cx >= 45;
        return (
          <circle
            key={`${cx}-${cy}`}
            cx={cx}
            cy={cy}
            r={isHot ? 2.8 : 2.3}
            fill={isHot ? hot : fill}
          />
        );
      })}
    </svg>
  );
}

/**
 * "Pneumo" in the inherited colour, "Vision" in `tone`. The parent sets the
 * size, so this works at 17px in the sidebar and 64px on the login screen.
 */
export function Wordmark({ tone = 'accent' }: { tone?: 'accent' | 'muted' | undefined }) {
  return (
    <>
      Pneumo
      <span className={tone === 'accent' ? 'text-[var(--accent)]' : 'text-[var(--sidebar-muted)]'}>
        Vision
      </span>
    </>
  );
}
