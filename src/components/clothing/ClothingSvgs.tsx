import type { BottomPalette, HatPalette, ShoesPalette, TopPalette } from '../../types/wardrobe'

type SvgProps = { className?: string }

export function HatGraphic({ palette, ...rest }: SvgProps & { palette: HatPalette }) {
  const p = palette
  return (
    <svg
      className={rest.className}
      width={72}
      height={52}
      viewBox="0 0 72 52"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      aria-hidden
    >
      <ellipse cx="36" cy="44" rx="34" ry="6" fill={p.brim} />
      <path
        d="M14 44C14 44 12 28 22 20C28 15 34 14 36 14C38 14 44 15 50 20C60 28 58 44 58 44H14Z"
        fill={p.crown}
      />
      <path
        d="M20 44C20 44 18 30 26 23C30.5 19 35 18 36 18C37 18 41.5 19 46 23C54 30 52 44 52 44H20Z"
        fill={p.crownInner}
      />
      <rect x="10" y="42" width="52" height="5" rx="2.5" fill={p.band} />
      <path
        d="M28 18C28 18 26 10 36 8C46 10 44 18 44 18"
        stroke={p.accentStroke}
        strokeWidth="1.5"
        fill="none"
      />
    </svg>
  )
}

export function TopGraphic({ palette, ...rest }: SvgProps & { palette: TopPalette }) {
  const p = palette
  return (
    <svg
      className={rest.className}
      width={110}
      height={100}
      viewBox="0 0 110 100"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      aria-hidden
    >
      <path
        d="M28 20 L15 36 L8 34 L4 50 L18 54 L18 96 L92 96 L92 54 L106 50 L102 34 L95 36 L82 20 Z"
        fill={p.body}
      />
      <path d="M28 20 L15 36 L8 34 L4 50 L18 54 L22 20 Z" fill={p.sleeve} />
      <path d="M82 20 L95 36 L102 34 L106 50 L92 54 L88 20 Z" fill={p.sleeve} />
      <path
        d="M35 20 C35 20 30 8 55 6 C80 8 75 20 75 20 L65 22 C65 22 62 14 55 13 C48 14 45 22 45 22 Z"
        fill={p.hood}
      />
      <path d="M45 22 C45 22 48 14 55 13 C62 14 65 22 65 22 L60 24 L50 24 Z" fill={p.hoodShade} />
      <rect x="46" y="50" width="18" height="14" rx="3" fill={p.pocket} opacity={0.5} />
      <text
        x="55"
        y="60"
        textAnchor="middle"
        fontFamily="Georgia, serif"
        fontStyle="italic"
        fontSize="7"
        fill="rgba(255,255,255,0.7)"
      >
        f
      </text>
      <rect x="18" y="90" width="74" height="6" rx="2" fill={p.rib} opacity={0.6} />
      <path
        d="M47 22 L45 32 M63 22 L65 32"
        stroke="rgba(255,255,255,0.5)"
        strokeWidth="1"
        strokeLinecap="round"
      />
    </svg>
  )
}

export function BottomGraphic({ palette, ...rest }: SvgProps & { palette: BottomPalette }) {
  const p = palette
  return (
    <svg
      className={rest.className}
      width={90}
      height={100}
      viewBox="0 0 90 100"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      aria-hidden
    >
      <rect x="14" y="4" width="62" height="10" rx="3" fill={p.waist} />
      <path d="M14 14 L12 96 L42 96 L45 50 L48 96 L78 96 L76 14 Z" fill={p.leg} />
      <path d="M45 14 L45 50" stroke={p.seam} strokeWidth="1" />
      <path d="M20 16 Q30 20 35 28" stroke={p.pocketStroke} strokeWidth="1" fill="none" />
      <path d="M70 16 Q60 20 55 28" stroke={p.pocketStroke} strokeWidth="1" fill="none" />
      <path d="M12 96 L42 96 L45 50 L14 14 Z" fill={p.leg} opacity={0.3} />
      <rect x="24" y="3" width="4" height="9" rx="1" fill={p.loop} />
      <rect x="62" y="3" width="4" height="9" rx="1" fill={p.loop} />
      <path
        d="M12 94 Q17 97 22 94 Q27 97 32 94 Q37 97 42 94"
        stroke={p.fray}
        strokeWidth="1.5"
        fill="none"
      />
      <path
        d="M48 94 Q53 97 58 94 Q63 97 68 94 Q73 97 78 94"
        stroke={p.fray}
        strokeWidth="1.5"
        fill="none"
      />
    </svg>
  )
}

export function ShoesGraphic({ palette, ...rest }: SvgProps & { palette: ShoesPalette }) {
  const p = palette
  return (
    <svg
      className={rest.className}
      width={80}
      height={44}
      viewBox="0 0 80 44"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      aria-hidden
    >
      <path d="M6 34 Q6 40 14 40 L66 40 Q76 40 74 34 L70 28 L10 28 Z" fill={p.sole} />
      <path d="M10 28 L12 18 Q16 12 26 12 L52 14 Q62 14 70 20 L70 28 Z" fill={p.upper} />
      <path d="M10 28 L12 18 Q14 12 20 12 L28 12 L24 28 Z" fill={p.toe} />
      <path
        d="M28 14 Q42 16 58 22"
        stroke={p.stripe}
        strokeWidth="2"
        fill="none"
        strokeLinecap="round"
      />
      <path
        d="M28 20 L36 19 M30 23 L38 22 M32 26 L40 25"
        stroke={p.lace}
        strokeWidth="1"
        strokeLinecap="round"
      />
      <path d="M26 12 L24 20 L30 20 L32 12 Z" fill={p.tongue} opacity={0.8} />
      <rect x="62" y="14" width="6" height="12" rx="2" fill={p.heel} />
    </svg>
  )
}
