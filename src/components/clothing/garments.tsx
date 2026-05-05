import type { ComponentType } from 'react'
import type { BottomPalette, HatPalette, ShoesPalette, TopPalette } from '../../types/wardrobe'
import {
  BottomGraphic,
  HatGraphic,
  ShoesGraphic,
  TopGraphic,
} from './ClothingSvgs'
import { BaggyJeansGraphic } from './BaggyJeansModel'
import { OversizedTeeGraphic } from './OversizedTeeModel'
import { HAT_PIECE_MODELS, SHOE_PIECE_MODELS } from './wardrobePieceModels'
import { WardrobePieceModel3D } from './WardrobePieceModel3D'

type HC = ComponentType<{ palette: HatPalette; className?: string }>
type TC = ComponentType<{ palette: TopPalette; className?: string }>
type BC = ComponentType<{ palette: BottomPalette; className?: string }>
type SC = ComponentType<{ palette: ShoesPalette; className?: string }>

/** Baseball cap */
function CapGraphic({ palette: p, className }: { palette: HatPalette; className?: string }) {
  return (
    <svg className={className} width={78} height={48} viewBox="0 0 78 48" fill="none" aria-hidden>
      <ellipse cx="40" cy="38" rx="32" ry="5" fill={p.brim} />
      <path
        d="M18 38C18 38 20 18 40 14C60 18 62 38 62 38C52 34 28 34 18 38Z"
        fill={p.crown}
      />
      <path d="M24 32C28 24 36 20 40 20C44 20 52 24 56 32" stroke={p.accentStroke} strokeWidth="1" fill="none" />
      <ellipse cx="40" cy="22" rx="10" ry="4" fill={p.crownInner} opacity={0.35} />
    </svg>
  )
}

/** Bucket hat */
function BucketHatGraphic({ palette: p, className }: { palette: HatPalette; className?: string }) {
  return (
    <svg className={className} width={72} height={54} viewBox="0 0 72 54" fill="none" aria-hidden>
      <ellipse cx="36" cy="48" rx="32" ry="5" fill={p.brim} />
      <path
        d="M16 46V28C16 20 24 12 36 12C48 12 56 20 56 28V46H16Z"
        fill={p.crown}
      />
      <path d="M20 28C20 24 28 18 36 18C44 18 52 24 52 28" stroke={p.accentStroke} strokeWidth="1.2" fill="none" />
      <path d="M22 32 H50" stroke={p.crownInner} strokeWidth="0.8" opacity={0.5} />
    </svg>
  )
}

/** Crewneck sweater — no hood */
function CrewSweaterGraphic({ palette: p, className }: { palette: TopPalette; className?: string }) {
  return (
    <svg className={className} width={110} height={100} viewBox="0 0 110 100" fill="none" aria-hidden>
      <path
        d="M28 22 L15 38 L8 36 L4 52 L18 56 L18 96 L92 96 L92 56 L106 52 L102 36 L95 38 L82 22 Z"
        fill={p.body}
      />
      <path d="M28 22 L15 38 L8 36 L4 52 L18 56 L22 22 Z" fill={p.sleeve} />
      <path d="M82 22 L95 38 L102 36 L106 52 L92 56 L88 22 Z" fill={p.sleeve} />
      <path
        d="M38 22 Q55 16 72 22 L68 30 Q55 26 42 30 Z"
        fill={p.hoodShade}
        opacity={0.6}
      />
      <ellipse cx="55" cy="24" rx="9" ry="5" fill={p.rib} opacity={0.45} />
      <rect x="18" y="90" width="74" height="8" rx="3" fill={p.rib} opacity={0.7} />
    </svg>
  )
}

function VNeckKnitGraphic({ palette: p, className }: { palette: TopPalette; className?: string }) {
  return (
    <svg className={className} width={110} height={100} viewBox="0 0 110 100" fill="none" aria-hidden>
      <path
        d="M28 26 L15 38 L8 36 L4 52 L18 56 L18 96 L92 96 L92 56 L106 52 L102 36 L95 38 L82 26 Z"
        fill={p.body}
      />
      <path d="M28 26 L15 38 L8 36 L4 52 L18 56 L24 26 Z" fill={p.sleeve} />
      <path d="M82 26 L95 38 L102 36 L106 52 L92 56 L86 26 Z" fill={p.sleeve} />
      <path d="M44 26 L55 44 L66 26" fill="none" stroke={p.hoodShade} strokeWidth="2" strokeLinejoin="round" />
      <path d="M50 32 L55 40 L60 32" fill={p.hood} opacity={0.35} />
      <rect x="18" y="90" width="74" height="6" rx="2" fill={p.rib} opacity={0.6} />
    </svg>
  )
}

function PoloGraphic({ palette: p, className }: { palette: TopPalette; className?: string }) {
  return (
    <svg className={className} width={110} height={100} viewBox="0 0 110 100" fill="none" aria-hidden>
      <path
        d="M30 24 L18 40 L10 38 L6 54 L20 58 L20 94 L90 94 L90 58 L104 54 L100 38 L92 40 L80 24 Z"
        fill={p.body}
      />
      <path d="M30 24 L18 40 L10 38 L6 54 L20 58 L26 24 Z" fill={p.sleeve} />
      <path d="M80 24 L92 40 L100 38 L104 54 L90 58 L84 24 Z" fill={p.sleeve} />
      <path d="M48 24 L55 36 L62 24 Z" fill={p.hood} />
      <rect x="52" y="28" width="6" height="12" rx="1" fill={p.pocket} />
      <line x1="55" y1="28" x2="55" y2="40" stroke={p.hoodShade} strokeWidth="0.8" />
    </svg>
  )
}

function TeeGraphic({ palette: p, className }: { palette: TopPalette; className?: string }) {
  return (
    <svg className={className} width={110} height={96} viewBox="0 0 110 96" fill="none" aria-hidden>
      <path
        d="M32 22 L22 34 L12 32 L8 46 L22 50 L22 90 L88 90 L88 50 L102 46 L98 32 L88 34 L78 22 Z"
        fill={p.body}
      />
      <path d="M32 22 L22 34 L12 32 L8 46 L22 50 L28 22 Z" fill={p.sleeve} />
      <path d="M78 22 L88 34 L98 32 L102 46 L88 50 L82 22 Z" fill={p.sleeve} />
      <path d="M44 22 Q55 30 66 22" stroke={p.hoodShade} strokeWidth="1.5" fill="none" />
    </svg>
  )
}

function TankGraphic({ palette: p, className }: { palette: TopPalette; className?: string }) {
  return (
    <svg className={className} width={100} height={96} viewBox="0 0 100 96" fill="none" aria-hidden>
      <path d="M36 28 L28 34 L24 88 L76 88 L72 34 L64 28 Z" fill={p.body} />
      <path d="M36 28 L30 40 L24 88" fill={p.sleeve} opacity={0.4} />
      <path d="M64 28 L70 40 L76 88" fill={p.sleeve} opacity={0.4} />
      <path d="M40 28 Q50 22 60 28" stroke={p.hoodShade} strokeWidth="1.2" fill="none" />
    </svg>
  )
}

function OxfordShirtGraphic({ palette: p, className }: { palette: TopPalette; className?: string }) {
  return (
    <svg className={className} width={110} height={100} viewBox="0 0 110 100" fill="none" aria-hidden>
      <path
        d="M28 26 L14 40 L8 38 L5 52 L18 56 L18 96 L92 96 L92 56 L105 52 L102 38 L96 40 L82 26 Z"
        fill={p.body}
      />
      <path d="M28 26 L14 40 L8 38 L5 52 L18 56 L24 26 Z" fill={p.sleeve} />
      <path d="M82 26 L96 40 L102 38 L105 52 L92 56 L86 26 Z" fill={p.sleeve} />
      <path d="M46 26 L55 40 L64 26" fill="none" stroke={p.hoodShade} strokeWidth="1" />
      <line x1="55" y1="40" x2="55" y2="62" stroke={p.pocket} strokeWidth="1" />
      <circle cx="55" cy="48" r="1.2" fill={p.hoodShade} />
      <circle cx="55" cy="54" r="1.2" fill={p.hoodShade} />
    </svg>
  )
}

function CardiganGraphic({ palette: p, className }: { palette: TopPalette; className?: string }) {
  return (
    <svg className={className} width={110} height={100} viewBox="0 0 110 100" fill="none" aria-hidden>
      <path
        d="M28 26 L14 40 L8 38 L5 52 L18 56 L18 96 L46 96 L55 52 L64 96 L92 96 L92 56 L105 52 L102 38 L96 40 L82 26 Z"
        fill={p.body}
      />
      <path d="M28 26 L14 40 L8 38 L5 52 L18 56 L24 26 Z" fill={p.sleeve} />
      <path d="M82 26 L96 40 L102 38 L105 52 L92 56 L86 26 Z" fill={p.sleeve} />
      <path d="M46 26 L55 52 L64 26" fill="none" stroke={p.hoodShade} strokeWidth="1.2" />
      <circle cx="55" cy="62" r="1.2" fill={p.pocket} />
      <circle cx="55" cy="70" r="1.2" fill={p.pocket} />
    </svg>
  )
}

function ZipHoodieGraphic({ palette: p, className }: { palette: TopPalette; className?: string }) {
  return (
    <svg className={className} width={110} height={100} viewBox="0 0 110 100" fill="none" aria-hidden>
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
      <line x1="55" y1="24" x2="55" y2="88" stroke={p.hoodShade} strokeWidth="1.5" />
      <rect x="18" y="90" width="74" height="6" rx="2" fill={p.rib} opacity={0.6} />
    </svg>
  )
}

function ChinosGraphic({ palette: p, className }: { palette: BottomPalette; className?: string }) {
  return (
    <svg className={className} width={90} height={100} viewBox="0 0 90 100" fill="none" aria-hidden>
      <rect x="14" y="4" width="62" height="10" rx="3" fill={p.waist} />
      <path d="M14 14 L13 96 L42 96 L45 50 L48 96 L77 96 L76 14 Z" fill={p.leg} />
      <path d="M45 14 L45 50" stroke={p.seam} strokeWidth="1" />
      <path d="M22 16 L32 26" stroke={p.pocketStroke} strokeWidth="0.8" />
      <path d="M68 16 L58 26" stroke={p.pocketStroke} strokeWidth="0.8" />
    </svg>
  )
}

function JoggersGraphic({ palette: p, className }: { palette: BottomPalette; className?: string }) {
  return (
    <svg className={className} width={90} height={100} viewBox="0 0 90 100" fill="none" aria-hidden>
      <rect x="14" y="6" width="62" height="12" rx="4" fill={p.waist} />
      <path d="M14 18 L13 88 L40 88 L45 50 L50 88 L77 88 L76 18 Z" fill={p.leg} />
      <path d="M45 18 L45 50" stroke={p.seam} strokeWidth="1" />
      <rect x="12" y="84" width="30" height="8" rx="3" fill={p.loop} />
      <rect x="48" y="84" width="30" height="8" rx="3" fill={p.loop} />
    </svg>
  )
}

function ShortsGraphic({ palette: p, className }: { palette: BottomPalette; className?: string }) {
  return (
    <svg className={className} width={90} height={72} viewBox="0 0 90 72" fill="none" aria-hidden>
      <rect x="14" y="4" width="62" height="10" rx="3" fill={p.waist} />
      <path d="M14 14 L12 62 L40 62 L45 28 L50 62 L78 62 L76 14 Z" fill={p.leg} />
      <path d="M45 14 L45 28" stroke={p.seam} strokeWidth="1" />
      <path
        d="M12 58 Q16 60 20 58 M48 58 Q52 60 56 58"
        stroke={p.fray}
        strokeWidth="1"
        fill="none"
      />
    </svg>
  )
}

function DressPantsGraphic({ palette: p, className }: { palette: BottomPalette; className?: string }) {
  return (
    <svg className={className} width={90} height={100} viewBox="0 0 90 100" fill="none" aria-hidden>
      <rect x="14" y="4" width="62" height="10" rx="2" fill={p.waist} />
      <path d="M14 14 L13 96 L41 96 L45 50 L49 96 L77 96 L76 14 Z" fill={p.leg} />
      <path d="M45 14 L45 96" stroke={p.seam} strokeWidth="1.2" opacity={0.7} />
      <path d="M18 16 L28 22" stroke={p.pocketStroke} strokeWidth="0.8" />
      <path d="M72 16 L62 22" stroke={p.pocketStroke} strokeWidth="0.8" />
    </svg>
  )
}

function BootGraphic({ palette: p, className }: { palette: ShoesPalette; className?: string }) {
  return (
    <svg className={className} width={80} height={52} viewBox="0 0 80 52" fill="none" aria-hidden>
      <path d="M8 40 Q8 48 18 48 L62 48 Q74 48 72 40 L68 32 L12 32 Z" fill={p.sole} />
      <path d="M12 32 L14 18 Q18 10 28 10 L50 12 Q60 14 66 22 L68 32 Z" fill={p.upper} />
      <path d="M12 32 L14 18 Q16 12 24 12 L32 14 L28 32 Z" fill={p.toe} />
      <path d="M28 14 L48 16" stroke={p.stripe} strokeWidth="1.2" />
      <rect x="54" y="12" width="10" height="18" rx="2" fill={p.heel} />
    </svg>
  )
}

function LoaferGraphic({ palette: p, className }: { palette: ShoesPalette; className?: string }) {
  return (
    <svg className={className} width={80} height={44} viewBox="0 0 80 44" fill="none" aria-hidden>
      <path d="M8 34 Q8 40 16 40 L64 40 Q72 40 72 34 L70 28 L10 28 Z" fill={p.sole} />
      <path d="M10 28 L12 22 Q20 16 40 16 Q58 16 68 22 L70 28 Z" fill={p.upper} />
      <path d="M22 24 Q40 20 58 26" stroke={p.stripe} strokeWidth="1.5" fill="none" />
      <ellipse cx="40" cy="22" rx="16" ry="5" fill={p.toe} opacity={0.35} />
    </svg>
  )
}

function SandalGraphic({ palette: p, className }: { palette: ShoesPalette; className?: string }) {
  return (
    <svg className={className} width={80} height={44} viewBox="0 0 80 44" fill="none" aria-hidden>
      <path d="M10 36 Q10 40 18 40 L62 40 Q70 40 70 36 L68 30 L12 30 Z" fill={p.sole} />
      <path d="M18 22 Q24 18 32 20 L48 20 Q56 18 62 22" stroke={p.upper} strokeWidth="4" strokeLinecap="round" />
      <path d="M26 14 L40 12 L54 14" stroke={p.lace} strokeWidth="3" strokeLinecap="round" />
    </svg>
  )
}

function RunnerGraphic({ palette: p, className }: { palette: ShoesPalette; className?: string }) {
  return <ShoesGraphic palette={p} className={className} />
}

function hat3d(id: keyof typeof HAT_PIECE_MODELS): HC {
  return function HatModel3d({ palette, className }: { palette: HatPalette; className?: string }) {
    return (
      <WardrobePieceModel3D config={HAT_PIECE_MODELS[id]} tint={palette.crown} variant="hat" className={className} />
    )
  }
}

function shoe3d(id: keyof typeof SHOE_PIECE_MODELS): SC {
  return function ShoeModel3d({ palette, className }: { palette: ShoesPalette; className?: string }) {
    return (
      <WardrobePieceModel3D config={SHOE_PIECE_MODELS[id]} tint={palette.upper} variant="shoes" className={className} />
    )
  }
}

export const HAT_GARMENTS: { id: string; label: string; Graphic: HC; is3d?: boolean }[] = [
  { id: 'british-helmet', label: 'British helmet', Graphic: hat3d('british-helmet'), is3d: true },
  { id: 'felt-cap', label: 'Felt cap', Graphic: hat3d('felt-cap'), is3d: true },
  { id: 'french-helmet', label: 'French helmet', Graphic: hat3d('french-helmet'), is3d: true },
  { id: 'hat-hip-hop', label: 'Hip-hop hat', Graphic: hat3d('hat-hip-hop'), is3d: true },
  { id: 'sunday-hat', label: 'Sunday hat', Graphic: hat3d('sunday-hat'), is3d: true },
  { id: 'victoriques-sunday-hat', label: 'Victorique Sunday', Graphic: hat3d('victoriques-sunday-hat'), is3d: true },
  { id: 'vintage-helmet', label: 'Vintage helmet', Graphic: hat3d('vintage-helmet'), is3d: true },
  { id: 'beanie', label: 'Beanie', Graphic: HatGraphic },
  { id: 'cap', label: 'Cap', Graphic: CapGraphic },
  { id: 'bucket', label: 'Bucket', Graphic: BucketHatGraphic },
]

export const TOP_GARMENTS: { id: string; label: string; Graphic: TC; is3d?: boolean }[] = [
  { id: 'oversized-tee', label: 'Oversized tee', Graphic: OversizedTeeGraphic, is3d: true },
  { id: 'hoodie', label: 'Hoodie', Graphic: TopGraphic },
  { id: 'zip-hoodie', label: 'Zip hoodie', Graphic: ZipHoodieGraphic },
  { id: 'sweater', label: 'Sweater', Graphic: CrewSweaterGraphic },
  { id: 'vneck', label: 'V-neck', Graphic: VNeckKnitGraphic },
  { id: 'cardigan', label: 'Cardigan', Graphic: CardiganGraphic },
  { id: 'polo', label: 'Polo', Graphic: PoloGraphic },
  { id: 'oxford', label: 'Oxford', Graphic: OxfordShirtGraphic },
  { id: 'tee', label: 'Tee', Graphic: TeeGraphic },
  { id: 'tank', label: 'Tank', Graphic: TankGraphic },
]

export const BOTTOM_GARMENTS: { id: string; label: string; Graphic: BC; is3d?: boolean }[] = [
  { id: 'baggy-jeans', label: 'Baggy jeans', Graphic: BaggyJeansGraphic, is3d: true },
  { id: 'jeans', label: 'Jeans', Graphic: BottomGraphic },
  { id: 'chinos', label: 'Chinos', Graphic: ChinosGraphic },
  { id: 'joggers', label: 'Joggers', Graphic: JoggersGraphic },
  { id: 'shorts', label: 'Shorts', Graphic: ShortsGraphic },
  { id: 'slacks', label: 'Slacks', Graphic: DressPantsGraphic },
]

export const SHOE_GARMENTS: { id: string; label: string; Graphic: SC; is3d?: boolean }[] = [
  { id: 'retopologized-shoes', label: 'Studio shoes', Graphic: shoe3d('retopologized-shoes'), is3d: true },
  { id: 'futuristic-shoe', label: 'Futuristic shoe', Graphic: shoe3d('futuristic-shoe'), is3d: true },
  { id: 'sports-sneaker', label: 'Sports sneaker', Graphic: shoe3d('sports-sneaker'), is3d: true },
  { id: 'sneakers', label: 'Sneakers', Graphic: RunnerGraphic },
  { id: 'boots', label: 'Boots', Graphic: BootGraphic },
  { id: 'loafers', label: 'Loafers', Graphic: LoaferGraphic },
  { id: 'sandals', label: 'Sandals', Graphic: SandalGraphic },
]

export const GARMENT_LISTS: [typeof HAT_GARMENTS, typeof TOP_GARMENTS, typeof BOTTOM_GARMENTS, typeof SHOE_GARMENTS] = [
  HAT_GARMENTS,
  TOP_GARMENTS,
  BOTTOM_GARMENTS,
  SHOE_GARMENTS,
]
