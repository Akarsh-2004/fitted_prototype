import type { SVGProps } from 'react'
import type { SvgGarmentKey } from '../../data/garments'

type Props = {
  type: SvgGarmentKey
  color: string
  className?: string
}

const svgBase = { width: '100%', height: '100%', viewBox: '0 0 220 240' } satisfies SVGProps<SVGSVGElement>

function FilterDefs({ id }: { id: string }) {
  return (
    <defs>
      <filter id={`grain-${id}`} x="-20%" y="-20%" width="140%" height="140%">
        <feTurbulence type="fractalNoise" baseFrequency="0.9" numOctaves={2} result="noise" />
        <feColorMatrix
          in="noise"
          type="matrix"
          values="0 0 0 0 0.6 0 0 0 0 0.6 0 0 0 0 0.6 0 0 0 0.08 0"
        />
      </filter>
      <filter id={`cloth-shadow-${id}`} x="-30%" y="-30%" width="160%" height="160%">
        <feDropShadow dx="0" dy="6" stdDeviation="6" floodOpacity="0.2" />
      </filter>
    </defs>
  )
}

export function GarmentSVG({ type, color, className }: Props) {
  const id = `${type}-${color.replace('#', '')}`
  if (type === 'hoodie') return <Hoodie color={color} id={id} className={className} />
  if (type === 'jeans') return <Jeans color={color} id={id} className={className} />
  if (type === 'dress') return <Dress color={color} id={id} className={className} />
  if (type === 'jacket') return <Jacket color={color} id={id} className={className} />
  if (type === 'sneaker') return <Sneaker color={color} id={id} className={className} />
  return <TShirt color={color} id={id} className={className} />
}

function TShirt({ color, id, className }: { color: string; id: string; className?: string }) {
  return (
    <svg {...svgBase} className={className}>
      <FilterDefs id={id} />
      <path
        d="M68 40c-19-8-41-2-52 13L6 66l12 34 42-15-2 142h104l-2-142 42 15 12-34-10-13c-11-15-33-21-52-13l-20 16c-8 6-16 6-24 0l-20-16Z"
        fill={color}
        filter={`url(#cloth-shadow-${id})`}
      />
      <path d="M88 44c4 10 13 16 22 16s18-6 22-16" stroke="#2a2a2a55" strokeWidth="2" fill="none" />
      <rect x="70" y="214" width="80" height="6" rx="2" fill="#0000002b" />
      <path d="M68 40c-19-8-41-2-52 13L6 66l12 34 42-15-2 142h104l-2-142 42 15 12-34-10-13c-11-15-33-21-52-13l-20 16c-8 6-16 6-24 0l-20-16Z" filter={`url(#grain-${id})`} />
    </svg>
  )
}

function Hoodie({ color, id, className }: { color: string; id: string; className?: string }) {
  return (
    <svg {...svgBase} className={className}>
      <FilterDefs id={id} />
      <path d="M74 50c-18-7-40-1-52 14L10 78l10 34 42-15-4 130h104l-4-130 42 15 10-34-12-14c-12-15-34-21-52-14-10 18-34 18-44 0Z" fill={color} filter={`url(#cloth-shadow-${id})`} />
      <path d="M84 50c6-18 17-28 26-28s20 10 26 28c-18 10-34 10-52 0Z" fill="#00000033" />
      <rect x="78" y="158" width="64" height="24" rx="10" fill="#0000003b" />
      <path d="M106 76v66M114 76v66" stroke="#00000066" strokeWidth="2" />
      <path d="M74 50c-18-7-40-1-52 14L10 78l10 34 42-15-4 130h104l-4-130 42 15 10-34-12-14c-12-15-34-21-52-14-10 18-34 18-44 0Z" filter={`url(#grain-${id})`} />
    </svg>
  )
}

function Jeans({ color, id, className }: { color: string; id: string; className?: string }) {
  return (
    <svg {...svgBase} className={className} viewBox="0 0 170 260">
      <FilterDefs id={id} />
      <path d="M16 20h138v86l-10 22-29 124H54L25 128l-9-22V20Z" fill={color} filter={`url(#cloth-shadow-${id})`} />
      <path d="M84 20v102l31 130M84 122 53 252" stroke="#ffffff4f" strokeWidth="2" />
      <rect x="16" y="8" width="138" height="16" rx="3" fill="#00000046" />
      <circle cx="31" cy="25" r="3" fill="#ffffff66" />
      <path d="M16 20h138v86l-10 22-29 124H54L25 128l-9-22V20Z" filter={`url(#grain-${id})`} />
    </svg>
  )
}

function Dress({ color, id, className }: { color: string; id: string; className?: string }) {
  return (
    <svg {...svgBase} className={className} viewBox="0 0 170 280">
      <FilterDefs id={id} />
      <path d="M53 24c-10 0-23 7-25 18L23 63l28 10-4 42L8 270h154l-39-155-4-42 28-10-5-21c-2-11-15-18-25-18-5-14-27-14-32 0-5-14-27-14-32 0Z" fill={color} filter={`url(#cloth-shadow-${id})`} />
      <path d="M51 74h68M85 24v36M52 115l32 155M118 115 86 270" stroke="#0000003d" strokeWidth="1.8" />
      <path d="M53 24c-10 0-23 7-25 18L23 63l28 10-4 42L8 270h154l-39-155-4-42 28-10-5-21c-2-11-15-18-25-18-5-14-27-14-32 0-5-14-27-14-32 0Z" filter={`url(#grain-${id})`} />
    </svg>
  )
}

function Jacket({ color, id, className }: { color: string; id: string; className?: string }) {
  return (
    <svg {...svgBase} className={className} viewBox="0 0 220 250">
      <FilterDefs id={id} />
      <path d="M74 38c-21-10-47-4-60 14L2 71l10 38 48-17-3 154h106l-3-154 48 17 10-38-12-19c-13-18-39-24-60-14l-20 18-16 68-16-68-20-18Z" fill={color} filter={`url(#cloth-shadow-${id})`} />
      <path d="M90 56 74 38h20l16 68 16-68h20l-16 18-20 44H94l-20-44Z" fill="#00000054" />
      <circle cx="110" cy="146" r="5" fill="#ffffff66" />
      <circle cx="110" cy="174" r="5" fill="#ffffff66" />
      <path d="M74 38c-21-10-47-4-60 14L2 71l10 38 48-17-3 154h106l-3-154 48 17 10-38-12-19c-13-18-39-24-60-14l-20 18-16 68-16-68-20-18Z" filter={`url(#grain-${id})`} />
    </svg>
  )
}

function Sneaker({ color, id, className }: { color: string; id: string; className?: string }) {
  return (
    <svg {...svgBase} className={className} viewBox="0 0 260 170">
      <FilterDefs id={id} />
      <g filter={`url(#cloth-shadow-${id})`}>
        <path d="M16 96c26-5 46-16 66-36h30l15 20c8 11 19 18 34 22l34 8v24H16V96Z" fill={color} />
        <path d="M245 96c-26-5-46-16-66-36h-30l-15 20c-8 11-19 18-34 22l-34 8v24h179V96Z" fill={color} />
      </g>
      <path d="M22 120h90M34 111h76M166 120h90M178 111h76" stroke="#00000044" strokeWidth="2" />
      <path d="M16 96c26-5 46-16 66-36h30l15 20c8 11 19 18 34 22l34 8v24H16V96Z" filter={`url(#grain-${id})`} />
      <path d="M245 96c-26-5-46-16-66-36h-30l-15 20c-8 11-19 18-34 22l-34 8v24h179V96Z" filter={`url(#grain-${id})`} />
    </svg>
  )
}
