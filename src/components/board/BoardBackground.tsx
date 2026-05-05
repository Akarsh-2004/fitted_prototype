import { useMemo } from 'react'
import type { BackgroundPreset } from '../../store/wardrobeStore'

export function BoardBackground({
  preset,
  parallax,
}: {
  preset: BackgroundPreset
  parallax: { x: number; y: number }
}) {
  const className = useMemo(() => `board-bg board-bg--${preset}`, [preset])
  return (
    <div className={className}>
      <div className="board-bg-mesh" style={{ transform: `translate(${parallax.x}px, ${parallax.y}px)` }} />
      <svg className="board-grain" aria-hidden viewBox="0 0 200 200" preserveAspectRatio="none">
        <filter id="noiseFilter">
          <feTurbulence type="fractalNoise" baseFrequency="1.2" numOctaves={2} stitchTiles="stitch" />
        </filter>
        <rect width="100%" height="100%" filter="url(#noiseFilter)" opacity="0.06" />
      </svg>
    </div>
  )
}
