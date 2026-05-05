import type { ReactNode } from 'react'

/**
 * Wraps SVG (or future Lottie) clothing renders.
 * When you add Lottie: install `@lottiefiles/dotlottie-react` (or `lottie-react`),
 * pass `lottieSrc` (public URL or imported JSON), and render the player instead of `children`.
 */
type LottiePlaceholderProps = {
  /** Accessible name for the slot, e.g. "Top — hoodie" */
  label: string
  /** e.g. "/lottie/shirt.json" — optional, not used until v2 */
  lottieSrc?: string | null
  slot: 'hat' | 'top' | 'bottom' | 'shoes'
  children: ReactNode
}

export function LottiePlaceholder({ label, lottieSrc, slot, children }: LottiePlaceholderProps) {
  if (lottieSrc) {
    return (
      <div
        className="lottie-slot lottie-slot--pending"
        data-wardrobe-slot={slot}
        data-lottie-src={lottieSrc}
        role="img"
        aria-label={label}
      >
        {/* v2: <DotLottieReact src={lottieSrc} loop autoplay /> */}
        {children}
      </div>
    )
  }

  return (
    <div
      className="lottie-slot"
      data-wardrobe-slot={slot}
      data-lottie-src=""
      role="img"
      aria-label={label}
    >
      {children}
    </div>
  )
}
