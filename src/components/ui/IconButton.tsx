import type { ReactNode } from 'react'

type Props = { title: string; onClick?: () => void; children: ReactNode }

export function IconButton({ title, onClick, children }: Props) {
  return (
    <button type="button" className="icon-btn" onClick={onClick} title={title} aria-label={title}>
      {children}
    </button>
  )
}
