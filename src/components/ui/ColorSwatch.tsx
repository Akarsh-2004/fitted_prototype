type Props = { color: string; active?: boolean; onClick: () => void; title?: string }

export function ColorSwatch({ color, active, onClick, title }: Props) {
  return (
    <button
      type="button"
      className={`swatch ${active ? 'is-active' : ''}`}
      style={{ background: color }}
      onClick={onClick}
      title={title}
      aria-label={title ?? color}
    />
  )
}
