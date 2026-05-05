type Props = {
  x: number
  y: number
  onFront: () => void
  onBack: () => void
  onDelete: () => void
}

export function ContextMenu({ x, y, onFront, onBack, onDelete }: Props) {
  return (
    <div className="context-menu" style={{ left: x, top: y }}>
      <button type="button" onClick={onFront}>
        Bring to front
      </button>
      <button type="button" onClick={onBack}>
        Send to back
      </button>
      <button type="button" onClick={onDelete}>
        Delete
      </button>
    </div>
  )
}
