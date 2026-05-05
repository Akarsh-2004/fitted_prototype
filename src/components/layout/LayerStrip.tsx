import type { BoardPiece } from '../../store/wardrobeStore'

export function LayerStrip({
  pieces,
  selectedPieceId,
  onSelect,
  onMove,
}: {
  pieces: BoardPiece[]
  selectedPieceId: string | null
  onSelect: (id: string) => void
  onMove: (id: string, z: number) => void
}) {
  const ordered = [...pieces].sort((a, b) => b.zIndex - a.zIndex)
  const max = Math.max(ordered.length, 1)

  return (
    <aside className="layer-strip">
      {ordered.map((p, index) => (
        <button
          key={p.id}
          type="button"
          className={`layer-chip ${selectedPieceId === p.id ? 'is-active' : ''}`}
          onClick={() => onSelect(p.id)}
          onDoubleClick={() => onMove(p.id, max - index)}
          title="Double click to reorder"
        >
          {p.garmentId}
        </button>
      ))}
    </aside>
  )
}
