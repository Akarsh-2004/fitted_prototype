import { useMemo, useState } from 'react'
import type { BackgroundPreset, BoardPiece as BoardPieceType } from '../../store/wardrobeStore'
import type { Garment } from '../../data/garments'
import { BoardBackground } from './BoardBackground'
import { BoardPiece } from './BoardPiece'
import { ContextMenu } from './ContextMenu'
import { clampDropPoint } from '../../utils/geometry'

type Props = {
  pieces: BoardPieceType[]
  garments: Garment[]
  selectedPieceId: string | null
  backgroundPreset: BackgroundPreset
  onSelect: (id: string | null) => void
  onChangePiece: (id: string, patch: Partial<BoardPieceType>) => void
  onDelete: (id: string) => void
  onReorder: (id: string, z: number) => void
  onDropGarment: (garmentId: string, point: { x: number; y: number }) => void
}

export function BoardCanvas({
  pieces,
  garments,
  selectedPieceId,
  backgroundPreset,
  onSelect,
  onChangePiece,
  onDelete,
  onReorder,
  onDropGarment,
}: Props) {
  const [parallax, setParallax] = useState({ x: 0, y: 0 })
  const [menu, setMenu] = useState<{ id: string; x: number; y: number } | null>(null)
  const light = useMemo(() => ({ x: parallax.x * 6, y: parallax.y * 6 }), [parallax.x, parallax.y])

  return (
    <div
      className="board-canvas"
      onPointerMove={(e) => {
        const rect = (e.currentTarget as HTMLDivElement).getBoundingClientRect()
        const x = ((e.clientX - rect.left) / rect.width - 0.5) * 12
        const y = ((e.clientY - rect.top) / rect.height - 0.5) * 12
        setParallax({ x, y })
      }}
      onClick={() => {
        onSelect(null)
        setMenu(null)
      }}
      onDragOver={(e) => e.preventDefault()}
      onDrop={(e) => {
        e.preventDefault()
        const garmentId = e.dataTransfer.getData('text/garment')
        if (!garmentId) return
        const rect = (e.currentTarget as HTMLDivElement).getBoundingClientRect()
        const point = clampDropPoint(e.clientX - rect.left, e.clientY - rect.top, rect.width, rect.height)
        onDropGarment(garmentId, point)
      }}
    >
      <BoardBackground preset={backgroundPreset} parallax={parallax} />
      {pieces.map((piece) => (
        <BoardPiece
          key={piece.id}
          piece={piece}
          garment={garments.find((g) => g.id === piece.garmentId)}
          selected={piece.id === selectedPieceId}
          light={light}
          onSelect={onSelect}
          onChange={onChangePiece}
          onContextMenu={(e, id) => {
            e.preventDefault()
            setMenu({ id, x: e.clientX, y: e.clientY })
            onSelect(id)
          }}
        />
      ))}
      {menu ? (
        <ContextMenu
          x={menu.x}
          y={menu.y}
          onFront={() => {
            onReorder(menu.id, pieces.length + 1)
            setMenu(null)
          }}
          onBack={() => {
            onReorder(menu.id, 1)
            setMenu(null)
          }}
          onDelete={() => {
            onDelete(menu.id)
            setMenu(null)
          }}
        />
      ) : null}
    </div>
  )
}
