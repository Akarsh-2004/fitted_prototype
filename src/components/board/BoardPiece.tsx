import { memo, useMemo, useRef, type MouseEvent, type PointerEvent as ReactPointerEvent } from 'react'
import { useGesture } from '@use-gesture/react'
import { motion } from 'framer-motion'
import type { BoardPiece as BoardPieceType } from '../../store/wardrobeStore'
import type { Garment } from '../../data/garments'
import { GarmentSVG } from '../garments/GarmentSVG'
import { TransformHandles } from './TransformHandles'

type Props = {
  piece: BoardPieceType
  garment?: Garment
  selected: boolean
  light: { x: number; y: number }
  onSelect: (id: string) => void
  onChange: (id: string, patch: Partial<BoardPieceType>) => void
  onContextMenu: (e: MouseEvent, id: string) => void
}

export const BoardPiece = memo(function BoardPiece({
  piece,
  garment,
  selected,
  light,
  onSelect,
  onChange,
  onContextMenu,
}: Props) {
  const ref = useRef<HTMLDivElement | null>(null)
  const startRef = useRef({ x: piece.x, y: piece.y, width: piece.width, height: piece.height, rotation: piece.rotation })

  useGesture(
    {
      onDrag: ({ first, movement: [mx, my] }) => {
        if (first) startRef.current = { ...startRef.current, x: piece.x, y: piece.y }
        onChange(piece.id, { x: startRef.current.x + mx, y: startRef.current.y + my })
      },
    },
    { target: ref, drag: { pointer: { touch: true }, from: () => [0, 0], filterTaps: true } },
  )

  const shadow = useMemo(() => `${light.x * 0.04}px ${light.y * 0.04}px 28px rgba(0,0,0,.17)`, [light.x, light.y])

  return (
    <motion.div
      ref={ref}
      className={`board-piece ${selected ? 'is-selected' : ''}`}
      onPointerDown={() => onSelect(piece.id)}
      onContextMenu={(e) => onContextMenu(e, piece.id)}
      style={{
        left: piece.x,
        top: piece.y,
        width: piece.width,
        height: piece.height,
        opacity: piece.opacity,
        zIndex: piece.zIndex,
        transform: `rotate(${piece.rotation}deg) scaleX(${piece.flipX ? -1 : 1}) scaleY(${piece.flipY ? -1 : 1})`,
        boxShadow: shadow,
      }}
      initial={{ opacity: 0, scale: 1.04, rotate: piece.rotation + 2 }}
      animate={{ opacity: piece.opacity, scale: 1, rotate: piece.rotation }}
      transition={{ type: 'spring', stiffness: 300, damping: 20 }}
    >
      {garment?.kind === 'svg' && garment.svgKey ? <GarmentSVG type={garment.svgKey} color={piece.colorOverride} /> : null}
      {garment?.kind === 'ai' ? <div className="ai-piece">{garment?.name ?? 'AI Piece'}</div> : null}
      {garment?.kind === 'upload' && garment.imageSrc ? (
        <img
          src={garment.imageSrc}
          alt={garment.name}
          className="board-upload-img"
          style={{
            filter: `contrast(${1 + (garment.refineParams?.smoothness ?? 0.5) * 0.5})`,
          }}
        />
      ) : null}

      {selected ? (
        <TransformHandles
          onResizeStart={(e: ReactPointerEvent<HTMLButtonElement>) => {
            e.stopPropagation()
            const sx = e.clientX
            const sy = e.clientY
            startRef.current = { ...startRef.current, width: piece.width, height: piece.height }
            const move = (ev: PointerEvent) => {
              onChange(piece.id, {
                width: Math.max(80, startRef.current.width + (ev.clientX - sx)),
                height: Math.max(100, startRef.current.height + (ev.clientY - sy)),
              })
            }
            const up = () => {
              window.removeEventListener('pointermove', move)
              window.removeEventListener('pointerup', up)
            }
            window.addEventListener('pointermove', move)
            window.addEventListener('pointerup', up)
          }}
          onRotateStart={(e: ReactPointerEvent<HTMLButtonElement>) => {
            e.stopPropagation()
            const sx = e.clientX
            startRef.current = { ...startRef.current, rotation: piece.rotation }
            const move = (ev: PointerEvent) => {
              onChange(piece.id, { rotation: startRef.current.rotation + (ev.clientX - sx) * 0.4 })
            }
            const up = () => {
              window.removeEventListener('pointermove', move)
              window.removeEventListener('pointerup', up)
            }
            window.addEventListener('pointermove', move)
            window.addEventListener('pointerup', up)
          }}
        />
      ) : null}
    </motion.div>
  )
})
