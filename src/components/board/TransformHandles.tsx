import type { PointerEvent } from 'react'
import { RotateCw } from 'lucide-react'

export function TransformHandles({
  onResizeStart,
  onRotateStart,
}: {
  onResizeStart: (e: PointerEvent<HTMLButtonElement>) => void
  onRotateStart: (e: PointerEvent<HTMLButtonElement>) => void
}) {
  return (
    <>
      <button type="button" className="handle resize" onPointerDown={onResizeStart} aria-label="Resize piece" />
      <button type="button" className="handle rotate" onPointerDown={onRotateStart} aria-label="Rotate piece">
        <RotateCw size={13} />
      </button>
    </>
  )
}
