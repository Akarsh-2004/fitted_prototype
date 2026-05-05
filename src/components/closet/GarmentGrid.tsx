import { useMemo } from 'react'
import type { Category, Garment } from '../../data/garments'
import { GarmentThumb } from './GarmentThumb'

export function GarmentGrid({
  garments,
  category,
  onAdd,
}: {
  garments: Garment[]
  category: Category
  onAdd: (garmentId: string) => void
}) {
  const filtered = useMemo(
    () => (category === 'all' ? garments : garments.filter((g) => g.category === category)),
    [category, garments],
  )

  return (
    <div className="garment-grid">
      {filtered.map((g, index) => (
        <GarmentThumb key={`${g.id}-${index}`} garment={g} onAdd={onAdd} />
      ))}
    </div>
  )
}
