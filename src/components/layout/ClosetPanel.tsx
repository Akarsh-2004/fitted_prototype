import type { Garment } from '../../data/garments'
import type { OutfitPreset } from '../../data/outfits'
import type { Category } from '../../data/garments'
import { CategoryRail } from '../closet/CategoryRail'
import { GarmentGrid } from '../closet/GarmentGrid'
import { OutfitPresets } from '../closet/OutfitPresets'
import { UploadZone } from '../closet/UploadZone'

type Props = {
  garments: Garment[]
  activeCategory: Category
  onCategoryChange: (category: Category) => void
  onAddGarment: (garmentId: string) => void
  onUpload: (files: File[]) => void
  outfits: OutfitPreset[]
  onLoadOutfit: (id: string) => void
  processingState: 'idle' | 'uploading' | 'processing' | 'refining' | 'done' | 'failed'
}

export function ClosetPanel({
  garments,
  activeCategory,
  onCategoryChange,
  onAddGarment,
  onUpload,
  outfits,
  onLoadOutfit,
  processingState,
}: Props) {
  return (
    <aside className="closet-panel">
      <h2>The Closet</h2>
      <CategoryRail activeCategory={activeCategory} onChange={onCategoryChange} />
      <UploadZone onFiles={onUpload} processingState={processingState} />
      <GarmentGrid garments={garments} category={activeCategory} onAdd={onAddGarment} />
      <OutfitPresets presets={outfits} onLoad={onLoadOutfit} />
    </aside>
  )
}
