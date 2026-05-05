export type OutfitPreset = {
  id: string
  name: string
  garmentIds: string[]
}

export const OUTFIT_PRESETS: OutfitPreset[] = [
  {
    id: 'city-layered',
    name: 'City Layered',
    garmentIds: ['baseball-cap', 'bomber-jacket', 'slim-jeans', 'street-sneakers', 'tote-bag'],
  },
  {
    id: 'soft-evening',
    name: 'Soft Evening',
    garmentIds: ['classic-tee', 'ai-wide-leg', 'street-sneakers', 'sunglasses'],
  },
  {
    id: 'editorial-float',
    name: 'Editorial Float',
    garmentIds: ['maxi-dress', 'street-sneakers', 'tote-bag'],
  },
]
