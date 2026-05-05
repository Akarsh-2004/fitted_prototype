export type Category = 'all' | 'head' | 'tops' | 'bottoms' | 'shoes' | 'accessories'
export type GarmentKind = 'svg' | 'ai' | 'upload'

export type Garment = {
  id: string
  name: string
  category: Exclude<Category, 'all'>
  kind: GarmentKind
  palette: string[]
  aspectRatio: number
  svgKey?: SvgGarmentKey
  variants?: Record<string, string>
  imageSrc?: string
  originalImageSrc?: string
  maskSrc?: string
  compositeSrc?: string
  processingState?: 'idle' | 'uploading' | 'processing' | 'refining' | 'done' | 'failed'
  processingMeta?: {
    maskQuality: number
    processingTime: number
    usedFallback: boolean
    warpMode: 'affine' | 'tps'
    requestKey: string
  }
  refineParams?: {
    edgeFeather: number
    morphKernel: number
    smoothness: number
  }
}

export type SvgGarmentKey = 'tshirt' | 'hoodie' | 'jeans' | 'dress' | 'jacket' | 'sneaker'

export const GARMENTS: Garment[] = [
  {
    id: 'classic-tee',
    name: 'Classic T-Shirt',
    category: 'tops',
    kind: 'svg',
    palette: ['#e8dfd2', '#3d4354', '#c88975'],
    aspectRatio: 0.95,
    svgKey: 'tshirt',
    variants: { collar: 'round' },
  },
  {
    id: 'oversized-hoodie',
    name: 'Oversized Hoodie',
    category: 'tops',
    kind: 'svg',
    palette: ['#b5a38f', '#6f7d74', '#2a2e3a'],
    aspectRatio: 0.94,
    svgKey: 'hoodie',
  },
  {
    id: 'slim-jeans',
    name: 'Slim Jeans',
    category: 'bottoms',
    kind: 'svg',
    palette: ['#3f5f94', '#4f6175', '#212638'],
    aspectRatio: 0.62,
    svgKey: 'jeans',
  },
  {
    id: 'maxi-dress',
    name: 'Maxi Dress',
    category: 'tops',
    kind: 'svg',
    palette: ['#d4a6af', '#98a8bf', '#322f42'],
    aspectRatio: 0.56,
    svgKey: 'dress',
  },
  {
    id: 'bomber-jacket',
    name: 'Bomber Jacket',
    category: 'tops',
    kind: 'svg',
    palette: ['#475064', '#8e9aad', '#7e6a52'],
    aspectRatio: 0.9,
    svgKey: 'jacket',
  },
  {
    id: 'street-sneakers',
    name: 'Sneaker Pair',
    category: 'shoes',
    kind: 'svg',
    palette: ['#efefef', '#d6b893', '#202225'],
    aspectRatio: 1.34,
    svgKey: 'sneaker',
  },
  {
    id: 'ai-wide-leg',
    name: 'Wide-leg Trousers',
    category: 'bottoms',
    kind: 'ai',
    palette: ['#ece4da', '#2f3545', '#7c5b4a'],
    aspectRatio: 0.72,
  },
  {
    id: 'ai-mini-skirt',
    name: 'Mini Skirt',
    category: 'bottoms',
    kind: 'ai',
    palette: ['#ceb2bc', '#282632', '#9b7f60'],
    aspectRatio: 0.9,
  },
  {
    id: 'baseball-cap',
    name: 'Baseball Cap',
    category: 'head',
    kind: 'ai',
    palette: ['#d3c7b7', '#31384a', '#7a7f8c'],
    aspectRatio: 1.24,
  },
  {
    id: 'tote-bag',
    name: 'Tote Bag',
    category: 'accessories',
    kind: 'ai',
    palette: ['#d8ccbb', '#3a3d47', '#8792aa'],
    aspectRatio: 0.9,
  },
  {
    id: 'sunglasses',
    name: 'Sunglasses',
    category: 'accessories',
    kind: 'ai',
    palette: ['#1f2025', '#515464', '#9e8f7f'],
    aspectRatio: 2.4,
  },
]
