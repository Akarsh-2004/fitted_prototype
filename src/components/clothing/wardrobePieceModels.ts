export type PieceModelFormat = 'obj' | 'fbx' | 'glb'

export type TextureSlotName = 'map' | 'normalMap' | 'roughnessMap' | 'metalnessMap'

export type WardrobePieceModelConfig = {
  id: string
  format: PieceModelFormat
  model: string
  /** Load in order; paired with textureSlots */
  texturePaths: string[]
  textureSlots: TextureSlotName[]
  fit: number
  /** Degrees, Euler order YXZ (match baggy jeans) */
  rotationDeg?: [number, number, number]
  /** When maps omit roughness/metalness */
  defaultRoughness?: number
  defaultMetalness?: number
}

export const HAT_PIECE_MODELS: Record<string, WardrobePieceModelConfig> = {
  'british-helmet': {
    id: 'british-helmet',
    format: 'obj',
    model: '/models/hats/british-helmet/model.obj',
    texturePaths: [
      '/models/hats/british-helmet/textures/Diffuse.png',
      '/models/hats/british-helmet/textures/Normals.png',
    ],
    textureSlots: ['map', 'normalMap'],
    fit: 0.52,
    defaultRoughness: 0.55,
    defaultMetalness: 0.08,
  },
  'felt-cap': {
    id: 'felt-cap',
    format: 'fbx',
    model: '/models/hats/felt-cap/model.fbx',
    texturePaths: ['/models/hats/felt-cap/textures/Hat6.png'],
    textureSlots: ['map'],
    fit: 0.55,
    rotationDeg: [0, 90, -90],
    defaultRoughness: 0.65,
    defaultMetalness: 0.05,
  },
  'french-helmet': {
    id: 'french-helmet',
    format: 'obj',
    model: '/models/hats/french-helmet/model.obj',
    texturePaths: [
      '/models/hats/french-helmet/textures/Diffuse.png',
      '/models/hats/french-helmet/textures/Normals.png',
    ],
    textureSlots: ['map', 'normalMap'],
    fit: 0.52,
    defaultRoughness: 0.55,
    defaultMetalness: 0.08,
  },
  'hat-hip-hop': {
    id: 'hat-hip-hop',
    format: 'obj',
    model: '/models/hats/hat-hip-hop/model.obj',
    texturePaths: [
      '/models/hats/hat-hip-hop/textures/Hat_COL.png',
      '/models/hats/hat-hip-hop/textures/Hat_NOR.png',
    ],
    textureSlots: ['map', 'normalMap'],
    fit: 0.5,
    defaultRoughness: 0.6,
    defaultMetalness: 0.05,
  },
  'sunday-hat': {
    id: 'sunday-hat',
    format: 'fbx',
    model: '/models/hats/sunday-hat/model.fbx',
    texturePaths: [
      '/models/hats/sunday-hat/textures/DefaultMaterial_Base_Color3.png',
      '/models/hats/sunday-hat/textures/DefaultMaterial_Normal_DirectX.png',
      '/models/hats/sunday-hat/textures/DefaultMaterial_Roughness.png',
      '/models/hats/sunday-hat/textures/DefaultMaterial_Metallic.png',
    ],
    textureSlots: ['map', 'normalMap', 'roughnessMap', 'metalnessMap'],
    fit: 0.5,
    rotationDeg: [0, 90, -90],
    defaultMetalness: 1,
    defaultRoughness: 1,
  },
  'victoriques-sunday-hat': {
    id: 'victoriques-sunday-hat',
    format: 'fbx',
    model: '/models/hats/victoriques-sunday-hat/model.fbx',
    texturePaths: ['/models/hats/victoriques-sunday-hat/textures/victorique_sundayB_hat_LP2_DefaultMaterial.png'],
    textureSlots: ['map'],
    fit: 0.52,
    rotationDeg: [0, 90, -90],
    defaultRoughness: 0.62,
    defaultMetalness: 0.06,
  },
  'vintage-helmet': {
    id: 'vintage-helmet',
    format: 'obj',
    model: '/models/hats/vintage-helmet/model.obj',
    texturePaths: [
      '/models/hats/vintage-helmet/textures/Diffuse.jpeg',
      '/models/hats/vintage-helmet/textures/Normals.jpeg',
    ],
    textureSlots: ['map', 'normalMap'],
    fit: 0.5,
    defaultRoughness: 0.55,
    defaultMetalness: 0.08,
  },
}

export const SHOE_PIECE_MODELS: Record<string, WardrobePieceModelConfig> = {
  'retopologized-shoes': {
    id: 'retopologized-shoes',
    format: 'fbx',
    model: '/models/shoes/retopologized-shoes/model.fbx',
    texturePaths: [
      '/models/shoes/retopologized-shoes/textures/Shoe_Diffuse.png',
      '/models/shoes/retopologized-shoes/textures/Shoe_Normal.png',
      '/models/shoes/retopologized-shoes/textures/Shoe_Roughness.png',
      '/models/shoes/retopologized-shoes/textures/Shoe_metalness.png',
    ],
    textureSlots: ['map', 'normalMap', 'roughnessMap', 'metalnessMap'],
    fit: 0.48,
    rotationDeg: [90, 90, -90],
    defaultMetalness: 1,
    defaultRoughness: 1,
  },
  'futuristic-shoe': {
    id: 'futuristic-shoe',
    format: 'fbx',
    model: '/models/shoes/futuristic-shoe/model.fbx',
    texturePaths: [
      '/models/shoes/futuristic-shoe/textures/Used Metal_basecolor.jpeg',
      '/models/shoes/futuristic-shoe/textures/Used Metal_normal.jpeg',
    ],
    textureSlots: ['map', 'normalMap'],
    fit: 0.46,
    rotationDeg: [90, 90, -90],
    defaultRoughness: 0.48,
    defaultMetalness: 0.45,
  },
  'sports-sneaker': {
    id: 'sports-sneaker',
    format: 'glb',
    model: '/models/shoes/sports-sneaker/model.glb',
    texturePaths: [],
    textureSlots: [],
    fit: 0.55,
    rotationDeg: [90, 90, 0],
    defaultRoughness: 0.85,
    defaultMetalness: 0.05,
  },
}
