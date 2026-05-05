export type TryonMeta = {
  mask_quality: number
  processing_time: number
  used_fallback: boolean
  warp_mode: 'affine' | 'tps'
  request_key: string
}

export type TryonResponse = {
  cutout: string
  composite: string
  mask: string
  meta: TryonMeta
}

const API_BASE = (import.meta.env.VITE_TRYON_API_BASE as string | undefined) ?? 'http://127.0.0.1:8000'

const toDataUrl = (base64: string) => `data:image/png;base64,${base64}`

export async function processSingle(
  file: File,
  params?: { edgeFeather?: number; morphKernel?: number; smoothness?: number },
): Promise<{ cutoutSrc: string; compositeSrc: string; maskSrc: string; meta: TryonMeta }> {
  const form = new FormData()
  form.append('image', file)
  form.append('edge_feather', String(params?.edgeFeather ?? 3))
  form.append('morph_kernel', String(params?.morphKernel ?? 3))
  form.append('smoothness', String(params?.smoothness ?? 0.5))

  const res = await fetch(`${API_BASE}/process-single`, { method: 'POST', body: form })
  if (!res.ok) throw new Error('Processing failed')
  const json = (await res.json()) as TryonResponse
  return {
    cutoutSrc: toDataUrl(json.cutout),
    compositeSrc: toDataUrl(json.composite),
    maskSrc: toDataUrl(json.mask),
    meta: json.meta,
  }
}

export async function refineSingle(input: {
  requestKey: string
  edgeFeather: number
  morphKernel: number
  smoothness: number
}): Promise<{ cutoutSrc: string; compositeSrc: string; maskSrc: string; meta: TryonMeta }> {
  const form = new FormData()
  form.append('request_key', input.requestKey)
  form.append('edge_feather', String(input.edgeFeather))
  form.append('morph_kernel', String(input.morphKernel))
  form.append('smoothness', String(input.smoothness))

  const res = await fetch(`${API_BASE}/refine-single`, { method: 'POST', body: form })
  if (!res.ok) throw new Error('Refine failed')
  const json = (await res.json()) as TryonResponse
  return {
    cutoutSrc: toDataUrl(json.cutout),
    compositeSrc: toDataUrl(json.composite),
    maskSrc: toDataUrl(json.mask),
    meta: json.meta,
  }
}
