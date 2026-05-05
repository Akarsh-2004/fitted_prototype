import { Canvas, useLoader } from '@react-three/fiber'
import { OrbitControls, useTexture } from '@react-three/drei'
import { Suspense, useEffect, useLayoutEffect, useMemo } from 'react'
import * as THREE from 'three'
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader.js'
import type { TopPalette } from '../../types/wardrobe'

const MODEL_URL = '/models/oversized-tshirt/oversized-tshirt.obj'

const TEX = [
  '/models/oversized-tshirt/textures/oversized-tshirt_diffuse_1001.png',
  '/models/oversized-tshirt/textures/oversized-tshirt_normal_1001.png',
  '/models/oversized-tshirt/textures/oversized-tshirt_roughness_1001.png',
  '/models/oversized-tshirt/textures/oversized-tshirt_metalness_1001.png',
] as const

function ShirtObject({ palette }: { palette: TopPalette }) {
  const src = useLoader(OBJLoader, MODEL_URL)
  const root = useMemo(() => src.clone(true), [src])
  const [map, normalMap, roughnessMap, metalnessMap] = useTexture([...TEX])

  useLayoutEffect(() => {
    map.colorSpace = THREE.SRGBColorSpace

    root.traverse((obj) => {
      const mesh = obj as THREE.Mesh
      if (!mesh.isMesh) return
      mesh.material = new THREE.MeshStandardMaterial({
        map,
        normalMap,
        roughnessMap,
        metalnessMap,
        metalness: 1,
        roughness: 1,
        color: new THREE.Color(palette.body),
      })
      const mat = mesh.material as THREE.MeshStandardMaterial
      mat.normalScale = new THREE.Vector2(0.55, 0.55)
    })

    const box = new THREE.Box3().setFromObject(root)
    const size = box.getSize(new THREE.Vector3())
    const max = Math.max(size.x, size.y, size.z, 1e-6)
    const s = 0.62 / max
    root.scale.setScalar(s)
    box.setFromObject(root)
    const center = box.getCenter(new THREE.Vector3())
    root.position.sub(center)
  // eslint-disable-next-line react-hooks/exhaustive-deps -- palette.body synced in useEffect; omit here to avoid re-fitting mesh
  }, [root, map, normalMap, roughnessMap, metalnessMap])

  useEffect(() => {
    root.traverse((obj) => {
      const mesh = obj as THREE.Mesh
      if (!mesh.isMesh || !mesh.material) return
      ;(mesh.material as THREE.MeshStandardMaterial).color.set(palette.body)
    })
  }, [palette.body, root])

  return <primitive object={root} />
}

function Scene({ palette }: { palette: TopPalette }) {
  return (
    <>
      <ambientLight intensity={0.52} />
      <directionalLight position={[3.2, 4.5, 5]} intensity={1.05} />
      <directionalLight position={[-4, 2.5, -3]} intensity={0.38} />
      <hemisphereLight args={['#f5f4f2', '#8a8580', 0.35]} />
      <ShirtObject palette={palette} />
      <OrbitControls
        makeDefault
        enablePan={false}
        autoRotate={false}
        minPolarAngle={Math.PI / 4}
        maxPolarAngle={Math.PI - Math.PI / 5}
        minDistance={0.55}
        maxDistance={2.4}
        enableDamping
        dampingFactor={0.08}
      />
    </>
  )
}

type Props = {
  palette: TopPalette
  className?: string
}

/**
 * Centered Three.js preview: drag to orbit, scroll/pinch to zoom (no auto-spin).
 */
export function OversizedTeeModel({ palette, className }: Props) {
  return (
    <div
      className={`piece-3d-wrap ${className ?? ''}`.trim()}
      title="Drag to rotate · scroll or pinch to zoom"
    >
      <Canvas
        camera={{ position: [0, 0.05, 1.05], fov: 42, near: 0.01, far: 50 }}
        gl={{ alpha: true, antialias: true, powerPreference: 'high-performance' }}
        dpr={[1, 2]}
      >
        <Suspense fallback={null}>
          <Scene palette={palette} />
        </Suspense>
      </Canvas>
    </div>
  )
}

export function OversizedTeeGraphic({ palette, className }: Props) {
  return <OversizedTeeModel palette={palette} className={className} />
}
