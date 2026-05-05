import { Canvas, invalidate, useLoader } from '@react-three/fiber'
import { OrbitControls, useTexture } from '@react-three/drei'
import { Suspense, useEffect, useLayoutEffect, useMemo } from 'react'
import * as THREE from 'three'
import { FBXLoader } from 'three/examples/jsm/loaders/FBXLoader.js'
import type { BottomPalette } from '../../types/wardrobe'

const MODEL_URL = '/models/baggy-jeans/JNCO_Twin_Cannon_Model.fbx'

const TEX = [
  '/models/baggy-jeans/textures/DefaultMaterial_Base_color.png',
  '/models/baggy-jeans/textures/DefaultMaterial_Normal_OpenGL.png',
  '/models/baggy-jeans/textures/DefaultMaterial_Roughness.png',
  '/models/baggy-jeans/textures/DefaultMaterial_Metallic.png',
] as const

/** Applied before fitting: align FBX so legs run vertically (Y) and the front faces the camera (+Z). */
function straightenJeansRoot(root: THREE.Object3D) {
  root.position.set(0, 0, 0)
  root.scale.set(1, 1, 1)
  root.rotation.order = 'YXZ'
  root.rotation.set(
    THREE.MathUtils.degToRad(-90),
    THREE.MathUtils.degToRad(90),
    THREE.MathUtils.degToRad(-90),
  )
  root.updateMatrixWorld(true)
}

function JeansObject({ palette }: { palette: BottomPalette }) {
  const src = useLoader(FBXLoader, MODEL_URL)
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
        color: new THREE.Color(palette.leg),
      })
      const mat = mesh.material as THREE.MeshStandardMaterial
      mat.normalScale = new THREE.Vector2(0.5, 0.5)
    })

    straightenJeansRoot(root)

    const box = new THREE.Box3().setFromObject(root)
    const size = box.getSize(new THREE.Vector3())
    const max = Math.max(size.x, size.y, size.z, 1e-6)
    const s = 0.5 / max
    root.scale.multiplyScalar(s)
    box.setFromObject(root)
    const center = box.getCenter(new THREE.Vector3())
    root.position.sub(center)
    // eslint-disable-next-line react-hooks/exhaustive-deps -- palette.leg synced in useEffect; omit here to avoid re-fitting mesh
  }, [root, map, normalMap, roughnessMap, metalnessMap])

  useEffect(() => {
    const c = new THREE.Color(palette.leg)
    root.traverse((obj) => {
      const mesh = obj as THREE.Mesh
      if (!mesh.isMesh || !mesh.material) return
      const mats = Array.isArray(mesh.material) ? mesh.material : [mesh.material]
      for (const raw of mats) {
        if (!(raw instanceof THREE.MeshStandardMaterial)) continue
        const m = raw
        m.color.copy(c)
        m.emissive.copy(c)
        m.emissiveIntensity = 0.14
        m.needsUpdate = true
      }
    })
    invalidate()
  }, [palette, root])

  return <primitive object={root} />
}

function Scene({ palette }: { palette: BottomPalette }) {
  return (
    <>
      <ambientLight intensity={0.52} />
      <directionalLight position={[3.2, 4.5, 5]} intensity={1.05} />
      <directionalLight position={[-4, 2.5, -3]} intensity={0.38} />
      <hemisphereLight args={['#f5f4f2', '#8a8580', 0.35]} />
      <JeansObject palette={palette} />
      <OrbitControls
        makeDefault
        enablePan={false}
        autoRotate={false}
        minPolarAngle={Math.PI / 4}
        maxPolarAngle={Math.PI - Math.PI / 5}
        minDistance={0.5}
        maxDistance={2.2}
        enableDamping
        dampingFactor={0.08}
      />
    </>
  )
}

type Props = {
  palette: BottomPalette
  className?: string
}

/**
 * Centered FBX + PBR: drag to orbit, scroll/pinch to zoom (no auto-spin).
 */
export function BaggyJeansModel({ palette, className }: Props) {
  return (
    <div
      className={`piece-3d-wrap piece-3d-wrap--bottom ${className ?? ''}`.trim()}
      title="Drag to rotate · scroll or pinch to zoom"
    >
      <Canvas
        camera={{ position: [0, 0.12, 1.02], fov: 40, near: 0.01, far: 80 }}
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

export function BaggyJeansGraphic({ palette, className }: Props) {
  return <BaggyJeansModel palette={palette} className={className} />
}
