import { Canvas, useLoader } from '@react-three/fiber'
import { OrbitControls, useGLTF, useTexture } from '@react-three/drei'
import { Suspense, useEffect, useLayoutEffect, useMemo } from 'react'
import * as THREE from 'three'
import { FBXLoader } from 'three/examples/jsm/loaders/FBXLoader.js'
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader.js'
import type { WardrobePieceModelConfig } from './wardrobePieceModels'

function applyRootOrientation(root: THREE.Object3D, rotationDeg?: [number, number, number]) {
  if (!rotationDeg) return
  root.rotation.order = 'YXZ'
  root.rotation.set(
    THREE.MathUtils.degToRad(rotationDeg[0]),
    THREE.MathUtils.degToRad(rotationDeg[1]),
    THREE.MathUtils.degToRad(rotationDeg[2]),
  )
}

function fitAndCenter(root: THREE.Object3D, fit: number) {
  const box = new THREE.Box3().setFromObject(root)
  const size = box.getSize(new THREE.Vector3())
  const max = Math.max(size.x, size.y, size.z, 1e-6)
  const s = fit / max
  root.scale.multiplyScalar(s)
  box.setFromObject(root)
  const center = box.getCenter(new THREE.Vector3())
  root.position.sub(center)
}

function ObjMesh({ config, tint }: { config: WardrobePieceModelConfig; tint: string }) {
  const src = useLoader(OBJLoader, config.model)
  const root = useMemo(() => src.clone(true), [src])
  const texList = useTexture(config.texturePaths)
  const rough = config.defaultRoughness ?? 0.6
  const metal = config.defaultMetalness ?? 0.1

  useLayoutEffect(() => {
    const maps: Partial<Record<string, THREE.Texture>> = {}
    config.textureSlots.forEach((slot, i) => {
      const t = texList[i]!
      if (slot === 'map') t.colorSpace = THREE.SRGBColorSpace
      maps[slot] = t
    })

    root.traverse((obj) => {
      const mesh = obj as THREE.Mesh
      if (!mesh.isMesh) return
      mesh.material = new THREE.MeshStandardMaterial({
        map: maps.map,
        normalMap: maps.normalMap,
        roughnessMap: maps.roughnessMap,
        metalnessMap: maps.metalnessMap,
        metalness: maps.metalnessMap ? 1 : metal,
        roughness: maps.roughnessMap ? 1 : rough,
        color: new THREE.Color(tint),
      })
      const mat = mesh.material as THREE.MeshStandardMaterial
      if (maps.normalMap) mat.normalScale = new THREE.Vector2(0.45, 0.45)
    })

    root.position.set(0, 0, 0)
    root.scale.set(1, 1, 1)
    root.rotation.set(0, 0, 0)
    applyRootOrientation(root, config.rotationDeg)
    root.updateMatrixWorld(true)
    fitAndCenter(root, config.fit)
    // eslint-disable-next-line react-hooks/exhaustive-deps -- tint in separate effect
  }, [root, texList, config])

  useEffect(() => {
    root.traverse((obj) => {
      const mesh = obj as THREE.Mesh
      if (!mesh.isMesh || !mesh.material) return
      ;(mesh.material as THREE.MeshStandardMaterial).color.set(tint)
    })
  }, [tint, root])

  return <primitive object={root} />
}

function FbxMesh({ config, tint }: { config: WardrobePieceModelConfig; tint: string }) {
  const src = useLoader(FBXLoader, config.model)
  const root = useMemo(() => src.clone(true), [src])
  const texList = useTexture(config.texturePaths)
  const rough = config.defaultRoughness ?? 0.6
  const metal = config.defaultMetalness ?? 0.1

  useLayoutEffect(() => {
    const maps: Partial<Record<string, THREE.Texture>> = {}
    config.textureSlots.forEach((slot, i) => {
      const t = texList[i]!
      if (slot === 'map') t.colorSpace = THREE.SRGBColorSpace
      maps[slot] = t
    })

    root.traverse((obj) => {
      const mesh = obj as THREE.Mesh
      if (!mesh.isMesh) return
      mesh.material = new THREE.MeshStandardMaterial({
        map: maps.map,
        normalMap: maps.normalMap,
        roughnessMap: maps.roughnessMap,
        metalnessMap: maps.metalnessMap,
        metalness: maps.metalnessMap ? 1 : metal,
        roughness: maps.roughnessMap ? 1 : rough,
        color: new THREE.Color(tint),
      })
      const mat = mesh.material as THREE.MeshStandardMaterial
      if (maps.normalMap) mat.normalScale = new THREE.Vector2(0.45, 0.45)
    })

    root.position.set(0, 0, 0)
    root.scale.set(1, 1, 1)
    root.rotation.set(0, 0, 0)
    applyRootOrientation(root, config.rotationDeg)
    root.updateMatrixWorld(true)
    fitAndCenter(root, config.fit)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [root, texList, config])

  useEffect(() => {
    root.traverse((obj) => {
      const mesh = obj as THREE.Mesh
      if (!mesh.isMesh || !mesh.material) return
      ;(mesh.material as THREE.MeshStandardMaterial).color.set(tint)
    })
  }, [tint, root])

  return <primitive object={root} />
}

function GlbMesh({ config, tint }: { config: WardrobePieceModelConfig; tint: string }) {
  const { scene } = useGLTF(config.model)
  const root = useMemo(() => scene.clone(true), [scene])

  useLayoutEffect(() => {
    root.traverse((obj) => {
      const mesh = obj as THREE.Mesh
      if (!mesh.isMesh || !mesh.material) return
      const m = mesh.material
      if (Array.isArray(m)) {
        m.forEach((mat) => tintMaterial(mat, tint))
      } else {
        tintMaterial(m, tint)
      }
    })
    root.position.set(0, 0, 0)
    root.scale.set(1, 1, 1)
    root.rotation.set(0, 0, 0)
    applyRootOrientation(root, config.rotationDeg)
    root.updateMatrixWorld(true)
    fitAndCenter(root, config.fit)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [root, config])

  useEffect(() => {
    root.traverse((obj) => {
      const mesh = obj as THREE.Mesh
      if (!mesh.isMesh || !mesh.material) return
      const m = mesh.material
      if (Array.isArray(m)) m.forEach((mat) => tintMaterial(mat, tint))
      else tintMaterial(m, tint)
    })
  }, [tint, root])

  return <primitive object={root} />
}

function tintMaterial(mat: THREE.Material, tint: string) {
  if (mat instanceof THREE.MeshStandardMaterial || mat instanceof THREE.MeshPhysicalMaterial) {
    mat.color.set(tint)
    return
  }
  if (mat instanceof THREE.MeshBasicMaterial) {
    mat.color.set(tint)
  }
}

function PieceMesh({ config, tint }: { config: WardrobePieceModelConfig; tint: string }) {
  if (config.format === 'obj') return <ObjMesh config={config} tint={tint} />
  if (config.format === 'fbx') return <FbxMesh config={config} tint={tint} />
  return <GlbMesh config={config} tint={tint} />
}

function Scene({ config, tint }: { config: WardrobePieceModelConfig; tint: string }) {
  return (
    <>
      <ambientLight intensity={0.52} />
      <directionalLight position={[3.2, 4.5, 5]} intensity={1.05} />
      <directionalLight position={[-4, 2.5, -3]} intensity={0.38} />
      <hemisphereLight args={['#f5f4f2', '#8a8580', 0.35]} />
      <PieceMesh config={config} tint={tint} />
      <OrbitControls
        makeDefault
        enablePan={false}
        autoRotate={false}
        minPolarAngle={Math.PI / 4}
        maxPolarAngle={Math.PI - Math.PI / 5}
        minDistance={0.45}
        maxDistance={2.5}
        enableDamping
        dampingFactor={0.08}
      />
    </>
  )
}

export type PieceVariant = 'hat' | 'shoes'

const CAMERA: Record<
  PieceVariant,
  { position: [number, number, number]; fov: number }
> = {
  hat: { position: [0, 0.02, 0.92], fov: 42 },
  shoes: { position: [0, 0.06, 1.02], fov: 40 },
}

type Props = {
  config: WardrobePieceModelConfig
  tint: string
  variant: PieceVariant
  className?: string
}

export function WardrobePieceModel3D({ config, tint, variant, className }: Props) {
  const cam = CAMERA[variant]
  const wrapClass =
    variant === 'hat'
      ? `piece-3d-wrap piece-3d-wrap--hat ${className ?? ''}`.trim()
      : `piece-3d-wrap piece-3d-wrap--shoes ${className ?? ''}`.trim()

  return (
    <div className={wrapClass} title="Drag to rotate · scroll or pinch to zoom">
      <Canvas
        camera={{ position: cam.position, fov: cam.fov, near: 0.01, far: 80 }}
        gl={{ alpha: true, antialias: true, powerPreference: 'high-performance' }}
        dpr={[1, 2]}
      >
        <Suspense key={config.id} fallback={null}>
          <Scene config={config} tint={tint} />
        </Suspense>
      </Canvas>
    </div>
  )
}

useGLTF.preload('/models/shoes/sports-sneaker/model.glb')
