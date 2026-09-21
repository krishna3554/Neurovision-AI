"use client";
import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import * as THREE from "three";
import { useMemo } from "react";

export type Mesh = { vertices: number[][]; faces: number[][] };

function useCentered(brain: Mesh, lesion: Mesh) {
  return useMemo(() => {
    const mk = (m: Mesh) => {
      const g = new THREE.BufferGeometry();
      g.setAttribute("position", new THREE.Float32BufferAttribute(m.vertices.flat(), 3));
      g.setIndex(m.faces.flat());
      g.computeVertexNormals();
      return g;
    };
    const gb = mk(brain);
    const gl = mk(lesion);
    // Compute the brain's bounding-box centre ONCE and apply the same
    // translation to both so the lesion stays inside the brain.
    gb.computeBoundingBox();
    const c = new THREE.Vector3();
    gb.boundingBox!.getCenter(c);
    gb.translate(-c.x, -c.y, -c.z);
    gl.translate(-c.x, -c.y, -c.z);
    return { gb, gl };
  }, [brain, lesion]);
}

export default function BrainViewer3D({ brain, lesion }: { brain: Mesh; lesion: Mesh }) {
  const { gb, gl } = useCentered(brain, lesion);
  return (
    <Canvas camera={{ position: [0, 0, 250], fov: 45 }}>
      <ambientLight intensity={0.7} />
      <directionalLight position={[100, 100, 100]} />
      <mesh geometry={gb}>
        <meshStandardMaterial color="#9fd3e8" transparent opacity={0.35} />
      </mesh>
      <mesh geometry={gl}>
        <meshStandardMaterial color="#e11d48" emissive="#e11d48" emissiveIntensity={0.6} />
      </mesh>
      <OrbitControls enablePan enableZoom />
    </Canvas>
  );
}
