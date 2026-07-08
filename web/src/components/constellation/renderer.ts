import { Geometry, Mesh, Program, Renderer } from "ogl";
import type { Star } from "@/lib/constellation";

/** Uniform state the scroll narrative drives each frame. */
export interface SceneState {
  /** 0..1 — stars gather from scattered darkness into the field. */
  reveal: number;
  /** 0..1 — morph field layout → per-year error columns. */
  morph: number;
  /** 0..1 — fade stars without predictions (scene 4). */
  dimUnpredicted: number;
  /** Camera zoom (1 = full field). */
  scale: number;
  /** Camera center in normalized field coords. */
  centerX: number;
  centerY: number;
  /** 0..1 global opacity. */
  alpha: number;
}

export const restingScene: SceneState = {
  reveal: 1,
  morph: 0,
  dimUnpredicted: 0,
  scale: 1,
  centerX: 0,
  centerY: 0,
  alpha: 1,
};

const VERTEX = /* glsl */ `
  attribute vec2 aField;
  attribute vec2 aScatter;
  attribute vec2 aYear;
  attribute float aSize;
  attribute float aIntensity;
  attribute float aPredicted;
  attribute float aPhase;

  uniform float uTime;
  uniform float uReveal;
  uniform float uMorph;
  uniform float uDim;
  uniform float uScale;
  uniform vec2 uCenter;
  uniform float uAspect;
  uniform float uDpr;
  uniform float uDrift;

  varying float vIntensity;
  varying float vAlpha;
  varying float vCool;

  void main() {
    vec2 pos = mix(aScatter, aField, uReveal);
    pos = mix(pos, aYear, uMorph);

    // Film-dust drift: slow, tiny, per-star phase. No bounce.
    float drift = 0.004 * uDrift;
    pos += vec2(
      sin(uTime * 0.11 + aPhase * 6.2831) * drift,
      cos(uTime * 0.13 + aPhase * 9.42) * drift
    );

    vec2 view = (pos - uCenter) * uScale;
    view.x /= uAspect;
    gl_Position = vec4(view, 0.0, 1.0);

    float sizeScale = mix(1.0, sqrt(uScale), 0.8);
    gl_PointSize = aSize * uDpr * sizeScale;

    vIntensity = aIntensity;
    float dimmed = mix(1.0, aPredicted, uDim);
    float appear = smoothstep(0.0, 0.6, uReveal + aPhase * 0.25 * uReveal);
    // As the field becomes the model's per-year report, drop the movies it never
    // graded and cool the survivors from tungsten to cyan — reality → machine.
    float graded = step(0.5, aPredicted);
    vAlpha = mix(0.0, dimmed, appear) * mix(1.0, graded, uMorph);
    vCool = uMorph * graded;
  }
`;

const FRAGMENT = /* glsl */ `
  precision highp float;

  uniform float uAlpha;

  varying float vIntensity;
  varying float vAlpha;
  varying float vCool;

  void main() {
    vec2 uv = gl_PointCoord - 0.5;
    float d = length(uv) * 2.0;
    // Soft glow: bright core, long falloff. Additive blending blooms overlaps.
    float core = smoothstep(0.5, 0.0, d);
    float halo = pow(max(1.0 - d, 0.0), 2.6) * 0.55;
    float glow = core * 0.85 + halo;

    // Tungsten amber (reality) → electric cyan (the machine), whitening at the core.
    vec3 warm = mix(vec3(0.94, 0.68, 0.32), vec3(1.0, 0.90, 0.72), vIntensity * core);
    vec3 cool = mix(vec3(0.46, 0.79, 1.0), vec3(0.85, 0.95, 1.0), vIntensity * core);
    vec3 color = mix(warm, cool, vCool);

    // The machine's columns are sparser than the field — lift them so they read.
    float a = glow * (0.55 + vIntensity * 0.45) * vAlpha * uAlpha * (1.0 + vCool * 0.6);
    gl_FragColor = vec4(color * a, a);
  }
`;

/** Imperative ogl wrapper — the React component owns lifecycle and scroll state. */
export class ConstellationRenderer {
  private renderer: Renderer;
  private mesh: Mesh;
  private program: Program;
  private raf = 0;
  private startTime = performance.now();
  scene: SceneState = { ...restingScene };
  /** Set false to pause the drift (reduced motion). */
  drift = true;

  private canvas: HTMLCanvasElement;

  constructor(
    private container: HTMLElement,
    stars: Star[],
  ) {
    // Own canvas per instance: a StrictMode remount after destroy() would
    // otherwise reuse a canvas whose WebGL context is permanently lost.
    this.canvas = document.createElement("canvas");
    this.canvas.style.position = "absolute";
    this.canvas.style.inset = "0";
    this.canvas.style.width = "100%";
    this.canvas.style.height = "100%";
    container.appendChild(this.canvas);

    this.renderer = new Renderer({
      canvas: this.canvas,
      dpr: Math.min(window.devicePixelRatio || 1, 2),
      alpha: true,
      antialias: false,
      depth: false,
      premultipliedAlpha: true,
    });
    const gl = this.renderer.gl;
    gl.clearColor(0, 0, 0, 0);

    const n = stars.length;
    const field = new Float32Array(n * 2);
    const scatter = new Float32Array(n * 2);
    const year = new Float32Array(n * 2);
    const size = new Float32Array(n);
    const intensity = new Float32Array(n);
    const predicted = new Float32Array(n);
    const phase = new Float32Array(n);

    // Deterministic pseudo-random scatter so SSR/CSR and re-mounts agree.
    let seed = 987654321;
    const rand = () => {
      seed = (seed * 1103515245 + 12345) & 0x7fffffff;
      return seed / 0x7fffffff;
    };

    stars.forEach((s, i) => {
      field[i * 2] = s.fieldX;
      field[i * 2 + 1] = s.fieldY;
      const angle = rand() * Math.PI * 2;
      const radius = 1.4 + rand() * 1.2;
      scatter[i * 2] = Math.cos(angle) * radius;
      scatter[i * 2 + 1] = Math.sin(angle) * radius;
      year[i * 2] = s.yearX;
      year[i * 2 + 1] = s.yearY;
      size[i] = s.size;
      intensity[i] = s.intensity;
      predicted[i] = s.hasPrediction ? 1 : 0.06;
      phase[i] = rand();
    });

    const geometry = new Geometry(gl, {
      aField: { size: 2, data: field },
      aScatter: { size: 2, data: scatter },
      aYear: { size: 2, data: year },
      aSize: { size: 1, data: size },
      aIntensity: { size: 1, data: intensity },
      aPredicted: { size: 1, data: predicted },
      aPhase: { size: 1, data: phase },
    });

    this.program = new Program(gl, {
      vertex: VERTEX,
      fragment: FRAGMENT,
      transparent: true,
      depthTest: false,
      depthWrite: false,
      uniforms: {
        uTime: { value: 0 },
        uReveal: { value: 1 },
        uMorph: { value: 0 },
        uDim: { value: 0 },
        uScale: { value: 1 },
        uCenter: { value: [0, 0] },
        uAspect: { value: 1 },
        uDpr: { value: Math.min(window.devicePixelRatio || 1, 2) },
        uDrift: { value: 1 },
        uAlpha: { value: 1 },
      },
    });
    // Additive blending: overlapping stars bloom.
    this.program.setBlendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);

    this.mesh = new Mesh(gl, { mode: gl.POINTS, geometry, program: this.program });

    this.resize();
    this.loop();
  }

  resize = (): void => {
    const { clientWidth, clientHeight } = this.canvas.parentElement ?? this.canvas;
    if (clientWidth === 0 || clientHeight === 0) return;
    this.renderer.setSize(clientWidth, clientHeight);
    this.program.uniforms.uAspect.value = clientWidth / clientHeight;
  };

  /** Project normalized field coords to CSS pixel coords within the canvas. */
  project(x: number, y: number): { x: number; y: number } {
    const { scale, centerX, centerY } = this.scene;
    const aspect =
      (this.canvas.parentElement?.clientWidth ?? 1) /
      (this.canvas.parentElement?.clientHeight ?? 1);
    const vx = ((x - centerX) * scale) / aspect;
    const vy = (y - centerY) * scale;
    const w = this.canvas.parentElement?.clientWidth ?? 0;
    const h = this.canvas.parentElement?.clientHeight ?? 0;
    return { x: ((vx + 1) / 2) * w, y: ((1 - vy) / 2) * h };
  }

  /** Inverse of project — CSS pixel coords to normalized field coords. */
  unproject(px: number, py: number): { x: number; y: number } {
    const { scale, centerX, centerY } = this.scene;
    const w = this.canvas.parentElement?.clientWidth ?? 1;
    const h = this.canvas.parentElement?.clientHeight ?? 1;
    const aspect = w / h;
    const vx = (px / w) * 2 - 1;
    const vy = 1 - (py / h) * 2;
    return { x: (vx * aspect) / scale + centerX, y: vy / scale + centerY };
  }

  private loop = (): void => {
    this.raf = requestAnimationFrame(this.loop);
    const u = this.program.uniforms;
    u.uTime.value = (performance.now() - this.startTime) / 1000;
    u.uReveal.value = this.scene.reveal;
    u.uMorph.value = this.scene.morph;
    u.uDim.value = this.scene.dimUnpredicted;
    u.uScale.value = this.scene.scale;
    u.uCenter.value = [this.scene.centerX, this.scene.centerY];
    u.uAlpha.value = this.scene.alpha;
    u.uDrift.value = this.drift ? 1 : 0;
    try {
      this.renderer.render({ scene: this.mesh });
    } catch (e) {
      cancelAnimationFrame(this.raf);
      console.error(
        "constellation render failed :: " +
          ((e as Error).stack ?? "").split("\n").slice(0, 6).join(" || "),
      );
    }
  };

  destroy(): void {
    cancelAnimationFrame(this.raf);
    const ext = this.renderer.gl.getExtension("WEBGL_lose_context");
    ext?.loseContext();
    this.canvas.remove();
  }
}

export function webglSupported(): boolean {
  try {
    const canvas = document.createElement("canvas");
    return Boolean(
      canvas.getContext("webgl2") ?? canvas.getContext("webgl"),
    );
  } catch {
    return false;
  }
}
