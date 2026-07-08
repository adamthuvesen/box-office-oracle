"use client";

import {
  useCallback,
  useEffect,
  useEffectEvent,
  useMemo,
  useRef,
  useState,
  useSyncExternalStore,
} from "react";
import Image from "next/image";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useReducedMotion } from "motion/react";
import {
  StarGrid,
  segment,
  type Star,
  type StarField,
} from "@/lib/constellation";
import {
  ConstellationRenderer,
  webglSupported,
  type SceneState,
} from "@/components/constellation/renderer";
import { dollarsCompact } from "@/lib/format";
import { posterUrl } from "@/lib/types";

interface HeroProps {
  field: StarField;
  totalGross: number;
}

/** Piecewise-linear map of scroll progress to an opacity. */
function fade(p: number, stops: number[], values: number[]): number {
  if (p <= stops[0]) return values[0];
  for (let i = 1; i < stops.length; i++) {
    if (p <= stops[i]) {
      const t = (p - stops[i - 1]) / (stops[i] - stops[i - 1]);
      return values[i - 1] + (values[i] - values[i - 1]) * t;
    }
  }
  return values[values.length - 1];
}

/**
 * The signature: ~6,100 movies as glowing particles in a budget×gross field,
 * walked through a five-scene scroll narrative, settling into free explore.
 */
let webglCache: boolean | null = null;
const webglSnapshot = () => (webglCache ??= webglSupported());

export function ConstellationHero({ field, totalGross }: HeroProps) {
  const reducedMotion = useReducedMotion();
  const webgl = useSyncExternalStore(
    () => () => {},
    webglSnapshot,
    () => null,
  );

  if (webgl === null) {
    // One frame of server/first-paint markup: the headline, no canvas yet.
    return (
      <section className="flex min-h-[70svh] items-end px-6 pb-16">
        <Headline field={field} totalGross={totalGross} />
      </section>
    );
  }

  if (reducedMotion || !webgl) {
    return <StaticConstellation field={field} totalGross={totalGross} />;
  }

  return <ScrollNarrative field={field} totalGross={totalGross} />;
}

/* ------------------------------------------------------------------ */

/**
 * The five settled compositions, as points in the same 0→1 progress space the
 * scene math already speaks. The pager eases between these; it never leaves the
 * viewer parked in an in-between state. Values chosen so each rest lands on a
 * fully-formed scene (field, zoom+callout, beam, per-year morph, free explore).
 */
const REST = [0.0, 0.33, 0.48, 0.74, 1.0];
const LAST = REST.length - 1;
const STEP_MS = 2200; // one gesture → one scene, glided slowly enough to watch

/**
 * Ease-out cubic: kicks off at high velocity the instant you scroll (snappy
 * response, no dead start), then a long, slow deceleration so most of the
 * animation plays out at a watchable pace. Quick to start, unhurried to finish.
 */
const easeOutCubic = (t: number) => 1 - Math.pow(1 - t, 3);

/**
 * Linear 0..1 ramp. Used for spatial motion (camera, morph, beam draw) so it
 * tracks progress evenly and spans the whole transition. `segment()` layers its
 * own easeOutQuint on top of the tween's ease — fine for opacity, but on motion
 * that double-ease front-loads everything into the first fraction and leaves a
 * dead tail. A linear ramp keeps the movement smooth from rest point to rest point.
 */
const ramp = (p: number, a: number, b: number) =>
  Math.max(0, Math.min(1, (p - a) / (b - a)));

function ScrollNarrative({ field, totalGross }: HeroProps) {
  const router = useRouter();
  const canvasHostRef = useRef<HTMLDivElement>(null);
  const stageRef = useRef<HTMLDivElement>(null);
  const rendererRef = useRef<ConstellationRenderer | null>(null);
  const beamRef = useRef<SVGSVGElement>(null);
  const actualLabelRef = useRef<HTMLDivElement>(null);
  const predLabelRef = useRef<HTMLDivElement>(null);
  const calloutRef = useRef<HTMLDivElement>(null);
  // Scene overlays faded imperatively in applyScroll — one source of truth
  // with the canvas uniforms, no separate scroll subscriptions to drift.
  const cueRef = useRef<HTMLDivElement>(null);
  const capARef = useRef<HTMLDivElement>(null);
  const capBRef = useRef<HTMLDivElement>(null);
  const capCRef = useRef<HTMLDivElement>(null);
  const capDRef = useRef<HTMLDivElement>(null);
  const axesRef = useRef<HTMLDivElement>(null);
  const yearAxisRef = useRef<HTMLDivElement>(null);
  const [free, setFree] = useState(false);
  const [hovered, setHovered] = useState<Star | null>(null);
  const [hoverPos, setHoverPos] = useState({ x: 0, y: 0 });
  const [selected, setSelected] = useState<number>(-1);
  const [sceneIndex, setSceneIndex] = useState(0);

  const featured = field.stars[field.featuredIndex];
  const grid = useMemo(() => new StarGrid(field.stars), [field.stars]);

  // The gap between what the movie made and what the model guessed — the payload
  // of the beam scene.
  const featuredMiss =
    featured.predicted != null ? featured.gross - featured.predicted : 0;
  const missLabel =
    featuredMiss >= 0
      ? `undershot by ${dollarsCompact(featuredMiss)}`
      : `overshot by ${dollarsCompact(-featuredMiss)}`;

  // Imperative animation state — lives outside React's render cycle. `progress`
  // is owned by the pager tween; `intro` by the load-formation; both compose in
  // applyScroll so the two never fight over a frame.
  const animRef = useRef({
    intro: 0,
    progress: 0,
    from: 0,
    to: 0,
    startT: 0,
    dur: STEP_MS,
    tweening: false,
    introStart: 0,
  });
  const sceneRef = useRef(0);
  const goToRef = useRef<(target: number) => void>(() => {});

  const applyScroll = useEffectEvent((p: number) => {
    const r = rendererRef.current;
    if (!r) return;

    const reveal = Math.max(animRef.current.intro, segment(p, 0.0, 0.16));
      // Each band opens right at the rest point it departs and closes at the one
      // it arrives — motion begins within a frame of the scroll and spans the
      // whole transition (linear ramp, so no dead lead-in and no dead tail).
      const zoomIn = ramp(p, 0.04, 0.32);
      const zoomOut = ramp(p, 0.5, 0.68);
      const morph = ramp(p, 0.52, 0.72) - ramp(p, 0.76, 0.96);
      const dim = morph;
      const zoom = zoomIn * (1 - zoomOut);

      const scene: SceneState = {
        reveal,
        morph,
        dimUnpredicted: dim,
        scale: 1 + zoom * 2.2,
        centerX: featured.fieldX * zoom * 0.92,
        centerY: featured.fieldY * zoom * 0.92,
        alpha: 1,
      };
      r.scene = scene;

      // Featured-star callout (scene 1 zoom) and prediction beam (scene 2):
      // positioned imperatively from projected coords — no re-renders. The
      // callout clears before the beam's labels draw in — both anchor to the
      // same star, so they must not share the frame.
      const callout = calloutRef.current;
      const beam = beamRef.current;
      const calloutAlpha =
        segment(p, 0.22, 0.3) - segment(p, 0.35, 0.41);
      if (callout) {
        const pt = r.project(featured.fieldX, featured.fieldY);
        callout.style.opacity = String(calloutAlpha);
        callout.style.transform = `translate(${pt.x + 18}px, ${pt.y - 12}px)`;
      }
      if (beam && featured.predictedFieldY != null) {
        const beamAlpha = segment(p, 0.35, 0.42) - segment(p, 0.5, 0.58);
        const grow = ramp(p, 0.35, 0.48);
        beam.style.opacity = String(beamAlpha);
        const actualLabel = actualLabelRef.current;
        const predLabel = predLabelRef.current;
        if (beamAlpha > 0) {
          const a = r.project(featured.fieldX, featured.fieldY);
          const b = r.project(featured.fieldX, featured.predictedFieldY);
          // Projector origin, off the left edge — the beam fans open from it.
          const origin = r.project(-1.5, (featured.fieldY + featured.predictedFieldY) / 2);
          const ax = origin.x + (a.x - origin.x) * grow;
          const ay = origin.y + (a.y - origin.y) * grow;
          const bx = origin.x + (b.x - origin.x) * grow;
          const by = origin.y + (b.y - origin.y) * grow;
          const poly = beam.querySelector("polygon");
          const mouth = beam.querySelector<SVGLineElement>("line.beam-mouth");
          const dotA = beam.querySelector<SVGCircleElement>("circle.beam-actual-dot");
          const dotB = beam.querySelector<SVGCircleElement>("circle.beam-pred-dot");
          poly?.setAttribute(
            "points",
            `${origin.x},${origin.y} ${ax},${ay} ${bx},${by}`,
          );
          // The interval mouth — the model's spread between the two outcomes.
          if (mouth) {
            mouth.setAttribute("x1", String(ax));
            mouth.setAttribute("y1", String(ay));
            mouth.setAttribute("x2", String(bx));
            mouth.setAttribute("y2", String(by));
            mouth.style.opacity = String(grow);
          }
          dotA?.setAttribute("cx", String(ax));
          dotA?.setAttribute("cy", String(ay));
          dotB?.setAttribute("cx", String(bx));
          dotB?.setAttribute("cy", String(by));
          if (actualLabel) {
            actualLabel.style.opacity = String(grow);
            actualLabel.style.transform = `translate(${a.x + 22}px, ${a.y - 34}px)`;
          }
          if (predLabel) {
            predLabel.style.opacity = String(grow);
            predLabel.style.transform = `translate(${b.x + 22}px, ${b.y - 14}px)`;
          }
        } else {
          if (actualLabel) actualLabel.style.opacity = "0";
          if (predLabel) predLabel.style.opacity = "0";
        }
      }

      const setOpacity = (el: HTMLElement | null, v: number) => {
        if (el) el.style.opacity = String(v);
      };
      setOpacity(cueRef.current, fade(p, [0, 0.04], [1, 0]));
      setOpacity(capARef.current, fade(p, [0, 0.02, 0.12], [1, 1, 0]));
      setOpacity(capBRef.current, fade(p, [0.22, 0.3, 0.5, 0.58], [0, 1, 1, 0]));
      setOpacity(capCRef.current, fade(p, [0.6, 0.68, 0.78, 0.86], [0, 1, 1, 0]));
      setOpacity(capDRef.current, fade(p, [0.88, 0.96], [0, 1]));
      setOpacity(
        axesRef.current,
        fade(p, [0, 0.1, 0.48, 0.62, 0.76, 0.92], [0, 0.7, 0.7, 0, 0, 0.7]),
      );
      setOpacity(
        yearAxisRef.current,
        fade(p, [0.56, 0.66, 0.8, 0.9], [0, 0.8, 0.8, 0]),
      );
  });

  useEffect(() => {
    const host = canvasHostRef.current;
    const stage = stageRef.current;
    if (!host || !stage) return;
    const renderer = new ConstellationRenderer(host, field.stars);
    renderer.scene.reveal = 0;
    rendererRef.current = renderer;

    const a = animRef.current;
    a.intro = 0;
    a.progress = 0;
    a.tweening = false;
    a.introStart = performance.now();
    sceneRef.current = 0;

    // Single rAF driver: advances the load-formation and the active pager tween
    // each frame, then paints once. Sleeps when nothing is animating.
    let raf = 0;
    let running = false;
    const tick = () => {
      const now = performance.now();
      const it = Math.min((now - a.introStart) / 1600, 1);
      a.intro = 1 - Math.pow(1 - it, 5);
      if (a.tweening) {
        const t = Math.min((now - a.startT) / a.dur, 1);
        a.progress = a.from + (a.to - a.from) * easeOutCubic(t);
        if (t >= 1) {
          a.tweening = false;
          if (sceneRef.current === LAST) setFree(true);
        }
      }
      applyScroll(a.progress);
      if (it < 1 || a.tweening) {
        raf = requestAnimationFrame(tick);
      } else {
        running = false;
      }
    };
    const ensureLoop = () => {
      if (!running) {
        running = true;
        raf = requestAnimationFrame(tick);
      }
    };

    const goTo = (target: number) => {
      const next = Math.max(0, Math.min(LAST, target));
      if (next === sceneRef.current && !a.tweening) return;
      sceneRef.current = next;
      setSceneIndex(next);
      if (next !== LAST) setFree(false);
      a.from = a.progress;
      a.to = REST[next];
      a.startT = performance.now();
      a.dur = STEP_MS;
      a.tweening = true;
      ensureLoop();
    };
    goToRef.current = goTo;

    ensureLoop(); // play the formation

    // Robust gesture detection. A deliberate scroll ACCELERATES (deltas rising),
    // while trailing trackpad momentum DECELERATES (deltas falling). We step on
    // the acceleration and ignore the decaying tail — so momentum never skips a
    // scene, and a fresh scroll during a momentum tail still registers because it
    // re-accelerates. After any real pause the history resets, so the next scroll
    // fires on its very first event. This is what the old idle-timeout couldn't
    // do: it mistook a follow-up scroll for the same gesture and dropped it.
    const deltas: number[] = [];
    let lastWheel = 0;
    let lockUntil = 0;
    const meanOfLast = (n: number) => {
      const s = deltas.slice(-n);
      return s.length ? s.reduce((sum, d) => sum + d, 0) / s.length : 0;
    };
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const ad = Math.abs(e.deltaY);
      if (ad < 4) return;
      const now = performance.now();
      if (now - lastWheel > 180) deltas.length = 0; // a pause ⇒ a new gesture
      lastWheel = now;
      deltas.push(ad);
      if (deltas.length > 40) deltas.shift();
      if (now < lockUntil) return; // swallow the rest of the firing burst
      // Rising (or steady) ⇒ still pushing; falling ⇒ momentum tail, ignore it.
      if (meanOfLast(4) >= meanOfLast(14)) {
        lockUntil = now + 260;
        goTo(sceneRef.current + (e.deltaY > 0 ? 1 : -1));
      }
    };
    let touchY = 0;
    const onTouchStart = (e: TouchEvent) => {
      touchY = e.touches[0]?.clientY ?? 0;
    };
    const onTouchMove = (e: TouchEvent) => e.preventDefault();
    const onTouchEnd = (e: TouchEvent) => {
      const dy = touchY - (e.changedTouches[0]?.clientY ?? touchY);
      if (Math.abs(dy) > 40) goTo(sceneRef.current + (dy > 0 ? 1 : -1));
    };

    stage.addEventListener("wheel", onWheel, { passive: false });
    stage.addEventListener("touchstart", onTouchStart, { passive: true });
    stage.addEventListener("touchmove", onTouchMove, { passive: false });
    stage.addEventListener("touchend", onTouchEnd, { passive: true });
    const onResize = () => renderer.resize();
    window.addEventListener("resize", onResize);
    return () => {
      cancelAnimationFrame(raf);
      running = false;
      stage.removeEventListener("wheel", onWheel);
      stage.removeEventListener("touchstart", onTouchStart);
      stage.removeEventListener("touchmove", onTouchMove);
      stage.removeEventListener("touchend", onTouchEnd);
      window.removeEventListener("resize", onResize);
      renderer.destroy();
      rendererRef.current = null;
    };
  }, [field.stars]);

  /* Free-explore interactions (active once the narrative settles). */
  const placeCard = useCallback((pt: { x: number; y: number }) => {
    const stage = stageRef.current;
    const w = stage?.clientWidth ?? 400;
    setHoverPos({
      x: Math.min(pt.x + 16, w - 240),
      y: Math.max(pt.y - 60, 12),
    });
  }, []);

  const onPointerMove = useCallback(
    (e: React.PointerEvent) => {
      const r = rendererRef.current;
      if (!r || !free) return;
      const rect = stageRef.current!.getBoundingClientRect();
      const { x, y } = r.unproject(e.clientX - rect.left, e.clientY - rect.top);
      const idx = grid.nearest(x, y);
      setHovered(idx >= 0 ? field.stars[idx] : null);
      if (idx >= 0) {
        placeCard(
          r.project(field.stars[idx].fieldX, field.stars[idx].fieldY),
        );
      }
    },
    [free, grid, field.stars, placeCard],
  );

  const onKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (!free) {
        // Pager mode: down/space advance a scene, up retreats one.
        if (e.key === "ArrowDown" || e.key === "PageDown" || e.key === " ") {
          e.preventDefault();
          goToRef.current(sceneRef.current + 1);
        } else if (e.key === "ArrowUp" || e.key === "PageUp") {
          e.preventDefault();
          goToRef.current(sceneRef.current - 1);
        }
        return;
      }
      // Free explore: PageUp steps back into the story; arrows walk the field.
      if (e.key === "PageUp") {
        e.preventDefault();
        goToRef.current(sceneRef.current - 1);
        return;
      }
      const dirs: Record<string, [number, number]> = {
        ArrowRight: [1, 0],
        ArrowLeft: [-1, 0],
        ArrowUp: [0, 1],
        ArrowDown: [0, -1],
      };
      if (e.key === "Enter" && selected >= 0) {
        router.push(`/movies/${field.stars[selected].tmdbId}`);
        return;
      }
      const dir = dirs[e.key];
      if (!dir) return;
      e.preventDefault();
      const next =
        selected < 0
          ? field.featuredIndex
          : grid.neighbor(selected, dir[0], dir[1]);
      if (next >= 0) {
        setSelected(next);
        const r = rendererRef.current;
        if (r) {
          placeCard(
            r.project(field.stars[next].fieldX, field.stars[next].fieldY),
          );
          setHovered(field.stars[next]);
        }
      }
    },
    [free, selected, grid, field, router, placeCard],
  );

  const active = hovered ?? (selected >= 0 ? field.stars[selected] : null);

  return (
    <div className="relative h-svh w-full">
      <div
        ref={stageRef}
        role="application"
        aria-label="A story of movies by budget and gross, in five scenes. Scroll, swipe, or use arrow keys to move between scenes; on the last, arrow keys walk between movies and Enter opens one."
        tabIndex={0}
        onPointerMove={onPointerMove}
        onPointerLeave={() => setHovered(null)}
        onKeyDown={onKeyDown}
        onClick={() => {
          if (free && hovered) router.push(`/movies/${hovered.tmdbId}`);
        }}
        className={`h-svh touch-none overflow-hidden ${free && hovered ? "cursor-pointer" : ""}`}
      >
        <div ref={canvasHostRef} className="absolute inset-0" />

        {/* Field axes */}
        <div
          ref={axesRef}
          aria-hidden
          className="pointer-events-none absolute inset-0 opacity-0"
        >
          <span className="absolute bottom-6 right-8 font-mono text-[11px] tracking-widest text-dim">
            BUDGET →
          </span>
          <span className="absolute left-6 top-24 font-mono text-[11px] tracking-widest text-dim [writing-mode:vertical-rl]">
            ← WORLDWIDE GROSS
          </span>
        </div>

        {/* Per-year axis (scene 4) */}
        <div
          ref={yearAxisRef}
          aria-hidden
          className="pointer-events-none absolute inset-x-0 bottom-10 flex justify-between px-[6%] font-mono text-[11px] text-dim opacity-0"
        >
          <span>{field.minYear}</span>
          <span className="text-predicted">over ↑ / under ↓ the real gross</span>
          <span>{field.maxYear}</span>
        </div>

        {/* Featured-star callout (scene 2) */}
        <div
          ref={calloutRef}
          aria-hidden
          className="pointer-events-none absolute left-0 top-0 opacity-0"
        >
          <div className="rounded border border-hairline bg-screen/90 px-3 py-2 backdrop-blur-sm">
            <p className="text-sm font-medium text-ink">{featured.title}</p>
            <p className="font-mono tabular text-xs text-dim">
              {featured.releaseYear} · budget{" "}
              <span className="text-ink">{dollarsCompact(featured.budget)}</span>{" "}
              · grossed{" "}
              <span className="text-actual">{dollarsCompact(featured.gross)}</span>
            </p>
          </div>
        </div>

        {/* Prediction beam (scene 2): a projector cone from off-frame opening onto
            the two candidate outcomes — amber actual, cyan prediction. */}
        <svg
          ref={beamRef}
          aria-hidden
          className="pointer-events-none absolute inset-0 h-full w-full opacity-0"
        >
          <defs>
            <linearGradient id="beam-cone" x1="0" y1="0" x2="1" y2="0">
              <stop offset="0" stopColor="oklch(0.75 0.13 230)" stopOpacity="0.01" />
              <stop offset="1" stopColor="oklch(0.75 0.13 230)" stopOpacity="0.11" />
            </linearGradient>
            <filter id="beam-glow" x="-60%" y="-60%" width="220%" height="220%">
              <feGaussianBlur stdDeviation="5" result="b" />
              <feMerge>
                <feMergeNode in="b" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
          </defs>
          <polygon points="" fill="url(#beam-cone)" />
          {/* The interval mouth — the model's spread between the outcomes. */}
          <line
            className="beam-mouth"
            stroke="oklch(0.75 0.13 230 / 0.5)"
            strokeWidth="1.5"
            strokeDasharray="2 4"
          />
          <circle
            className="beam-actual-dot"
            r="6"
            fill="oklch(0.78 0.14 75)"
            filter="url(#beam-glow)"
          />
          <circle
            className="beam-pred-dot"
            r="6"
            fill="oklch(0.75 0.13 230)"
            filter="url(#beam-glow)"
          >
            <animate attributeName="opacity" values="1;0.55;1" dur="2.4s" repeatCount="indefinite" />
          </circle>
        </svg>

        {/* Beam labels (HTML for real typography), positioned imperatively. */}
        <div
          ref={actualLabelRef}
          aria-hidden
          className="pointer-events-none absolute left-0 top-0 opacity-0"
        >
          <p className="title-caps text-[10px] tracking-[0.2em] text-actual/70">
            It actually made
          </p>
          <p className="mt-0.5 font-mono tabular text-3xl leading-none font-medium text-actual">
            {dollarsCompact(featured.gross)}
          </p>
        </div>
        <div
          ref={predLabelRef}
          aria-hidden
          className="pointer-events-none absolute left-0 top-0 opacity-0"
        >
          <p className="title-caps text-[10px] tracking-[0.2em] text-predicted/70">
            The model guessed
          </p>
          <p className="mt-0.5 font-mono tabular text-3xl leading-none font-medium text-predicted">
            {featured.predicted ? dollarsCompact(featured.predicted) : "—"}
          </p>
          <p className="mt-1.5 font-mono text-[11px] text-dim">{missLabel}</p>
        </div>

        {/* Captions */}
        <div ref={capARef} className="pointer-events-none absolute inset-x-0 bottom-[18svh] px-6">
          <div className="mx-auto max-w-6xl">
            <Headline field={field} totalGross={totalGross} />
          </div>
        </div>

        <div ref={capBRef} className="pointer-events-none absolute inset-x-0 bottom-[14svh] px-6 opacity-0">
          <div className="mx-auto max-w-6xl">
          <p className="max-w-md text-lg leading-relaxed text-dim">
            Every point of light is a movie. This one burned{" "}
            <span className="font-mono tabular text-ink">
              {dollarsCompact(featured.budget)}
            </span>{" "}
            of budget —{" "}
            <span className="text-actual">warm light is what really happened.</span>
          </p>
          </div>
        </div>

        <div ref={capCRef} className="pointer-events-none absolute inset-x-0 bottom-[16svh] px-6 opacity-0">
          <div className="mx-auto max-w-6xl">
            <p className="max-w-md text-lg leading-relaxed text-dim">
              Since {field.minYear}, the model has called every release blind —
              trained only on the past.{" "}
              <span className="text-predicted">Cool light is the machine.</span>{" "}
              It beats the naive baseline every year but one: 2020, when COVID
              broke everyone&apos;s model.
            </p>
          </div>
        </div>

        <div
          ref={capDRef}
          className="pointer-events-none absolute inset-x-0 bottom-[10svh] z-20 px-6 opacity-0"
        >
          <div className="mx-auto flex max-w-6xl flex-col gap-5">
            <p className="max-w-md text-lg text-dim">
              The field is yours — hover a star, or take the tour of the data.
            </p>
            {/* The CTAs win the click: they re-enable pointer events and stop the
                event bubbling to the stage, which would otherwise navigate to
                whatever star sits behind the button. */}
            <div
              className={`flex flex-wrap gap-3 ${free ? "pointer-events-auto" : "pointer-events-none"}`}
              onPointerEnter={() => setHovered(null)}
              onPointerMove={(e) => e.stopPropagation()}
              onPointerDown={(e) => e.stopPropagation()}
              onClick={(e) => e.stopPropagation()}
            >
              <Link
                href="/movies"
                className="rounded bg-actual px-4 py-2 text-sm font-medium text-screen transition-opacity duration-150 hover:opacity-90"
              >
                Explore the movies
              </Link>
              <Link
                href="/predict"
                className="rounded border border-predicted-deep px-4 py-2 text-sm text-predicted transition-colors duration-150 hover:bg-surface"
              >
                Ask the oracle
              </Link>
              <Link
                href="/model"
                className="rounded border border-hairline px-4 py-2 text-sm text-ink transition-colors duration-150 hover:bg-surface"
              >
                Read the report card
              </Link>
            </div>
          </div>
        </div>

        {/* Hover / keyboard-selection card */}
        {free && active && (
          <div
            className="pointer-events-none absolute z-10"
            style={{
              transform: `translate(${hoverPos.x}px, ${hoverPos.y}px)`,
            }}
          >
            <div className="flex w-56 gap-3 rounded border border-hairline bg-screen/95 p-3 backdrop-blur-sm">
              {posterUrl(active.posterPath, "w185") && (
                <div className="relative h-20 w-14 shrink-0 overflow-hidden rounded-sm">
                  <Image
                    src={posterUrl(active.posterPath, "w185")!}
                    alt=""
                    fill
                    sizes="56px"
                    className="object-cover"
                  />
                </div>
              )}
              <div className="min-w-0">
                <p className="truncate text-sm font-medium text-ink">{active.title}</p>
                <p className="font-mono tabular text-xs text-dim">{active.releaseYear}</p>
                <p className="font-mono tabular text-xs text-actual">
                  {dollarsCompact(active.gross)}
                </p>
                <p className="font-mono tabular text-[11px] text-dim">
                  on {dollarsCompact(active.budget)}
                </p>
              </div>
            </div>
          </div>
        )}

        <p aria-live="polite" className="sr-only">
          {active
            ? `${active.title}, ${active.releaseYear}, grossed ${dollarsCompact(active.gross)} on a ${dollarsCompact(active.budget)} budget. Press Enter to open.`
            : ""}
        </p>

        {/* Scroll cue */}
        <div
          ref={cueRef}
          aria-hidden
          className="absolute bottom-6 left-1/2 -translate-x-1/2 font-mono text-[11px] tracking-[0.3em] text-dim"
        >
          SCROLL
        </div>

        {/* Pager rail — progress and control, one dot per scene */}
        <nav
          aria-label="Scenes"
          className="absolute right-5 top-1/2 flex -translate-y-1/2 flex-col gap-3"
        >
          {REST.map((_, i) => (
            <button
              key={i}
              type="button"
              onClick={() => goToRef.current(i)}
              aria-label={`Go to scene ${i + 1} of ${REST.length}`}
              aria-current={sceneIndex === i}
              className="group grid place-items-center p-1.5"
            >
              <span
                className={`block rounded-full transition-all duration-300 ease-enter ${
                  sceneIndex === i
                    ? "h-2 w-2 bg-actual"
                    : "h-1.5 w-1.5 bg-dim/40 group-hover:bg-dim"
                }`}
              />
            </button>
          ))}
        </nav>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */

function Headline({ field, totalGross }: HeroProps) {
  return (
    <div>
      <p className="font-mono text-sm text-predicted">
        XGBoost · Snowflake · SageMaker · Lambda
      </p>
      <h1 className="title-caps mt-4 max-w-3xl text-4xl leading-tight text-ink sm:text-6xl">
        {field.stars.length.toLocaleString("en-US")} movies.{" "}
        <span className="text-actual">{dollarsCompact(totalGross)}</span> at
        the box office.
      </h1>
      <p className="mt-4 max-w-xl text-lg text-dim">
        One model with opinions about all of it.
      </p>
    </div>
  );
}

/**
 * Reduced-motion / no-WebGL fallback: the same field, composed as a still.
 * A designed state, not a degraded one.
 */
function StaticConstellation({ field, totalGross }: HeroProps) {
  // Sample to keep the DOM light; keep the brightest stars.
  const sample = useMemo(() => {
    const sorted = [...field.stars].sort((a, b) => b.gross - a.gross);
    return sorted.slice(0, 700);
  }, [field.stars]);

  return (
    <section className="relative overflow-hidden">
      <svg
        viewBox="0 0 1000 640"
        aria-hidden
        className="absolute inset-0 h-full w-full opacity-80"
        preserveAspectRatio="xMidYMid slice"
      >
        {sample.map((s) => (
          <circle
            key={s.tmdbId}
            cx={((s.fieldX + 1) / 2) * 1000}
            cy={((1 - (s.fieldY + 1) / 2) * 640)}
            r={s.size * 0.9}
            fill="oklch(0.78 0.14 75)"
            opacity={0.15 + s.intensity * 0.5}
          />
        ))}
      </svg>
      <div className="relative mx-auto flex min-h-[80svh] max-w-6xl items-end px-6 pb-16 pt-40">
        <div>
          <Headline field={field} totalGross={totalGross} />
          <div className="mt-8 flex flex-wrap gap-3">
            <Link
              href="/movies"
              className="rounded bg-actual px-4 py-2 text-sm font-medium text-screen transition-opacity duration-150 hover:opacity-90"
            >
              Explore the movies
            </Link>
            <Link
              href="/predict"
              className="rounded border border-predicted-deep px-4 py-2 text-sm text-predicted transition-colors duration-150 hover:bg-surface"
            >
              Ask the oracle
            </Link>
          </div>
        </div>
      </div>
    </section>
  );
}
