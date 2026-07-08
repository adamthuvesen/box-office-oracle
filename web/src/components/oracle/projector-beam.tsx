"use client";

import { useEffect } from "react";
import {
  animate,
  motion,
  useMotionValue,
  useReducedMotion,
  useTransform,
} from "motion/react";
import { dollarsCompact } from "@/lib/format";

const EASE_ENTER = [0.22, 1, 0.36, 1] as const;

const W = 640;
const H = 170;
const PAD_L = 40;
const PAD_R = 18;
const AXIS_Y = 130;
const SRC = { x: 12, y: 30 };
const TICKS = [1e6, 1e7, 1e8, 1e9];

interface ProjectorBeamProps {
  prediction: number;
  lower: number;
  upper: number;
}

/**
 * A cone of cyan light sweeps open from a projector point on the left to
 * bracket [lower, upper] on a log-scale dollar axis. The predicted gross is
 * the bright dot on the axis.
 */
export function ProjectorBeam({ prediction, lower, upper }: ProjectorBeamProps) {
  const domainMin = Math.min(1e6, lower / 1.4);
  const domainMax = Math.max(1e9, upper * 1.3);
  const logSpan = Math.log10(domainMax) - Math.log10(domainMin);
  const x = (v: number) =>
    PAD_L +
    ((Math.log10(Math.max(v, 1)) - Math.log10(domainMin)) / logSpan) *
      (W - PAD_L - PAD_R);

  const lowerX = x(lower);
  const upperX = x(upper);
  const predX = x(prediction);

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      role="img"
      aria-label={`Predicted ${dollarsCompact(prediction)}, heuristic interval ${dollarsCompact(lower)} to ${dollarsCompact(upper)}`}
      className="w-full"
    >
      <defs>
        <filter id="oracle-glow" x="-80%" y="-80%" width="260%" height="260%">
          <feGaussianBlur stdDeviation="4" result="blur" />
          <feMerge>
            <feMergeNode in="blur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>

      {/* Dollar axis */}
      <line
        x1={PAD_L}
        y1={AXIS_Y}
        x2={W - PAD_R}
        y2={AXIS_Y}
        stroke="var(--color-hairline)"
        strokeWidth={1}
      />
      {TICKS.filter((t) => t >= domainMin && t <= domainMax).map((t) => (
        <g key={t}>
          <line
            x1={x(t)}
            y1={AXIS_Y}
            x2={x(t)}
            y2={AXIS_Y + 5}
            stroke="var(--color-hairline)"
            strokeWidth={1}
          />
          <text
            x={x(t)}
            y={AXIS_Y + 19}
            textAnchor="middle"
            fontSize={11}
            className="fill-dim font-mono"
          >
            {dollarsCompact(t)}
          </text>
        </g>
      ))}

      <BeamCone lowerX={lowerX} upperX={upperX} predX={predX} />

      {/* The projector itself */}
      <circle cx={SRC.x} cy={SRC.y} r={2.5} fill="var(--color-predicted)" />

      {/* Interval bounds */}
      <IntervalLabel xPos={lowerX} value={lower} />
      <IntervalLabel xPos={upperX} value={upper} />

      {/* The prediction: a bright dot with a soft glow */}
      <motion.circle
        cx={predX}
        cy={AXIS_Y}
        r={4.5}
        fill="var(--color-predicted)"
        filter="url(#oracle-glow)"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.3, duration: 0.3, ease: "easeOut" }}
      />
    </svg>
  );
}

/** The cone opens outward from the prediction ray to the interval bounds. */
function BeamCone({
  lowerX,
  upperX,
  predX,
}: {
  lowerX: number;
  upperX: number;
  predX: number;
}) {
  const reduced = useReducedMotion();
  const progress = useMotionValue(reduced ? 1 : 0);

  useEffect(() => {
    if (reduced) {
      progress.set(1);
      return;
    }
    const controls = animate(progress, 1, {
      duration: 0.6,
      ease: EASE_ENTER,
    });
    return () => controls.stop();
  }, [progress, reduced]);

  const lowEdge = useTransform(progress, (p) => predX + (lowerX - predX) * p);
  const upEdge = useTransform(progress, (p) => predX + (upperX - predX) * p);
  const points = useTransform(
    progress,
    (p) =>
      `${SRC.x},${SRC.y} ${predX + (lowerX - predX) * p},${AXIS_Y} ${predX + (upperX - predX) * p},${AXIS_Y}`,
  );

  return (
    <>
      <motion.polygon
        points={points}
        fill="var(--color-predicted)"
        opacity={0.08}
      />
      <motion.line
        x1={SRC.x}
        y1={SRC.y}
        x2={lowEdge}
        y2={AXIS_Y}
        stroke="var(--color-predicted)"
        strokeOpacity={0.45}
        strokeWidth={1}
      />
      <motion.line
        x1={SRC.x}
        y1={SRC.y}
        x2={upEdge}
        y2={AXIS_Y}
        stroke="var(--color-predicted)"
        strokeOpacity={0.45}
        strokeWidth={1}
      />
    </>
  );
}

/** Bound label; anchors flip near the edges so text never leaves the viewBox. */
function IntervalLabel({ xPos, value }: { xPos: number; value: number }) {
  const anchor = xPos < 60 ? "start" : xPos > W - 60 ? "end" : "middle";
  return (
    <motion.text
      x={xPos}
      y={AXIS_Y - 10}
      textAnchor={anchor}
      fontSize={11}
      className="fill-predicted font-mono"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ delay: 0.4, duration: 0.3, ease: "easeOut" }}
    >
      {dollarsCompact(value)}
    </motion.text>
  );
}
