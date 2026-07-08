import type { OraclePrediction } from "@/lib/types";
import { dollarsCompact, percent } from "@/lib/format";

interface PredictionPanelProps {
  /** Oracle prediction from the local retrain (web/data/predictions.json). */
  oracle: OraclePrediction | null;
}

/**
 * Actual vs the model's guess — the amber/cyan dichotomy at movie level.
 */
export function PredictionPanel({ oracle }: PredictionPanelProps) {
  if (oracle) {
    return <OraclePredictionPanel oracle={oracle} />;
  }
  return (
    <section className="rounded border border-hairline bg-surface p-5">
      <h2 className="title-caps text-sm text-dim">The model&apos;s view</h2>
      <p className="mt-3 max-w-prose text-sm leading-relaxed text-dim">
        No Oracle prediction for this movie — run{" "}
        <code className="font-mono text-ink">scripts/score_all_movies.py</code>{" "}
        to regenerate the predictions snapshot.
      </p>
    </section>
  );
}

const ORACLE_KIND_BADGE: Record<
  OraclePrediction["prediction_kind"],
  { label: string; title: string }
> = {
  out_of_sample: {
    label: "out-of-sample",
    title:
      "Honest prediction: made by a model trained only on earlier release years.",
  },
  in_sample: {
    label: "in-sample",
    title:
      "The model trained on this movie, so treat this prediction as a fun replay, not a test.",
  },
  no_actuals: {
    label: "no actuals",
    title: "No final worldwide gross exists yet to grade this prediction.",
  },
};

/** Oracle prediction from the local retrain: predicted vs actual (when
 * known), the error, and a badge separating honest out-of-sample calls from
 * in-sample replays. */
function OraclePredictionPanel({ oracle }: { oracle: OraclePrediction }) {
  const { predicted_gross, actual_gross, ape, prediction_kind } = oracle;
  const badge = ORACLE_KIND_BADGE[prediction_kind];
  const max = Math.max(predicted_gross, actual_gross ?? 0);
  const delta = actual_gross != null ? predicted_gross - actual_gross : null;

  return (
    <section className="rounded border border-hairline bg-surface p-5">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h2 className="title-caps text-sm text-dim">Oracle prediction</h2>
        <span
          title={badge.title}
          className="rounded-full border border-hairline px-2 py-0.5 font-mono text-xs text-dim"
        >
          {badge.label}
        </span>
      </div>

      <dl className="mt-5 flex flex-col gap-4">
        {actual_gross != null && (
          <Bar
            label="Actual"
            value={actual_gross}
            pct={(actual_gross / max) * 100}
            barClass="bg-actual"
            valueClass="text-actual"
          />
        )}
        <Bar
          label="Predicted"
          value={predicted_gross}
          pct={(predicted_gross / max) * 100}
          barClass="bg-predicted"
          valueClass="text-predicted"
        />
      </dl>

      {delta != null && ape != null ? (
        <p className="mt-5 text-sm text-dim">
          <span className={delta > 0 ? "text-predicted" : "text-under"}>
            {delta > 0 ? "+" : "−"}
            {percent(ape)}
          </span>{" "}
          vs actual — {delta > 0 ? "over" : "under"} by{" "}
          <span className="font-mono tabular text-ink">
            {dollarsCompact(Math.abs(delta))}
          </span>
          .
        </p>
      ) : (
        <p className="mt-5 text-sm text-dim">
          No final gross to grade this one against yet.
        </p>
      )}
    </section>
  );
}

function Bar({
  label,
  value,
  pct,
  barClass,
  valueClass,
}: {
  label: string;
  value: number;
  pct: number;
  barClass: string;
  valueClass: string;
}) {
  return (
    <div>
      <div className="flex items-baseline justify-between gap-4">
        <dt className="text-xs uppercase tracking-wider text-dim">{label}</dt>
        <dd className={`font-mono tabular text-sm ${valueClass}`}>
          {dollarsCompact(value)}
        </dd>
      </div>
      <div className="mt-1.5 h-2 rounded-full bg-surface-2">
        <div
          className={`h-full rounded-full ${barClass}`}
          style={{ width: `${Math.max(pct, 1)}%` }}
        />
      </div>
    </div>
  );
}
