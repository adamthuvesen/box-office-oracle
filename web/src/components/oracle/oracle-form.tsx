"use client";

import { useId, useState } from "react";
import { dollarsCompact } from "@/lib/format";
import {
  GENRES,
  IP_TIERS,
  MONTHS,
  MPAA_RATINGS,
  type PredictRequestInput,
} from "@/lib/predict";

const YEARS = [2025, 2026, 2027, 2028, 2029, 2030];

const FIELD_CLASS =
  "h-9 w-full rounded border border-hairline bg-surface px-3 text-sm text-ink placeholder:text-dim focus:border-predicted-deep focus:outline-none";

interface OracleFormProps {
  pending: boolean;
  onAsk: (request: PredictRequestInput) => void;
}

export function OracleForm({ pending, onAsk }: OracleFormProps) {
  const uid = useId();
  const [budget, setBudget] = useState(50_000_000);
  const [genres, setGenres] = useState<string[]>(["Action"]);
  const [runtime, setRuntime] = useState(110);
  const [month, setMonth] = useState(7);
  const [year, setYear] = useState(2026);
  const [mpaa, setMpaa] = useState("PG-13");
  const [director, setDirector] = useState("");
  const [actors, setActors] = useState("");
  const [company, setCompany] = useState("");
  const [ipTier, setIpTier] = useState(5);
  const [priorGross, setPriorGross] = useState(0);
  const [isFollowup, setIsFollowup] = useState(false);

  function toggleGenre(genre: string) {
    setGenres((prev) =>
      prev.includes(genre)
        ? prev.filter((g) => g !== genre)
        : [...prev, genre],
    );
  }

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (pending || genres.length === 0) return;
    const actorList = actors
      .split(",")
      .map((s) => s.trim())
      .filter(Boolean);
    onAsk({
      budget,
      genre: genres,
      runtime: Number.isFinite(runtime) ? clamp(Math.round(runtime), 60, 240) : 110,
      release_month: month,
      release_year: year,
      mpaa,
      ...(director.trim() && { director: director.trim() }),
      ...(actorList.length > 0 && { actors: actorList }),
      ...(company.trim() && { production_company: company.trim() }),
      ip_tier: ipTier,
      prior_franchise_gross: priorGross,
      is_franchise_followup: isFollowup,
    });
  }

  return (
    <form onSubmit={handleSubmit} className="flex flex-col gap-6">
      <DollarSlider
        id={`${uid}-budget`}
        label="Production budget"
        min={1_000_000}
        max={400_000_000}
        value={budget}
        onChange={setBudget}
      />

      <fieldset>
        <legend className="text-sm text-ink">Genres</legend>
        <div className="mt-2 flex flex-wrap gap-1.5">
          {GENRES.map((g) => (
            <GenreChip
              key={g}
              label={g}
              active={genres.includes(g)}
              onClick={() => toggleGenre(g)}
            />
          ))}
        </div>
        {genres.length === 0 && (
          <p className="mt-2 text-xs text-dim">
            Pick at least one genre — the oracle needs it.
          </p>
        )}
      </fieldset>

      <div className="grid grid-cols-2 gap-4">
        <div>
          <label htmlFor={`${uid}-runtime`} className="text-sm text-ink">
            Runtime (minutes)
          </label>
          <input
            id={`${uid}-runtime`}
            type="number"
            min={60}
            max={240}
            value={Number.isFinite(runtime) ? runtime : ""}
            onChange={(e) => setRuntime(e.target.valueAsNumber)}
            onBlur={() =>
              setRuntime((v) =>
                Number.isFinite(v) ? clamp(Math.round(v), 60, 240) : 110,
              )
            }
            className={`mt-1.5 font-mono tabular ${FIELD_CLASS}`}
          />
        </div>
        <div>
          <label htmlFor={`${uid}-mpaa`} className="text-sm text-ink">
            MPAA rating
          </label>
          <select
            id={`${uid}-mpaa`}
            value={mpaa}
            onChange={(e) => setMpaa(e.target.value)}
            className={`mt-1.5 ${FIELD_CLASS}`}
          >
            {MPAA_RATINGS.map((r) => (
              <option key={r} value={r}>
                {r}
              </option>
            ))}
          </select>
        </div>
        <div>
          <label htmlFor={`${uid}-month`} className="text-sm text-ink">
            Release month
          </label>
          <select
            id={`${uid}-month`}
            value={month}
            onChange={(e) => setMonth(Number(e.target.value))}
            className={`mt-1.5 ${FIELD_CLASS}`}
          >
            {MONTHS.map((name, i) => (
              <option key={name} value={i + 1}>
                {name}
              </option>
            ))}
          </select>
        </div>
        <div>
          <label htmlFor={`${uid}-year`} className="text-sm text-ink">
            Release year
          </label>
          <select
            id={`${uid}-year`}
            value={year}
            onChange={(e) => setYear(Number(e.target.value))}
            className={`mt-1.5 ${FIELD_CLASS}`}
          >
            {YEARS.map((y) => (
              <option key={y} value={y}>
                {y}
              </option>
            ))}
          </select>
        </div>
      </div>

      <div className="flex flex-col gap-4">
        <div>
          <label htmlFor={`${uid}-director`} className="text-sm text-ink">
            Director <span className="text-dim">(optional)</span>
          </label>
          <input
            id={`${uid}-director`}
            type="text"
            value={director}
            onChange={(e) => setDirector(e.target.value)}
            placeholder="Greta Gerwig"
            className={`mt-1.5 ${FIELD_CLASS}`}
          />
        </div>
        <div>
          <label htmlFor={`${uid}-actors`} className="text-sm text-ink">
            Lead actors, comma-separated{" "}
            <span className="text-dim">(optional)</span>
          </label>
          <input
            id={`${uid}-actors`}
            type="text"
            value={actors}
            onChange={(e) => setActors(e.target.value)}
            placeholder="Margot Robbie, Ryan Gosling"
            className={`mt-1.5 ${FIELD_CLASS}`}
          />
        </div>
        <div>
          <label htmlFor={`${uid}-company`} className="text-sm text-ink">
            Studio <span className="text-dim">(optional)</span>
          </label>
          <input
            id={`${uid}-company`}
            type="text"
            value={company}
            onChange={(e) => setCompany(e.target.value)}
            placeholder="Warner Bros. Pictures"
            className={`mt-1.5 ${FIELD_CLASS}`}
          />
        </div>
      </div>

      <fieldset className="flex flex-col gap-4">
        <legend className="text-sm text-ink">
          Franchise / IP <span className="text-dim">(optional)</span>
        </legend>
        <div className="mt-2">
          <label htmlFor={`${uid}-ip-tier`} className="text-sm text-ink">
            Pre-sold IP tier
          </label>
          <select
            id={`${uid}-ip-tier`}
            value={ipTier}
            onChange={(e) => setIpTier(Number(e.target.value))}
            className={`mt-1.5 ${FIELD_CLASS}`}
          >
            {IP_TIERS.map((t) => (
              <option key={t.value} value={t.value}>
                {t.label}
              </option>
            ))}
          </select>
        </div>
        <DollarSlider
          id={`${uid}-prior-gross`}
          label="Prior franchise gross (earlier films in the series)"
          min={0}
          max={10_000_000_000}
          value={priorGross}
          onChange={setPriorGross}
        />
        <label
          htmlFor={`${uid}-followup`}
          className="flex items-center gap-2 text-sm text-ink"
        >
          <input
            id={`${uid}-followup`}
            type="checkbox"
            checked={isFollowup}
            onChange={(e) => setIsFollowup(e.target.checked)}
            className="h-4 w-4 accent-(--color-actual)"
          />
          Follow-up to an earlier film in the same series
        </label>
      </fieldset>

      <button
        type="submit"
        disabled={pending || genres.length === 0}
        className={`h-11 self-start rounded bg-actual px-6 text-sm font-medium text-screen transition-colors duration-150 hover:bg-actual/90 disabled:cursor-not-allowed ${
          pending ? "animate-pulse" : "disabled:opacity-50"
        }`}
      >
        {pending ? "Consulting…" : "Consult the oracle"}
      </button>
    </form>
  );
}

/** Below this a zero-allowing slider snaps to $0 — a log scale can't reach zero. */
const ZERO_FLOOR = 100_000;

interface DollarSliderProps {
  id: string;
  label: string;
  min: number;
  max: number;
  value: number;
  onChange: (value: number) => void;
}

/** Log-scaled dollar slider with a synced compact-dollar text input. */
function DollarSlider({ id, label, min, max, value, onChange }: DollarSliderProps) {
  const allowZero = min <= 0;
  const logMin = Math.log10(allowZero ? ZERO_FLOOR : min);
  const logMax = Math.log10(max);
  const sliderMin = allowZero ? logMin - 0.2 : logMin;
  const sliderPos =
    value <= 0 ? sliderMin : clamp(Math.log10(value), logMin, logMax);
  // While the user types, the text input holds a draft; null mirrors the value.
  const [draft, setDraft] = useState<string | null>(null);

  function commitDraft() {
    if (draft === null) return;
    const parsed = parseDollars(draft);
    if (parsed !== null) onChange(clamp(parsed, min, max));
    setDraft(null);
  }

  return (
    <div>
      <div className="flex items-baseline justify-between gap-3">
        <label htmlFor={id} className="text-sm text-ink">
          {label}
        </label>
        <input
          type="text"
          inputMode="decimal"
          aria-label={`${label} in dollars`}
          value={draft ?? dollarsCompact(value)}
          onChange={(e) => setDraft(e.target.value)}
          onBlur={commitDraft}
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              e.preventDefault();
              commitDraft();
            }
          }}
          className="h-8 w-24 rounded border border-hairline bg-surface px-2 text-right font-mono tabular text-sm text-ink focus:border-predicted-deep focus:outline-none"
        />
      </div>
      <input
        id={id}
        type="range"
        min={sliderMin}
        max={logMax}
        step={0.01}
        value={sliderPos}
        onChange={(e) => {
          const x = Number(e.target.value);
          setDraft(null);
          onChange(allowZero && x < logMin ? 0 : roundDollars(10 ** x));
        }}
        className="mt-2 w-full accent-(--color-actual)"
      />
    </div>
  );
}

function GenreChip({
  label,
  active,
  onClick,
}: {
  label: string;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={active}
      className={`rounded-full border px-3 py-1 text-xs transition-colors duration-150 ${
        active
          ? "border-actual-deep bg-actual/10 text-actual"
          : "border-hairline text-dim hover:border-actual-deep/50 hover:text-ink"
      }`}
    >
      {label}
    </button>
  );
}

function clamp(v: number, min: number, max: number): number {
  return Math.min(Math.max(v, min), max);
}

/** Round to 3 significant digits so slider values read as clean dollars. */
function roundDollars(v: number): number {
  const mag = 10 ** (Math.floor(Math.log10(v)) - 2);
  return Math.round(v / mag) * mag;
}

/** Parse "$150M", "1.5b", "80000000" → dollars. Null when unparseable. */
function parseDollars(s: string): number | null {
  const m = s
    .trim()
    .replace(/[$,\s]/g, "")
    .match(/^(\d+(?:\.\d+)?)([kmb])?$/i);
  if (!m) return null;
  const mult = { k: 1e3, m: 1e6, b: 1e9 }[m[2]?.toLowerCase() ?? ""] ?? 1;
  return Number(m[1]) * mult;
}
