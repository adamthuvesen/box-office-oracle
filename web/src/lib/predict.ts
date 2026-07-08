import { z } from "zod";

/** Mirrors the inference API contract (box_office/inference/app/predictor.py). */
export const predictRequestSchema = z.strictObject({
  budget: z.number().min(0),
  runtime: z.number().min(1).max(500),
  genre: z.union([z.string().min(1), z.array(z.string().min(1)).min(1)]),
  release_month: z.number().int().min(1).max(12),
  release_year: z.number().int().min(1900).max(2030),
  mpaa: z.string().default("Not Rated"),
  director: z.string().optional(),
  actors: z.array(z.string()).optional(),
  production_company: z.string().optional(),
  // Pre-release IP/franchise strength; defaults describe an original movie.
  ip_tier: z.number().int().min(1).max(5).default(5),
  prior_franchise_gross: z.number().min(0).default(0),
  is_franchise_followup: z.boolean().default(false),
  return_confidence: z.boolean().default(true),
});

export type PredictRequest = z.infer<typeof predictRequestSchema>;
/** What the client sends — fields with server-side defaults stay optional. */
export type PredictRequestInput = z.input<typeof predictRequestSchema>;

export const GENRES = [
  "Action",
  "Adventure",
  "Animation",
  "Comedy",
  "Crime",
  "Drama",
  "Family",
  "Fantasy",
  "History",
  "Horror",
  "Music",
  "Mystery",
  "Romance",
  "Science Fiction",
  "Thriller",
  "War",
  "Western",
] as const;

/** Ordinal IP tiers, mirrored from the classifier (1 = strongest, 5 = none). */
export const IP_TIERS = [
  { value: 5, label: "Original / no IP" },
  { value: 4, label: "Nominal IP (adaptation, minor property)" },
  { value: 3, label: "Known source work" },
  { value: 2, label: "Strong franchise / brand" },
  { value: 1, label: "Top-tier pre-sold IP" },
] as const;

export const MPAA_RATINGS = [
  "G",
  "PG",
  "PG-13",
  "R",
  "NC-17",
  "Not Rated",
] as const;

export const MONTHS = [
  "January",
  "February",
  "March",
  "April",
  "May",
  "June",
  "July",
  "August",
  "September",
  "October",
  "November",
  "December",
] as const;

/** The slice of a catalog movie the comparables strip needs — keeps the RSC→client payload small. */
export interface ComparableMovie {
  tmdb_id: number;
  title: string;
  release_year: number;
  poster_path: string | null;
  worldwide_gross: number;
}
