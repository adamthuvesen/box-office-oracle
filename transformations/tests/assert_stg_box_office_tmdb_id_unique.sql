-- Singular test: assert that stg_box_office contains no duplicate TMDB_IDs.
-- The staging model deduplicates the raw source via QUALIFY ROW_NUMBER() = 1;
-- if a duplicate slips through it is a defect in that dedup logic.
-- This test returns rows on failure (dbt convention: non-empty result = failure).

SELECT
    tmdb_id,
    COUNT(*) AS occurrences
FROM {{ ref('stg_box_office') }}
GROUP BY tmdb_id
HAVING COUNT(*) > 1
