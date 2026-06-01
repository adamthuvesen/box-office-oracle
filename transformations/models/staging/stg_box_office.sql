WITH source_data AS (
  SELECT * FROM {{ source('RAW', 'BOX_OFFICE_V3') }}
),

deduped AS (
  SELECT *
  FROM source_data
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY TMDB_ID
    ORDER BY
      VOTES DESC NULLS LAST,
      RELEASE_DATE DESC NULLS LAST,
      TMDB_ID ASC
  ) = 1
)

SELECT
  -- Avoid exposing RANK as a downstream column name because it collides with
  -- the SQL reserved word.
  * EXCLUDE (RANK),
  RANK AS MOVIE_RANK,
  current_timestamp() AS loaded_at
FROM deduped
