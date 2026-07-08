export function NoDataYet({ what = "the data snapshot" }: { what?: string }) {
  return (
    <section className="mx-auto flex max-w-2xl flex-col items-start gap-4 px-6 py-24">
      <h1 className="title-caps text-lg text-ink">House lights are up</h1>
      <p className="text-dim">
        This page needs {what}, and it hasn&apos;t been exported yet. From the
        repo root, run:
      </p>
      <pre className="w-full overflow-x-auto rounded border border-hairline bg-surface px-4 py-3 font-mono text-sm text-actual">
        make web-data
      </pre>
      <p className="text-sm text-dim">
        That pulls the movie catalog from Snowflake (read-only) into{" "}
        <code className="font-mono">web/data/</code>, which stays out of git.
        Then refresh this page.
      </p>
    </section>
  );
}
