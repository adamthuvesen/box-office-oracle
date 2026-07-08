"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const LINKS = [
  { href: "/movies", label: "Movies" },
  { href: "/stats", label: "Stats" },
  { href: "/model", label: "Model" },
  { href: "/predict", label: "Predict" },
] as const;

/** Reads the pathname — must sit under a Suspense boundary (PPR). */
export function SiteNav() {
  const pathname = usePathname();
  return <NavShell activePath={pathname} />;
}

/** Presentational nav; used directly as the static Suspense fallback. */
export function NavShell({ activePath }: { activePath: string | null }) {
  return (
    <header className="sticky top-0 z-40 border-b border-hairline bg-screen/85 backdrop-blur-sm">
      <nav
        aria-label="Main"
        className="mx-auto flex h-14 max-w-6xl items-center justify-between gap-2 px-4 sm:px-6"
      >
        <Link
          href="/"
          className="title-caps whitespace-nowrap text-[11px] text-ink sm:text-sm"
        >
          <span className="max-sm:hidden">Box Office </span>
          <span className="sm:hidden">B.O. </span>
          <span className="text-actual">Oracle</span>
        </Link>
        <ul className="flex items-center gap-0 sm:gap-1">
          {LINKS.map(({ href, label }) => {
            const active =
              activePath === href || activePath?.startsWith(`${href}/`);
            return (
              <li key={href}>
                <Link
                  href={href}
                  aria-current={active ? "page" : undefined}
                  className={`rounded px-2 py-1.5 text-xs transition-colors duration-150 sm:px-3 sm:text-sm ${
                    active
                      ? "text-actual"
                      : "text-dim hover:bg-surface hover:text-ink"
                  }`}
                >
                  {label}
                </Link>
              </li>
            );
          })}
        </ul>
      </nav>
    </header>
  );
}
