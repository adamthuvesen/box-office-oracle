import type { Metadata } from "next";
import { Archivo, Instrument_Sans, Geist_Mono } from "next/font/google";
import { Suspense } from "react";
import { NavShell, SiteNav } from "@/components/site-nav";
import "./globals.css";
import { cn } from "@/lib/utils";

const archivo = Archivo({
  variable: "--font-archivo",
  subsets: ["latin"],
  axes: ["wdth"],
});

const instrumentSans = Instrument_Sans({
  variable: "--font-instrument",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: {
    default: "Box Office Oracle",
    template: "%s — Box Office Oracle",
  },
  description:
    "A machine-learning model that predicts what movies gross, and the data it learned from.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={cn(
        "h-full antialiased",
        archivo.variable,
        instrumentSans.variable,
        geistMono.variable,
      )}
    >
      <body className="flex min-h-full flex-col">
        <Suspense fallback={<NavShell activePath={null} />}>
          <SiteNav />
        </Suspense>
        <main className="flex-1">{children}</main>
        <footer className="border-t border-hairline px-6 py-5 text-sm text-dim">
          <div className="mx-auto flex max-w-6xl items-baseline justify-between gap-4">
            <span className="title-caps text-xs">Box Office Oracle</span>
            <span>
              Predictions from pre-release features only. Data: TMDB. Not
              financial advice for studio executives.
            </span>
          </div>
        </footer>
      </body>
    </html>
  );
}
