import type { Metadata } from "next";
import type { ReactNode } from "react";
import "./globals.css";

const description = "Static dashboard for reviewing RAG hallucination benchmark results.";

const PRODUCTION_URL = "https://cs4701.vercel.app";
const siteUrl = process.env.NEXT_PUBLIC_SITE_URL ?? PRODUCTION_URL;

export const metadata: Metadata = {
  metadataBase: new URL(siteUrl),
  title: "Cornell Policy RAG Results Explorer",
  description,
  openGraph: {
    title: "Cornell Policy RAG Results Explorer",
    description,
    images: [
      {
        url: "/og-image.png",
        width: 1200,
        height: 630,
        alt: "RAG Evaluation Dashboard preview showing the system summary table.",
      },
    ],
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: "Cornell Policy RAG Results Explorer",
    description,
    images: ["/og-image.png"],
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
