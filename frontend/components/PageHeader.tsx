import Link from "next/link";
import type { ReactNode } from "react";
import { SiteNav } from "./SiteNav";

export function PageHeader({
  title,
  current,
  children,
}: {
  title: string;
  current: "terminal" | "training" | "background";
  children?: ReactNode;
}) {
  return (
    <header className="topbar">
      <Link
        className="brand brand-home"
        href="/"
        aria-label="Market Gate Lab home"
      >
        <span className="brand-mark" aria-hidden="true">
          MG
        </span>
        <div>
          <h1>{title}</h1>
          <span className="brand-caption">Market Gate Lab</span>
        </div>
      </Link>
      <div className="topbar-actions">
        <SiteNav current={current} />
        {children}
      </div>
    </header>
  );
}
