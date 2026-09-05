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
      <details className="banner-about">
        <summary>
          <span>
            A simple demo of neural networks guiding fast market rules.
          </span>
          <span className="banner-about-toggle" aria-hidden="true">
            About this demo
          </span>
        </summary>
        <div className="banner-about-body">
          <p>
            A small neural controller reads recent market summaries and adjusts
            the mix of three specialist rules. Those rules react to accepted
            book and trade events, while fixed risk checks decide whether to
            allow a synthetic quote.
          </p>
          <p>
            The architecture illustrates how neural context could work alongside
            high-resolution data and low-latency algorithms. This Python demo
            runs both cadences in one process; it does not measure production
            HFT speed or send exchange orders.
          </p>
        </div>
      </details>
    </header>
  );
}
