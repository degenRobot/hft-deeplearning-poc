import Link from "next/link";

const destinations = [
  { id: "terminal", href: "/", label: "Live Terminal" },
  { id: "training", href: "/training", label: "Training Lab" },
  { id: "background", href: "/background", label: "Background" },
] as const;

export function SiteNav({
  current,
}: {
  current: "terminal" | "training" | "background";
}) {
  return (
    <nav className="site-nav" aria-label="Main navigation">
      {destinations.map(({ id, href, label }) => (
        <Link
          key={id}
          href={href}
          aria-current={current === id ? "page" : undefined}
        >
          {label}
        </Link>
      ))}
    </nav>
  );
}
