"use client";

import { useId, useState, type ReactNode } from "react";

/** A labelled disclosure, usable by touch, keyboard and pointer. */
export function ContextTip({
  label,
  children,
}: {
  label: string;
  children?: ReactNode;
}) {
  const [open, setOpen] = useState(false);
  const id = useId();
  return (
    <span
      className="context-tip"
      onKeyDown={(event) => {
        if (event.key === "Escape") {
          setOpen(false);
          event.stopPropagation();
        }
      }}
      onBlur={(event) => {
        if (!event.currentTarget.contains(event.relatedTarget as Node | null))
          setOpen(false);
      }}
    >
      <button
        type="button"
        aria-label={label}
        aria-expanded={open}
        aria-controls={id}
        onClick={() => setOpen(!open)}
      >
        ?
      </button>
      <span id={id} className="context-tip-content" hidden={!open}>
        {children}
      </span>
    </span>
  );
}
