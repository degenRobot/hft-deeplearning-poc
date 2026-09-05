import { createElement, type ReactNode } from "react";
import { act, create, type ReactTestRenderer } from "react-test-renderer";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { InferenceBudget } from "../components/InferenceBudget";

let renderer: ReactTestRenderer | undefined;
const text = (node: { children: unknown[] }): string =>
  node.children
    .map((child) =>
      typeof child === "object" && child !== null && "children" in child
        ? text(child as { children: unknown[] })
        : String(child as ReactNode),
    )
    .join("");

beforeEach(async () => {
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  vi.spyOn(console, "error").mockImplementation(() => {});
  await act(async () => {
    renderer = create(createElement(InferenceBudget));
  });
});
afterEach(async () => {
  await act(async () => renderer?.unmount());
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

it("shows the duty cycle and serial throughput with explicit units", () => {
  const status = text(renderer!.root.findByProps({ role: "status" }));
  expect(status).toContain("0.2% of the interval");
  expect(status).toContain("500 updates / second");
  const fastest = renderer!.root.findAllByType("li")[0];
  expect(text(fastest)).toContain("200%over budget");
  expect(
    fastest
      .findAllByType("span")
      .some((node) => node.props.style?.width === "100%"),
  ).toBe(true);
});

it("distinguishes fully occupied from overloaded and updates capped chart bars", async () => {
  const root = renderer!.root;
  await act(async () => {
    root
      .findByProps({ id: "controller-interval" })
      .props.onChange({ target: { value: "2" } });
  });
  expect(text(root.findByProps({ role: "status" }))).toContain(
    "Fully occupied",
  );
  await act(async () => {
    root
      .findByProps({ id: "inference-time" })
      .props.onChange({ target: { value: "10" } });
  });
  const status = text(root.findByProps({ role: "status" }));
  expect(status).toContain("500% of the interval");
  expect(status).toContain("cannot keep this cadence");
  expect(status).toContain("100 updates / second");
  const hundredMs = root.findAllByType("li")[2];
  expect(text(hundredMs)).toContain("10%");
  expect(
    hundredMs
      .findAllByType("span")
      .some((node) => node.props.style?.width === "10%"),
  ).toBe(true);
});

it("keeps small nonzero duty cycles visible at the slider minimum", async () => {
  await act(async () => {
    renderer!.root
      .findByProps({ id: "inference-time" })
      .props.onChange({ target: { value: "0.1" } });
  });
  expect(text(renderer!.root.findByProps({ role: "status" }))).toContain(
    "0.01% of the interval",
  );
  expect(text(renderer!.root.findAllByType("li")[3])).toContain("0.01%");
});
