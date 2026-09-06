import { createElement } from "react";
import { act, create, type ReactTestRenderer } from "react-test-renderer";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { TrainingLab } from "../components/TrainingLab";
vi.mock("../components/SgdPlayground", () => ({ SgdPlayground: () => null }));
import { useLiveTraining } from "../hooks/useLiveTraining";
import type { LiveTraining } from "../lib/liveTraining";

vi.mock("../components/PageHeader", () => ({ PageHeader: () => null }));
vi.mock("../hooks/useLiveTraining", () => ({ useLiveTraining: vi.fn() }));
vi.mock("../components/LiveLearning", () => ({
  useLiveLearning: () => ({ data: null }),
  LiveLearningPanel: () => null,
}));
vi.mock("../components/TrainingDataset", () => ({
  TrainingDataset: () => null,
}));
vi.mock("../components/TrainingControls", () => ({
  TrainingControls: () => null,
}));
let renderer: ReactTestRenderer;
const start = vi.fn();
const stop = vi.fn();
const focus = vi.fn();
const scrollIntoView = vi.fn();
const details = {
  open: false,
  scrollIntoView,
  querySelector: () => ({ focus }),
};
const idle: LiveTraining = {
  schema_version: 1,
  run_id: null,
  status: "idle",
  backend: "local",
  updated_at: null,
  dataset: null,
  latest: null,
  history: [],
  evaluation: null,
  error: null,
};
beforeEach(() => {
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  vi.spyOn(console, "error").mockImplementation(() => {});
  details.open = false;
  vi.mocked(useLiveTraining).mockReturnValue({
    data: idle,
    error: "",
    loading: false,
    pending: false,
    start,
    stop,
  });
});
afterEach(async () => {
  if (renderer) await act(async () => renderer.unmount());
  vi.restoreAllMocks();
  vi.clearAllMocks();
  vi.unstubAllGlobals();
});
async function mount() {
  await act(async () => {
    renderer = create(createElement(TrainingLab), {
      createNodeMock: (element) =>
        element.type === "details" ? details : null,
    });
  });
}
it("opens and focuses settings from the charts without starting a run", async () => {
  await mount();
  const actions = renderer.root.findByProps({
    className: "training-next-action",
  });
  expect(details.open).toBe(false);
  expect(actions.findByType("p").children.join("")).toContain(
    "bundled dataset",
  );
  await act(async () => actions.findByType("button").props.onClick());
  expect(details.open).toBe(true);
  expect(scrollIntoView).toHaveBeenCalledWith({ block: "start" });
  expect(focus).toHaveBeenCalledWith({ preventScroll: true });
  expect(start).not.toHaveBeenCalled();
  expect(stop).not.toHaveBeenCalled();
});
it("keeps Stop run above the collapsed settings only when the backend owns a process", async () => {
  vi.mocked(useLiveTraining).mockReturnValue({
    data: { ...idle, status: "running", can_stop: true },
    error: "",
    loading: false,
    pending: false,
    start,
    stop,
  });
  await mount();
  const actions = renderer.root.findByProps({
    className: "training-next-action",
  });
  const button = actions
    .findAllByType("button")
    .find((b) => b.children.includes("Stop run"))!;
  expect(actions.findByType("p").children.join("")).toContain("Charts update");
  expect(button.props.disabled).toBe(false);
  await act(async () => button.props.onClick());
  expect(stop).toHaveBeenCalledOnce();
  expect(details.open).toBe(false);
  expect(start).not.toHaveBeenCalled();
});
