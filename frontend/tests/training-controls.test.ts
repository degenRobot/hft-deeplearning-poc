import { createElement } from "react";
import { act, create, type ReactTestRenderer } from "react-test-renderer";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { TrainingControls } from "../components/TrainingControls";
import {
  DEFAULT_TRAINING_OPTIONS,
  parseTrainingSettings,
  trainingCanStop,
  trainingParameterCount,
  validTrainingOptions,
  type LiveTraining,
} from "../lib/liveTraining";
const settings = {
  modal: { configured: true, available: true },
  defaults: DEFAULT_TRAINING_OPTIONS,
  limits: { hidden_min: 8, hidden_max: 1024, epochs_max: 50 },
  resources: { cpu: 2, memory_gib: 2, timeout_seconds: 600 },
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
let renderer: ReactTestRenderer | undefined;
const start = vi.fn();
const response = (body: unknown) => ({ ok: true, json: async () => body });
async function mount() {
  await act(async () => {
    renderer = create(
      createElement(TrainingControls, {
        data: idle,
        loading: false,
        pending: false,
        start,
        stop: vi.fn(),
      }),
    );
  });
}
const root = () => renderer!.root;
const launch = () =>
  root()
    .findAllByType("button")
    .find((b) => b.props.className === "button primary")!;
const passwordFields = () =>
  root()
    .findAllByType("input")
    .filter((x) => x.props.type === "password");
async function modal() {
  await act(async () =>
    root()
      .findAllByType("select")[0]
      .props.onChange({ target: { value: "modal" } }),
  );
}
beforeEach(() => {
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  vi.stubGlobal(
    "fetch",
    vi.fn(async () => response(settings)),
  );
  vi.spyOn(console, "error").mockImplementation(() => {});
  start.mockReset();
});
afterEach(async () => {
  if (renderer) await act(async () => renderer?.unmount());
  renderer = undefined;
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});
describe("training setup", () => {
  it("forwards selected Modal architecture and rejects invalid widths before launch", async () => {
    await mount();
    await modal();
    await act(async () =>
      root()
        .findAllByType("select")[1]
        .props.onChange({ target: { value: "2" } }),
    );
    expect(launch().props.disabled).toBe(false);
    await act(async () => launch().props.onClick());
    expect(start).toHaveBeenCalledWith({
      ...DEFAULT_TRAINING_OPTIONS,
      backend: "modal",
      hidden_1: 1024,
      hidden_2: 512,
    });
    await act(async () =>
      root()
        .findAllByType("input")[0]
        .props.onChange({ target: { value: "1025" } }),
    );
    expect(launch().props.disabled).toBe(true);
  });
  it("clears credentials on submit, does not render backend errors, and clears on disclosure close", async () => {
    await mount();
    await modal();
    const disclosure = () =>
      root()
        .findAllByType("button")
        .find((b) => b.props["aria-expanded"] !== undefined)!;
    await act(async () => disclosure().props.onClick());
    await act(async () => {
      passwordFields()[0].props.onChange({ target: { value: "test-id" } });
      passwordFields()[1].props.onChange({ target: { value: "test-secret" } });
    });
    vi.mocked(fetch).mockRejectedValueOnce(new Error("test-secret"));
    await act(async () =>
      root()
        .findByType("form")
        .props.onSubmit({ preventDefault() {} }),
    );
    expect(passwordFields().map((x) => x.props.value)).toEqual(["", ""]);
    expect(
      root().findByProps({ role: "alert" }).children.join(" "),
    ).not.toContain("test-secret");
    await act(async () =>
      passwordFields()[0].props.onChange({ target: { value: "test-id" } }),
    );
    await act(async () => disclosure().props.onClick());
    await act(async () => disclosure().props.onClick());
    expect(passwordFields().map((x) => x.props.value)).toEqual(["", ""]);
  });
  it("accepts only masked credential status and closes the form after saving", async () => {
    vi.mocked(fetch).mockResolvedValueOnce(
      response({
        ...settings,
        modal: { configured: false, available: true },
      }) as Response,
    );
    await mount();
    await modal();
    expect(launch().props.disabled).toBe(true);
    await act(async () => {
      passwordFields()[0].props.onChange({ target: { value: "test-id" } });
      passwordFields()[1].props.onChange({ target: { value: "test-secret" } });
    });
    vi.mocked(fetch).mockResolvedValueOnce(
      response({ modal: settings.modal }) as Response,
    );
    await act(async () =>
      root()
        .findByType("form")
        .props.onSubmit({ preventDefault() {} }),
    );
    expect(passwordFields()).toHaveLength(0);
    expect(launch().props.disabled).toBe(false);
    expect(vi.mocked(fetch).mock.calls.at(-1)?.[1]).toMatchObject({
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        token_id: "test-id",
        token_secret: "test-secret",
      }),
    });
  });
  it("fails closed when settings are malformed and disables edits during a run", async () => {
    vi.mocked(fetch).mockResolvedValueOnce(response({}) as Response);
    await mount();
    expect(launch().props.disabled).toBe(true);
    expect(root().findByProps({ role: "alert" })).toBeDefined();
    await act(async () =>
      renderer!.update(
        createElement(TrainingControls, {
          data: { ...idle, status: "running" },
          loading: false,
          pending: false,
          start,
          stop: vi.fn(),
        }),
      ),
    );
    expect(root().findByType("fieldset").props.disabled).toBe(true);
  });
});
describe("configuration validation", () => {
  it("counts the exact parameter budget and enforces numeric limits", () => {
    expect(trainingParameterCount(64, 32)).toBe(21443);
    expect(trainingParameterCount(1024, 512)).toBe(834563);
    expect(
      validTrainingOptions({ ...DEFAULT_TRAINING_OPTIONS, epochs: 51 }),
    ).toBe(false);
    expect(
      validTrainingOptions({ ...DEFAULT_TRAINING_OPTIONS, learning_rate: NaN }),
    ).toBe(false);
    expect(
      validTrainingOptions({ ...DEFAULT_TRAINING_OPTIONS, hidden_1: 8.5 }),
    ).toBe(false);
    expect(
      parseTrainingSettings({
        ...settings,
        modal: { configured: "yes", available: true },
      }),
    ).toBeNull();
  });
  it("stops only an owned cloud run or the legacy local run", () => {
    expect(
      trainingCanStop({ ...idle, status: "running", backend: "modal" }),
    ).toBe(false);
    expect(
      trainingCanStop({
        ...idle,
        status: "running",
        backend: "modal",
        can_stop: true,
      }),
    ).toBe(true);
    expect(
      trainingCanStop({ ...idle, status: "running", can_stop: false }),
    ).toBe(false);
    expect(trainingCanStop({ ...idle, status: "running" })).toBe(true);
  });
});
