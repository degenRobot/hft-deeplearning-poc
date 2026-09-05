import { TerminalLearning } from "./LiveLearning";
import type { MarketGateState } from "../lib/types";
import {
  ExpertsSection,
  GateSection,
  MarketSection,
  PaperSection,
} from "./DashboardSections";
import { SignalFlow } from "./SignalFlow";
import { TrainingSection } from "./TrainingSection";

export function Dashboard({
  state,
  ready,
  nextRefresh,
  mode,
}: {
  state: MarketGateState;
  ready: boolean;
  nextRefresh: number;
  mode?: MarketGateState["gate"]["mode"];
}) {
  return (
    <>
      <TerminalLearning />
      <SignalFlow state={state} ready={ready} nextRefresh={nextRefresh} />
      <details className="inspect-details">
        <summary>Inspect numeric state &amp; risk diagnostics</summary>
        <MarketSection state={state} ready={ready} />
        <GateSection
          state={state}
          ready={ready}
          nextRefresh={nextRefresh}
          mode={mode}
        />
        <ExpertsSection state={state} ready={ready} />
        <PaperSection state={state} ready={ready} />
      </details>
      <details className="inspect-details">
        <summary>Offline experiments &amp; training receipts</summary>
        <TrainingSection />
      </details>
    </>
  );
}
