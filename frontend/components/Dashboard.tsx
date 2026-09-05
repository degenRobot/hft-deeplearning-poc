import type { MarketGateState } from "../lib/types";
import {
  ExpertsSection,
  GateSection,
  MarketSection,
  PaperSection,
} from "./DashboardSections";
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
      <MarketSection state={state} ready={ready} />
      <GateSection
        state={state}
        ready={ready}
        nextRefresh={nextRefresh}
        mode={mode}
      />
      <ExpertsSection state={state} ready={ready} />
      <PaperSection state={state} ready={ready} />
      <TrainingSection />
    </>
  );
}
