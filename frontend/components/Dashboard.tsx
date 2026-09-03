import type { MarketGateState } from "../lib/types";
import {
  ExpertsSection,
  GateSection,
  MarketSection,
  PaperSection,
} from "./DashboardSections";

export function Dashboard({
  state,
  history,
  ready,
  nextRefresh,
  mode,
}: {
  state: MarketGateState;
  history: MarketGateState[];
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
      <ExpertsSection state={state} history={history} ready={ready} />
      <PaperSection state={state} ready={ready} />
    </>
  );
}
