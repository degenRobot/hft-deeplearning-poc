import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { expect, it } from "vitest";
import { TrainingLab } from "../components/TrainingLab";

it("puts learning charts before one collapsed data and run settings disclosure", () => {
  const html = renderToStaticMarkup(createElement(TrainingLab));
  const settings = html.indexOf('class="training-settings"');
  expect(settings).toBeGreaterThan(html.indexOf("30 frames × 10 features"));
  expect(html.indexOf("30 frames × 10 features")).toBeGreaterThan(0);
  expect(settings).toBeGreaterThan(
    html.indexOf("What changed in the weights?"),
  );
  expect(html.indexOf("What changed in the weights?")).toBeGreaterThan(0);
  expect(html.match(/<details class="training-settings"/g)).toHaveLength(1);
  expect(html).toContain('<details class="training-settings"><summary>');
  expect(html.indexOf("Prepare the data")).toBeGreaterThan(settings);
  expect(html.indexOf("Configure the run")).toBeGreaterThan(settings);
});
