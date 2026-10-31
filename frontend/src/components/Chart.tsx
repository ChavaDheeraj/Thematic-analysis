import * as echarts from "echarts";
import { useEffect, useRef, useState } from "react";
import { chrome } from "../colors";
import { useApp } from "../state";

/** Escape text before it goes into an HTML tooltip (ECharts renders custom formatter output as HTML). */
export const esc = (s: unknown): string => echarts.format.encodeHTML(String(s ?? ""));

/** Base chart chrome: recessive hairline grid, muted axis text, tooltip on hover. */
export function baseOption(): any { // eslint-disable-line @typescript-eslint/no-explicit-any
  const c = chrome();
  return {
    backgroundColor: "transparent",
    textStyle: { fontFamily: 'system-ui, -apple-system, "Segoe UI", sans-serif', color: c.ink2 },
    grid: { left: 8, right: 16, top: 28, bottom: 8, containLabel: true },
    tooltip: {
      trigger: "item", confine: true, backgroundColor: c.surface, borderColor: c.grid,
      textStyle: { color: c.ink, fontSize: 12 }, extraCssText: "box-shadow:0 4px 16px rgba(0,0,0,.12);border-radius:6px;",
    },
    legend: { top: 0, left: 0, icon: "roundRect", itemWidth: 10, itemHeight: 10, textStyle: { color: c.ink2, fontSize: 12 } },
  };
}

export function axis(kind: "category" | "value", extra: Record<string, unknown> = {}): any { // eslint-disable-line @typescript-eslint/no-explicit-any
  const c = chrome();
  return {
    type: kind,
    axisLine: { show: kind === "category", lineStyle: { color: c.axis } },
    axisTick: { show: false },
    axisLabel: { color: c.muted, fontSize: 11, hideOverlap: true },
    splitLine: { show: kind === "value", lineStyle: { color: c.grid, width: 1, type: "solid" } },
    ...extra,
  };
}

/** Bar series spec: thin bars with a 4px rounded data end. */
export function barStyle(color: string, horizontal = false) {
  return {
    type: "bar" as const,
    barMaxWidth: 24,
    itemStyle: { color, borderRadius: horizontal ? [0, 4, 4, 0] : [4, 4, 0, 0] },
    emphasis: { focus: "series" as const },
  };
}

export function lineStyle(color: string) {
  return {
    type: "line" as const,
    showSymbol: false,
    symbolSize: 8,
    lineStyle: { width: 2, color, cap: "round" as const, join: "round" as const },
    itemStyle: { color, borderColor: chrome().surface, borderWidth: 2 },
    emphasis: { focus: "series" as const },
  };
}

export function Chart({ option, height = 280, onClick, name = "chart", renderer = "canvas" }: {
  option: object | (() => object);
  height?: number;
  onClick?: (p: any) => void; // eslint-disable-line @typescript-eslint/no-explicit-any
  name?: string;
  renderer?: "canvas" | "svg";
}) {
  const el = useRef<HTMLDivElement>(null);
  const inst = useRef<echarts.ECharts | null>(null);
  const { theme } = useApp();
  const [schemeTick, setSchemeTick] = useState(0);

  // repaint with the selected dark/light steps when the OS scheme changes
  useEffect(() => {
    const mq = window.matchMedia?.("(prefers-color-scheme: dark)");
    const h = () => setSchemeTick((t) => t + 1);
    mq?.addEventListener("change", h);
    return () => mq?.removeEventListener("change", h);
  }, []);

  useEffect(() => {
    if (!el.current) return;
    inst.current = echarts.init(el.current, undefined, { renderer });
    const ro = new ResizeObserver(() => inst.current?.resize());
    ro.observe(el.current);
    return () => { ro.disconnect(); inst.current?.dispose(); inst.current = null; };
  }, [renderer]);

  useEffect(() => {
    const o = typeof option === "function" ? option() : option;
    inst.current?.setOption(o as echarts.EChartsOption, { notMerge: true });
  }, [option, theme, schemeTick]);

  useEffect(() => {
    const i = inst.current;
    if (!i || !onClick) return;
    i.on("click", onClick);
    return () => { i.off("click", onClick); };
  }, [onClick]);

  const save = (type: "png" | "svg") => {
    if (!inst.current) return;
    let url: string;
    if (type === "png") {
      url = inst.current.getDataURL({ type: "png", pixelRatio: 2, backgroundColor: chrome().surface });
    } else {
      // render an SVG copy off-screen so canvas charts can still be exported as vector
      const div = document.createElement("div");
      div.style.cssText = `position:fixed;left:-9999px;width:${el.current!.clientWidth}px;height:${height}px`;
      document.body.appendChild(div);
      const tmp = echarts.init(div, undefined, { renderer: "svg" });
      tmp.setOption({ ...(inst.current.getOption() as object), animation: false });
      url = "data:image/svg+xml;charset=utf-8," + encodeURIComponent(tmp.renderToSVGString());
      tmp.dispose();
      div.remove();
    }
    const a = document.createElement("a");
    a.href = url;
    a.download = `${name}.${type}`;
    a.click();
  };

  return (
    <div className="group relative">
      <div className="absolute right-0 top-0 z-[1] hidden gap-1 group-hover:flex">
        <button className="rounded border border-line bg-surface px-1.5 py-0.5 text-[10px] text-ink2 hover:bg-raised" onClick={() => save("png")}>PNG</button>
        <button className="rounded border border-line bg-surface px-1.5 py-0.5 text-[10px] text-ink2 hover:bg-raised" onClick={() => save("svg")}>SVG</button>
      </div>
      <div ref={el} style={{ height, width: "100%" }} />
    </div>
  );
}
