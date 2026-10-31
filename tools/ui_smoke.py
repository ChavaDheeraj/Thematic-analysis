"""Browser smoke check for the four pages (optional; needs `pip install playwright`).

Drives the Microsoft Edge (or Chrome) already installed on the machine, so no
browser download is needed. Start the app first, then:

    python tools/ui_smoke.py http://127.0.0.1:8000 screenshots/

It loads Upload, Dashboard, Graphs & Patterns and Database in light and dark
themes at 1440 px and at 390 px (phone), saves full-length screenshots, and
reports page errors, failed API calls and content wider than the screen.
Exit code 1 if anything is reported.
"""
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

PAGES = ("upload", "dashboard", "graphs", "database")


def main(base: str, out: Path, channel: str = "msedge") -> int:
    out.mkdir(parents=True, exist_ok=True)
    issues: list[str] = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel=channel, headless=True)
        for label, width, theme in (("desktop", 1440, "light"), ("desktop", 1440, "dark"), ("phone", 390, "light")):
            ctx = browser.new_context(viewport={"width": width, "height": 900})
            ctx.add_init_script(f"localStorage.setItem('tml.theme', '{theme}')")
            page = ctx.new_page()
            page.on("pageerror", lambda e: issues.append(f"page error: {e}"))
            page.on("response", lambda r: r.status >= 500 and "/api/" in r.url and issues.append(f"HTTP {r.status} {r.url}"))
            for name in PAGES:
                page.set_viewport_size({"width": width, "height": 900})
                page.goto(f"{base}/{name}")
                page.wait_for_load_state("networkidle")
                page.wait_for_timeout(1200)
                over = page.evaluate("(() => { const m = document.querySelector('main'); return m.scrollWidth - m.clientWidth; })()")
                if over > 0:
                    issues.append(f"{label}/{theme} /{name}: content {over}px wider than the screen")
                # the app scrolls inside <main>; grow the viewport so the screenshot holds the whole page
                height = page.evaluate("document.querySelector('main').scrollHeight + 140")
                page.set_viewport_size({"width": width, "height": min(int(height), 16000)})
                page.wait_for_timeout(800)
                page.screenshot(path=out / f"{name}-{label}-{theme}.png")
            ctx.close()
        browser.close()
    for i in issues:
        print("ISSUE:", i)
    print(f"{len(issues)} issue(s); screenshots in {out}")
    return 1 if issues else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000", Path(sys.argv[2] if len(sys.argv) > 2 else "screenshots")))
