"""
Zalo AI Challenge 2026 - LifeMate Portal Scraper
Phase 2: Scrape all tabs after login (run after user has logged in)
Sử dụng persistent context hoặc mở session mới với cookies đã được lưu.
"""

import time
import re
import json
from pathlib import Path
from playwright.sync_api import sync_playwright, Page

TARGET_URL   = "https://challenge.zalo.ai/portal/lifemate"
OUTPUT_FILE  = Path(__file__).resolve().parents[1] / "docs" / "lifemate_specs.md"
COOKIES_FILE = Path(__file__).resolve().parent / ".cookies.json"
USER_DATA_DIR = str(Path(__file__).resolve().parent / ".browser_profile")

# ── helpers ──────────────────────────────────────────────

def clean_text(raw: str) -> str:
    raw = re.sub(r'\n{4,}', '\n\n\n', raw.strip())
    raw = re.sub(r'[ \t]{3,}', '  ', raw)
    return raw


def get_text(page: Page) -> str:
    try:
        return page.evaluate("""() => {
            const clone = document.body.cloneNode(true);
            clone.querySelectorAll('script, style, noscript, svg').forEach(e => e.remove());
            return clone.innerText || clone.textContent || '';
        }""")
    except Exception as e:
        return f"[Error: {e}]"


def save_cookies(context, path: Path):
    cookies = context.cookies()
    path.write_text(json.dumps(cookies, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[INFO] Đã lưu {len(cookies)} cookies → {path}")


def load_cookies(context, path: Path):
    if path.exists():
        cookies = json.loads(path.read_text(encoding="utf-8"))
        context.add_cookies(cookies)
        print(f"[INFO] Đã nạp {len(cookies)} cookies từ {path}")
        return True
    return False


def wait_for_user():
    print("\n" + "="*65)
    print("🌐 Trình duyệt đã mở. Vui lòng ĐĂNG NHẬP vào Zalo AI Challenge")
    print("   và tham gia bảng LifeMate để mở khóa toàn bộ tab.")
    print("   Sau đó nhập 'done' và nhấn Enter.")
    print("="*65)
    ans = ""
    while ans.strip().lower() not in ("done", "xong", "ok"):
        ans = input(">>> ")


# ── tab navigation ────────────────────────────────────────

KNOWN_TABS = [
    "Overview", "Data", "Rules", "Evaluation",
    "Submission Guidelines", "Submission", "Leaderboard",
    "Announcement", "Discussion", "Giới thiệu", "Dữ liệu",
    "Quy định", "Đánh giá", "Hướng dẫn nộp bài",
]

def find_and_click_tabs(page: Page) -> list[dict]:
    """Detect all clickable nav tabs on page."""
    tabs = []
    selectors = [
        ".ant-tabs-tab", ".tab-item", "nav a", ".menu-item",
        "[role='tab']", ".nav-link", ".challenge-tab",
        "ul > li > a", ".sidebar-item", ".tab",
    ]
    seen = set()
    for sel in selectors:
        try:
            els = page.query_selector_all(sel)
            for el in els:
                txt = (el.inner_text() or "").strip()
                href = el.get_attribute("href") or ""
                if txt and txt not in seen and len(txt) < 60:
                    seen.add(txt)
                    tabs.append({"label": txt, "href": href, "elem": el})
        except Exception:
            pass
    return tabs


def scrape_tab(page: Page, label: str, href: str = "") -> str:
    """Navigate to a tab and return its text content."""
    try:
        if href and href.startswith("http"):
            page.goto(href, wait_until="networkidle", timeout=15000)
        elif href and href.startswith("/"):
            page.goto("https://challenge.zalo.ai" + href, wait_until="networkidle", timeout=15000)
        else:
            # Try clicking by text
            for locator in [
                page.get_by_role("tab", name=label),
                page.get_by_text(label, exact=True),
                page.get_by_text(label, exact=False),
            ]:
                try:
                    els = locator.all()
                    for el in els:
                        if el.is_visible():
                            el.click()
                            page.wait_for_load_state("networkidle", timeout=10000)
                            break
                    break
                except Exception:
                    pass
        time.sleep(2)
        return clean_text(get_text(page))
    except Exception as e:
        return f"[Error navigating to '{label}': {e}]"


# ── specific URL-based scraping ────────────────────────────

PORTAL_PAGES = [
    ("Overview",              "https://challenge.zalo.ai/portal/lifemate"),
    ("Data",                  "https://challenge.zalo.ai/portal/lifemate/data"),
    ("Rules",                 "https://challenge.zalo.ai/portal/lifemate/rules"),
    ("Evaluation",            "https://challenge.zalo.ai/portal/lifemate/evaluation"),
    ("Submission Guidelines", "https://challenge.zalo.ai/portal/lifemate/submission"),
    ("Leaderboard",           "https://challenge.zalo.ai/portal/lifemate/leaderboard"),
]


def scrape_by_url(page: Page) -> dict[str, str]:
    results = {}
    for label, url in PORTAL_PAGES:
        print(f"\n[→] Scraping: {label}  ({url})")
        try:
            page.goto(url, wait_until="networkidle", timeout=20000)
            time.sleep(2.5)
            content = clean_text(get_text(page))
            results[label] = content
            print(f"    ✓ {len(content):,} chars")
        except Exception as e:
            results[label] = f"[Error: {e}]"
            print(f"    ✗ Error: {e}")
    return results


def scrape_by_tabs(page: Page) -> dict[str, str]:
    """Fallback: detect tabs dynamically and click each."""
    results = {}
    page.goto(TARGET_URL, wait_until="networkidle", timeout=20000)
    time.sleep(2)
    tabs = find_and_click_tabs(page)
    print(f"\n[INFO] Phát hiện {len(tabs)} tab(s): {[t['label'] for t in tabs]}")
    for tab in tabs:
        label = tab["label"]
        href  = tab["href"]
        if label in results:
            continue
        print(f"\n[→] Tab: '{label}'")
        content = scrape_tab(page, label, href)
        results[label] = content
        print(f"    ✓ {len(content):,} chars")
        time.sleep(0.8)
    return results


def build_markdown(sections: dict[str, str]) -> str:
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# Zalo AI Challenge 2026 — LifeMate (Voice Agents) Technical Specs",
        "",
        f"> **Source:** https://challenge.zalo.ai/portal/lifemate  ",
        f"> **Scraped:** {now}",
        "",
        "---",
        "",
    ]
    for title, content in sections.items():
        lines += [f"## {title}", "", content, "", "---", ""]
    return "\n".join(lines)


# ── main ──────────────────────────────────────────────────

def main():
    print("\n🚀 LifeMate Scraper — Phase 2 (Post-Login Scraping)")
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        # Use persistent context so cookies are shared with any
        # existing browser session the user just logged into.
        Path(USER_DATA_DIR).mkdir(parents=True, exist_ok=True)

        context = p.chromium.launch_persistent_context(
            user_data_dir=USER_DATA_DIR,
            headless=False,
            args=["--start-maximized"],
            slow_mo=80,
            viewport=None,
        )
        page = context.new_page()

        # Navigate first to check login state
        print(f"\n[INFO] Navigating to {TARGET_URL} ...")
        page.goto(TARGET_URL, wait_until="networkidle", timeout=20000)
        time.sleep(2)

        # Check login
        content_lower = get_text(page).lower()
        is_logged_in = any(k in content_lower for k in [
            "sign out", "logout", "đăng xuất", "my team", "leaderboard",
            "data", "rules", "evaluation", "submission"
        ]) and not (
            page.query_selector("button:has-text('SIGN IN')") or
            page.query_selector("a:has-text('SIGN IN')")
        )

        if not is_logged_in:
            wait_for_user()

        print("\n[INFO] Bắt đầu thu thập dữ liệu từ tất cả các trang...")

        # Primary: URL-based scraping
        sections = scrape_by_url(page)

        # Supplement: tab-click based (catches dynamic content)
        page.goto(TARGET_URL, wait_until="networkidle", timeout=20000)
        time.sleep(2)
        tab_sections = scrape_by_tabs(page)

        # Merge (URL-based wins if both have same key)
        for k, v in tab_sections.items():
            if k not in sections and len(v) > 100:
                sections[k] = v

        # Build and save
        md = build_markdown(sections)
        OUTPUT_FILE.write_text(md, encoding="utf-8")

        print(f"\n✅ Đã lưu {len(sections)} mục → {OUTPUT_FILE}")
        print(f"   Tổng ký tự: {len(md):,}")
        print("\n[INFO] Nhấn Enter để đóng trình duyệt...")
        input()
        context.close()

    print("\n✅ Xong! Mở docs/lifemate_specs.md để xem kết quả.")


if __name__ == "__main__":
    main()

