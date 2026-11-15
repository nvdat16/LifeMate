"""
Zalo AI Challenge 2026 - LifeMate Portal Scraper
Step 1: Open browser (headless=False) → wait for human login
Step 2: After confirmation, scrape all tabs and save to docs/lifemate_specs.md
"""

import sys
import time
import json
import re
from pathlib import Path
from playwright.sync_api import sync_playwright, Page, Browser

TARGET_URL = "https://challenge.zalo.ai/portal/lifemate"
OUTPUT_FILE = Path(__file__).resolve().parents[1] / "docs" / "lifemate_specs.md"
STATE_FILE  = Path(__file__).resolve().parent / ".browser_state.json"

# ──────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────

def wait_for_signal(prompt_msg: str) -> str:
    """Pause and wait for user input from terminal."""
    print("\n" + "="*70)
    print(prompt_msg)
    print("="*70)
    return input(">>> ").strip().lower()


def extract_page_text(page: Page) -> str:
    """Extract visible text from the current page."""
    try:
        return page.evaluate("""() => {
            // Remove script/style elements
            const clone = document.body.cloneNode(true);
            clone.querySelectorAll('script, style, noscript').forEach(e => e.remove());
            return clone.innerText || clone.textContent || '';
        }""")
    except Exception as e:
        return f"[Error extracting text: {e}]"


def extract_page_html(page: Page) -> str:
    """Extract full inner HTML for deeper parsing."""
    try:
        return page.content()
    except Exception as e:
        return f"[Error extracting HTML: {e}]"


def get_all_tab_links(page: Page) -> list[dict]:
    """Find all navigation tabs/links on the portal page."""
    tabs = []
    try:
        # Common patterns for Zalo AI portal nav
        nav_selectors = [
            "nav a", ".tab a", ".menu a", ".nav-item a",
            "[role='tab']", ".ant-tabs-tab", ".ant-menu-item a",
            "ul.tabs li a", ".challenge-nav a", ".sidebar a"
        ]
        seen_hrefs = set()
        for sel in nav_selectors:
            links = page.query_selector_all(sel)
            for link in links:
                try:
                    href  = link.get_attribute("href") or ""
                    label = (link.inner_text() or "").strip()
                    if label and href not in seen_hrefs:
                        seen_hrefs.add(href)
                        tabs.append({"label": label, "href": href})
                except Exception:
                    pass
    except Exception as e:
        print(f"[WARN] Tab detection error: {e}")
    return tabs


def click_tab_and_wait(page: Page, label: str, href: str) -> bool:
    """Click a tab by label or href and wait for content."""
    try:
        # Try clicking by text first
        matches = page.get_by_text(label, exact=True).all()
        if matches:
            matches[0].click()
            page.wait_for_load_state("networkidle", timeout=10000)
            time.sleep(1.5)
            return True
        # Fallback: navigate to href
        if href.startswith("http"):
            page.goto(href, wait_until="networkidle")
        elif href.startswith("/"):
            base = "https://challenge.zalo.ai"
            page.goto(base + href, wait_until="networkidle")
        time.sleep(1.5)
        return True
    except Exception as e:
        print(f"[WARN] Could not click tab '{label}': {e}")
        return False


def check_logged_in(page: Page) -> bool:
    """Return True if the user appears to be logged in."""
    try:
        content = page.content().lower()
        # Signs of being logged in
        if any(k in content for k in ["sign out", "logout", "dashboard", "my team", "đăng xuất"]):
            return True
        # Signs of NOT being logged in
        if any(k in content for k in ["sign in", "login", "đăng nhập"]):
            # Check if sign-in button is visible
            btn = page.query_selector("button:has-text('SIGN IN'), a:has-text('SIGN IN'), button:has-text('Login')")
            if btn and btn.is_visible():
                return False
        return True  # Assume logged in if ambiguous
    except Exception:
        return False


# ──────────────────────────────────────────────────────────
# MAIN SCRAPING LOGIC
# ──────────────────────────────────────────────────────────

def scrape_all_content(page: Page) -> dict[str, str]:
    """
    Scrape all tabs on the LifeMate portal.
    Returns dict: {tab_label: text_content}
    """
    results = {}

    # 1. Capture the landing page first
    page.goto(TARGET_URL, wait_until="networkidle")
    time.sleep(2)
    
    print("\n[INFO] Capturing landing page...")
    results["__LANDING__"] = extract_page_text(page)

    # 2. Detect tabs
    tabs = get_all_tab_links(page)
    print(f"[INFO] Detected {len(tabs)} navigation tabs: {[t['label'] for t in tabs]}")

    # 3. Known tabs on Zalo AI portal (try them explicitly if not auto-detected)
    known_tab_labels = [
        "Overview", "Data", "Rules", "Evaluation",
        "Submission", "Submission Guidelines", "Leaderboard",
        "Announcement", "Discussion", "Forum"
    ]

    # Merge detected + known
    all_labels = list({t["label"] for t in tabs} | set(known_tab_labels))

    # 4. Iterate through tabs
    for tab in tabs:
        label = tab["label"]
        href  = tab["href"]
        if not label or len(label) < 2:
            continue
        print(f"\n[INFO] → Visiting tab: '{label}' ({href})")
        clicked = click_tab_and_wait(page, label, href)
        if clicked:
            content = extract_page_text(page)
            results[label] = content
            print(f"    Captured {len(content)} chars.")
        time.sleep(0.5)

    # 5. Also try clicking known tabs by text (catches tabs missed by href detection)
    for label in known_tab_labels:
        if label in results:
            continue
        try:
            elems = page.get_by_text(label, exact=True).all()
            if not elems:
                elems = page.get_by_text(label, exact=False).all()
            for elem in elems:
                if elem.is_visible():
                    print(f"\n[INFO] → Clicking known tab: '{label}'")
                    elem.click()
                    page.wait_for_load_state("networkidle", timeout=8000)
                    time.sleep(1.5)
                    content = extract_page_text(page)
                    results[label] = content
                    print(f"    Captured {len(content)} chars.")
                    break
        except Exception as e:
            print(f"[WARN] Known tab '{label}' not found: {e}")

    return results


def build_markdown(sections: dict[str, str]) -> str:
    """Format scraped sections into a Markdown document."""
    lines = [
        "# Zalo AI Challenge 2026 — LifeMate (Voice Agents) Technical Specs",
        "",
        "> Auto-scraped from https://challenge.zalo.ai/portal/lifemate",
        f"> Scraped at: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}",
        "",
        "---",
        ""
    ]
    for section, content in sections.items():
        display_name = section if section != "__LANDING__" else "Landing Page / Overview"
        lines.append(f"## {display_name}")
        lines.append("")
        # Clean up excessive blank lines
        cleaned = re.sub(r'\n{4,}', '\n\n\n', content.strip())
        lines.append(cleaned)
        lines.append("")
        lines.append("---")
        lines.append("")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────

def main():
    print("\n🚀 Zalo AI LifeMate Portal Scraper — Starting...")

    with sync_playwright() as p:
        # Launch in HEADED mode so user can interact
        browser: Browser = p.chromium.launch(
            headless=False,
            args=["--start-maximized"],
            slow_mo=100
        )
        context = browser.new_context(
            viewport={"width": 1600, "height": 900},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )
        )
        page = context.new_page()

        # ── STEP 1: Navigate to portal ──
        print(f"\n[INFO] Navigating to: {TARGET_URL}")
        page.goto(TARGET_URL, wait_until="networkidle")
        time.sleep(2)

        # ── STEP 1: Check login status ──
        logged_in = check_logged_in(page)
        if not logged_in:
            msg = (
                "\n🌐 Trình duyệt đã mở tại: https://challenge.zalo.ai/portal/lifemate\n\n"
                "⚠️  Trang đang yêu cầu đăng nhập.\n\n"
                "👉 Vui lòng:\n"
                "   1. Đăng nhập vào tài khoản Zalo AI Challenge\n"
                "   2. Nếu cần, bấm 'Register' để tham gia bảng LifeMate\n"
                "   3. Đảm bảo bạn thấy đầy đủ các tab: Overview, Data, Rules, Evaluation, Submission\n\n"
                "Sau khi hoàn tất, nhắn 'done' tại khung chat (hoặc nhập 'done' ở đây) để tiếp tục."
            )
            print(msg)
            # Signal file for external polling
            STATE_FILE.write_text(json.dumps({"status": "waiting_login"}))
            
            # Wait for user signal — either from terminal input
            ans = wait_for_signal(
                "Trình duyệt đã mở. Vui lòng đăng nhập trên cửa sổ trình duyệt,\n"
                "sau đó nhập 'done' ở đây và nhấn Enter để tiếp tục:"
            )
            while ans not in ("done", "xong", "ok", "continue"):
                ans = wait_for_signal("Nhập 'done' khi đã đăng nhập xong:")
        else:
            print("[INFO] Đã đăng nhập sẵn. Tiếp tục scraping...")

        STATE_FILE.write_text(json.dumps({"status": "scraping"}))

        # ── STEP 2: Scrape all tabs ──
        print("\n[INFO] Bắt đầu thu thập nội dung từ tất cả các tab...")
        sections = scrape_all_content(page)

        # ── STEP 3: Save to file ──
        markdown_content = build_markdown(sections)
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_FILE.write_text(markdown_content, encoding="utf-8")
        
        STATE_FILE.write_text(json.dumps({"status": "done", "sections": list(sections.keys())}))
        
        print(f"\n✅ Đã lưu {len(sections)} mục vào: {OUTPUT_FILE}")
        print(f"   Tổng ký tự: {len(markdown_content):,}")
        
        # Keep browser open for verification
        print("\n[INFO] Trình duyệt vẫn mở để bạn kiểm tra. Nhấn Enter để đóng.")
        input()
        browser.close()

    print("\n✅ Hoàn tất! File lưu tại: docs/lifemate_specs.md")


if __name__ == "__main__":
    main()

