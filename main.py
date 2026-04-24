"""Raptor Review Digest: weekly competitive intel scraper and emailer.

Scrapes Google Maps for Indianapolis roofing competitors, computes
week-over-week review velocity, and emails an HTML digest to Cameron.
"""
import json
import os
import re
import smtplib
import ssl
import sys
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
WATCH_LIST_PATH = DATA_DIR / "watch_list.json"
HISTORY_PATH = DATA_DIR / "history.json"


def load_watch_list():
    with open(WATCH_LIST_PATH) as f:
        return json.load(f)


def load_history():
    if not HISTORY_PATH.exists():
        return {"snapshots": []}
    with open(HISTORY_PATH) as f:
        return json.load(f)


def save_history(history):
    DATA_DIR.mkdir(exist_ok=True)
    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)


def scrape_google_maps(query, top_n=20):
    """Scrape Google Maps for the query. Visits each result's detail page to get
    review count, which is no longer present on the list-view cards."""
    results = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1400, "height": 900},
        )
        page = context.new_page()
        url = f"https://www.google.com/maps/search/{query.replace(' ', '+')}"
        page.goto(url, wait_until="domcontentloaded", timeout=30000)
        try:
            page.wait_for_selector('div[role="feed"]', timeout=15000)
        except Exception:
            print("feed selector never appeared; saving debug screenshot", file=sys.stderr)
            page.screenshot(path=str(ROOT / "debug_feed.png"))
            browser.close()
            return results

        feed = page.locator('div[role="feed"]')
        for _ in range(8):
            feed.evaluate("el => el.scrollBy(0, 1200)")
            page.wait_for_timeout(1000)

        # Collect name and href from every result anchor.
        links = page.locator('div[role="feed"] a.hfpxzc').all()
        items = []
        seen_names = set()
        for link in links:
            name = (link.get_attribute("aria-label") or "").strip()
            href = link.get_attribute("href") or ""
            if not name or not href:
                continue
            if name.lower() in seen_names:
                continue
            seen_names.add(name.lower())
            items.append({"name": name, "href": href})
            if len(items) >= top_n:
                break
        print(f"collected {len(items)} candidate businesses")

        # Visit each detail page and extract rating and review count.
        for i, item in enumerate(items, 1):
            data = _extract_detail(page, item["href"])
            if data is None:
                print(f"  [{i}/{len(items)}] skip {item['name']}: no data", file=sys.stderr)
                continue
            print(f"  [{i}/{len(items)}] {item['name']}: {data['rating']} stars, {data['review_count']} reviews")
            results.append({"name": item["name"], **data})

        # Second pass: scrape pinned competitors that auto-discovery missed.
        watch = load_watch_list()
        pinned = watch.get("pinned_competitors", [])
        def norm(s):
            return re.sub(r"[^a-z0-9]", "", s.lower().replace("&", "and"))
        found_norms = {norm(r["name"]) for r in results}
        for pin in pinned:
            pname = pin["name"]
            pnorm = norm(pname)
            if any(pnorm in f or f in pnorm for f in found_norms):
                continue
            search_url = f"https://www.google.com/maps/search/{pname.replace(' ', '+')}"
            try:
                page.goto(search_url, wait_until="domcontentloaded", timeout=30000)
                page.wait_for_timeout(2000)
                # Take first result anchor.
                first_link = page.locator('a.hfpxzc').first
                if first_link.count() == 0:
                    # Sometimes a single-place result loads directly.
                    text = page.locator("body").inner_text(timeout=5000)
                    combo = re.search(r"(\d\.\d)\s*\n?\s*\((\d[\d,]*)\)", text)
                    if combo:
                        results.append({
                            "name": pname,
                            "rating": float(combo.group(1)),
                            "review_count": int(combo.group(2).replace(",", "")),
                        })
                        print(f"  pinned {pname}: {combo.group(1)} stars, {combo.group(2)} reviews (direct)")
                    continue
                href = first_link.get_attribute("href") or ""
                if not href:
                    continue
                data = _extract_detail(page, href)
                if data is None:
                    continue
                results.append({"name": pname, **data})
                print(f"  pinned {pname}: {data['rating']} stars, {data['review_count']} reviews")
            except Exception as e:
                print(f"  pinned {pname}: {e}", file=sys.stderr)

        browser.close()
    return results


def _extract_detail(page, href):
    """Navigate to a place detail URL and extract rating + review count."""
    try:
        page.goto(href, wait_until="domcontentloaded", timeout=30000)
        page.wait_for_timeout(1500)
        text = page.locator("body").inner_text(timeout=5000)
        combo = re.search(r"(\d\.\d)\s*\n?\s*\((\d[\d,]*)\)", text)
        if combo:
            return {
                "rating": float(combo.group(1)),
                "review_count": int(combo.group(2).replace(",", "")),
            }
        if re.search(r"No reviews", text, re.IGNORECASE):
            return {"rating": None, "review_count": 0}
    except Exception as e:
        print(f"  detail extract error: {e}", file=sys.stderr)
    return None


def compute_weekly_deltas(current, history):
    """Add week-over-week deltas by name-matching against the last snapshot."""
    if not history["snapshots"]:
        for biz in current:
            biz["delta_reviews"] = None
            biz["prev_rating"] = None
        return current
    last = history["snapshots"][-1]
    last_map = {b["name"].lower(): b for b in last["businesses"]}
    for biz in current:
        prev = last_map.get(biz["name"].lower())
        if prev:
            biz["delta_reviews"] = biz["review_count"] - prev["review_count"]
            biz["prev_rating"] = prev["rating"]
        else:
            biz["delta_reviews"] = None
            biz["prev_rating"] = None
    return current


def find_new_entrants(current, history):
    if not history["snapshots"]:
        return []
    last_names = {b["name"].lower() for b in history["snapshots"][-1]["businesses"]}
    return [b for b in current if b["name"].lower() not in last_names]


def rank_velocity(current):
    with_delta = [b for b in current if b.get("delta_reviews") is not None]
    return sorted(with_delta, key=lambda b: b["delta_reviews"] or 0, reverse=True)


def generate_recommendations(snapshot, velocity_leaders, new_entrants):
    """Heuristic recommendations. Upgrade to Claude-generated in v2."""
    recs = []
    top = [b for b in velocity_leaders if "raptor" not in b["name"].lower()][:3]
    if top:
        names = ", ".join(b["name"] for b in top)
        recs.append(
            f"Top accelerators this week: {names}. "
            "Pull 3 of their most recent reviews and look for themes Raptor can match or counter."
        )
    if new_entrants:
        names = ", ".join(b["name"] for b in new_entrants[:3])
        recs.append(
            f"New competitor(s) surfaced in top results: {names}. "
            "Worth a closer look to see what is driving their visibility."
        )
    raptor = next(
        (b for b in snapshot["businesses"] if "raptor" in b["name"].lower()), None
    )
    if raptor and raptor.get("delta_reviews") is not None:
        delta = raptor["delta_reviews"]
        if delta < 5:
            recs.append(
                f"Raptor added {delta} reviews this week. Low velocity relative to the leaderboard. "
                "Tighten the closeout ask (Dylan and Taylor) and confirm the PM form captures review confirmation."
            )
        else:
            recs.append(
                f"Raptor gained {delta} reviews this week. Keep the closeout cadence humming."
            )
    return recs


def render_email(snapshot, prev_snapshot, new_entrants, velocity_leaders, recommendations):
    env = Environment(loader=FileSystemLoader(str(ROOT)))
    template = env.get_template("email_template.html")
    return template.render(
        snapshot=snapshot,
        prev_snapshot=prev_snapshot,
        new_entrants=new_entrants,
        velocity_leaders=velocity_leaders,
        recommendations=recommendations,
        run_date=datetime.now(timezone.utc).strftime("%B %d, %Y"),
    )


def send_email(html_body, subject):
    user = os.getenv("GMAIL_USER")
    pw = os.getenv("GMAIL_APP_PASSWORD")
    recipient = os.getenv("RECIPIENT_EMAIL")
    if not all([user, pw, recipient]):
        raise RuntimeError("Missing Gmail credentials in environment")
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = f"Raptor Review Digest <{user}>"
    msg["To"] = recipient
    msg.attach(MIMEText(html_body, "html"))
    ctx = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ctx) as server:
        server.login(user, pw.replace(" ", ""))
        server.send_message(msg)


def main():
    load_dotenv(ROOT / ".env")
    watch = load_watch_list()
    print(f"Scraping '{watch['query']}'...")
    businesses = scrape_google_maps(watch["query"], top_n=watch.get("discover_top_n", 20))
    print(f"scraped {len(businesses)} businesses")

    if not businesses:
        print("no businesses scraped; aborting", file=sys.stderr)
        sys.exit(1)

    history = load_history()
    businesses = compute_weekly_deltas(businesses, history)
    velocity_leaders = rank_velocity(businesses)
    new_entrants = find_new_entrants(businesses, history)
    businesses.sort(key=lambda b: b["review_count"], reverse=True)

    snapshot = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "businesses": businesses,
    }
    prev_snapshot = history["snapshots"][-1] if history["snapshots"] else None
    recommendations = generate_recommendations(snapshot, velocity_leaders, new_entrants)

    html = render_email(snapshot, prev_snapshot, new_entrants, velocity_leaders, recommendations)

    is_baseline = prev_snapshot is None
    date_tag = snapshot["timestamp"][:10]
    subject = (
        f"Raptor Review Digest | Baseline {date_tag}"
        if is_baseline
        else f"Raptor Review Digest | Week of {date_tag}"
    )

    if os.getenv("SKIP_EMAIL") == "1":
        (ROOT / "preview.html").write_text(html)
        print("skipped email (SKIP_EMAIL=1); wrote preview.html")
    else:
        send_email(html, subject)
        print(f"emailed: {subject}")

    history["snapshots"].append(snapshot)
    history["snapshots"] = history["snapshots"][-26:]
    save_history(history)


if __name__ == "__main__":
    main()
