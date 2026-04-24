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


def _norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower().replace("&", "and"))


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

        # Second pass: scrape every pinned competitor, even if also auto-discovered.
        # For multi-location businesses like Bone Dry, the pinned search finds the HQ
        # (most reviews) while auto-discovery may only surface a branch. After both
        # passes we dedup by name and keep whichever result has the higher review count.
        watch = load_watch_list()
        pinned = watch.get("pinned_competitors", [])
        for pin in pinned:
            pname = pin["name"]
            # Use custom search_query if given; otherwise default to name + Indianapolis.
            query_text = pin.get("search_query") or f"{pname} Indianapolis"
            search_url = f"https://www.google.com/maps/search/{query_text.replace(' ', '+')}"
            try:
                page.goto(search_url, wait_until="domcontentloaded", timeout=30000)
                page.wait_for_timeout(2000)
                # Sometimes a single-place result loads directly.
                top_links = page.locator('a.hfpxzc').all()
                if not top_links:
                    text = page.locator("body").inner_text(timeout=5000)
                    combo = re.search(r"(\d\.\d)\s*\n?\s*\((\d[\d,]*)\)", text)
                    if combo:
                        place_id = _extract_place_id(page.url)
                        results.append({
                            "name": pname,
                            "rating": float(combo.group(1)),
                            "review_count": int(combo.group(2).replace(",", "")),
                            "place_id": place_id,
                            "address": None,
                        })
                        print(f"  pinned {pname}: {combo.group(1)} stars, {combo.group(2)} reviews (direct)")
                    continue
                # Collect top 3 hrefs, visit each, pick the one with most reviews. This catches
                # multi-location businesses like Bone Dry where the HQ has far more reviews than a branch.
                hrefs = []
                for link in top_links[:3]:
                    href = link.get_attribute("href") or ""
                    if href:
                        hrefs.append(href)
                detail_results = []
                for href in hrefs:
                    data = _extract_detail(page, href)
                    if data is None:
                        continue
                    detail_results.append(data)
                if not detail_results:
                    continue
                best = max(detail_results, key=lambda d: d["review_count"])
                results.append({"name": pname, **best})
                print(f"  pinned {pname}: {best['rating']} stars, {best['review_count']} reviews")
            except Exception as e:
                print(f"  pinned {pname}: {e}", file=sys.stderr)

        browser.close()

    # Pinned direct-redirect results often lack a place_id because Google's place page
    # URL format sometimes drops the !1s parameter. Back-fill missing place_ids by
    # matching on normalized name against records that DID capture one.
    name_to_pid = {}
    for biz in results:
        if biz.get("place_id"):
            name_to_pid.setdefault(_norm(biz["name"]), biz["place_id"])
    for biz in results:
        if not biz.get("place_id"):
            biz["place_id"] = name_to_pid.get(_norm(biz["name"]))

    # Dedup by place_id (stable) when available, falling back to normalized name.
    # Keep whichever record has the higher review count (pinned searches often find HQ).
    merged = {}
    for biz in results:
        key = biz.get("place_id") or f"name:{_norm(biz['name'])}"
        if key not in merged or biz["review_count"] > merged[key]["review_count"]:
            merged[key] = biz
    return list(merged.values())


def _extract_place_id(url):
    """Pull Google's stable place identifier from a Maps URL (the !1s... parameter)."""
    m = re.search(r"!1s([^!?]+)", url or "")
    return m.group(1) if m else None


def _extract_detail(page, href):
    """Navigate to a place detail URL and extract rating, review count, address, place_id."""
    try:
        page.goto(href, wait_until="domcontentloaded", timeout=30000)
        page.wait_for_timeout(1500)
        current_url = page.url
        place_id = _extract_place_id(current_url) or _extract_place_id(href)
        text = page.locator("body").inner_text(timeout=5000)
        combo = re.search(r"(\d\.\d)\s*\n?\s*\((\d[\d,]*)\)", text)
        rating, review_count = None, 0
        if combo:
            rating = float(combo.group(1))
            review_count = int(combo.group(2).replace(",", ""))
        elif re.search(r"No reviews", text, re.IGNORECASE):
            pass
        else:
            return None
        addr_match = re.search(
            r"\n(\d+\s+[A-Z][^\n]+?,\s*(?:Indianapolis|Greenwood|Carmel|Fishers|Zionsville|Avon|Plainfield|Noblesville|Westfield|Brownsburg|Beech Grove|Southport|Speedway|Lawrence)[^\n]*)",
            text,
        )
        address = addr_match.group(1).strip() if addr_match else None
        return {
            "rating": rating,
            "review_count": review_count,
            "address": address,
            "place_id": place_id,
        }
    except Exception as e:
        print(f"  detail extract error: {e}", file=sys.stderr)
    return None


def compute_weekly_deltas(current, history):
    """Match current businesses to last week's. STRICT pid matching when available:
    if this week's record has a place_id, only match to a prior record with the same
    place_id. This prevents the geolocation-variance bug where 'Bone Dry Roofing' in
    auto-discovery meant different physical locations across runs. Only fall back to
    name matching when the current record has no place_id."""
    if not history["snapshots"]:
        for biz in current:
            biz["delta_reviews"] = None
            biz["prev_rating"] = None
        return current
    last = history["snapshots"][-1]
    by_pid = {b["place_id"]: b for b in last["businesses"] if b.get("place_id")}
    by_name = {b["name"].lower(): b for b in last["businesses"]}
    for biz in current:
        pid = biz.get("place_id")
        if pid:
            prev = by_pid.get(pid)
        else:
            prev = by_name.get(biz["name"].lower())
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
    last = history["snapshots"][-1]["businesses"]
    last_pids = {b["place_id"] for b in last if b.get("place_id")}
    last_names = {b["name"].lower() for b in last}
    out = []
    for biz in current:
        pid = biz.get("place_id")
        if pid and pid in last_pids:
            continue
        if biz["name"].lower() in last_names:
            continue
        out.append(biz)
    return out


def rank_velocity(current):
    with_delta = [b for b in current if b.get("delta_reviews") is not None]
    return sorted(with_delta, key=lambda b: b["delta_reviews"] or 0, reverse=True)


def generate_competitive_reads(snapshot, raptor):
    """Produce a 2-3 sentence 'read' per top-5 competitor. Data-driven observations
    about their positioning relative to Raptor."""
    reads = []
    top = snapshot["businesses"][:5]
    raptor_count = raptor["review_count"] if raptor else 663
    raptor_rating = raptor["rating"] if raptor else 5.0
    for biz in top:
        if "raptor" in biz["name"].lower():
            continue
        name = biz["name"]
        count = biz["review_count"]
        rating = biz["rating"] or 0
        delta = biz.get("delta_reviews")
        gap = count - raptor_count
        parts = []
        # Scale context
        if count > 3 * raptor_count:
            parts.append(f"Sits {gap:,} reviews ahead of Raptor. That volume almost certainly reflects a mature lead-gen machine (LSA max, aggressive canvassing, or multi-location branding).")
        elif count > 1.5 * raptor_count:
            parts.append(f"Leads Raptor by {gap:,} reviews. Within reach over 12 to 18 months of consistent closeout discipline.")
        else:
            parts.append(f"Only {gap:,} reviews ahead of Raptor. This is the nearest rank target.")
        # Rating context
        if rating < raptor_rating - 0.2:
            parts.append(f"Their {rating} rating sits below Raptor's {raptor_rating}, which is a real opening. Use the rating gap in LSA copy and sales-call intros.")
        elif rating < raptor_rating:
            parts.append(f"Slightly behind Raptor's {raptor_rating} at {rating}. Not a strong differentiator unless you hammer it.")
        else:
            parts.append(f"Matches Raptor on quality ({rating} stars). Differentiation has to come from systems, communication, or warranty.")
        # Velocity context if available
        if delta is not None and delta > 0:
            weekly_pace = delta
            annual = weekly_pace * 52
            parts.append(f"Gained +{delta} reviews this week (~{annual:,}/year at that pace).")
        reads.append({"name": name, "read": " ".join(parts)})
    return reads


def generate_focus_actions(snapshot, velocity_leaders, new_entrants, raptor):
    """Three concrete actions for the week, grounded in the data we have."""
    actions = []
    # Action 1: Raptor's velocity
    if raptor and raptor.get("delta_reviews") is not None:
        delta = raptor["delta_reviews"]
        top3_delta = sum((b.get("delta_reviews") or 0) for b in velocity_leaders[:3] if "raptor" not in b["name"].lower())
        top3_avg = top3_delta / 3 if top3_delta else 0
        if delta < top3_avg - 2:
            actions.append(
                f"Raptor gained {delta} reviews this week; the top 3 non-Raptor gainers averaged {top3_avg:.0f}. "
                "That gap is the single biggest factor holding Raptor at #6. Tighten closeout ask discipline: Dylan and Taylor ask at handoff, rep verifies landed, PM form captures confirmation."
            )
        elif delta == 0 and len([b for b in snapshot["businesses"] if (b.get("delta_reviews") or 0) > 0]) > 3:
            actions.append(
                "Raptor added zero reviews this week while 4+ competitors gained ground. "
                "Check Podium review-request flow; if it's firing but customers aren't clicking, the issue is timing or messaging, not the tool."
            )
        else:
            actions.append(
                f"Raptor added {delta} reviews this week, keeping pace with the top gainers. Keep the closeout cadence humming."
            )
    # Action 2: Top accelerator intel
    non_raptor_gainers = [b for b in velocity_leaders if "raptor" not in b["name"].lower()][:1]
    if non_raptor_gainers:
        top = non_raptor_gainers[0]
        actions.append(
            f"Biggest accelerator this week: {top['name']} at +{top['delta_reviews']}. "
            "Pull 5 of their most recent reviews (5 minutes on Google Maps). If a theme repeats (speed, cleanup, insurance help), that is what customers are responding to in market right now. Have Curtis work the theme into next week's RAPTOR WINS objection handling."
        )
    # Action 3: New entrants or unusual movement
    if new_entrants:
        n = new_entrants[0]
        rating_txt = f"at {n['rating']} stars" if n.get("rating") else ""
        actions.append(
            f"New in top results this week: {n['name']} ({n['review_count']:,} reviews {rating_txt}). "
            "Check if they are running LSA, Google Ads, or recent PR. If they just showed up, something changed in their marketing posture worth understanding."
        )
    return actions


def generate_recommendations(snapshot, velocity_leaders, new_entrants):
    """Kept for backward compatibility; delegates to focus_actions."""
    raptor = next(
        (b for b in snapshot["businesses"] if "raptor" in b["name"].lower()), None
    )
    return generate_focus_actions(snapshot, velocity_leaders, new_entrants, raptor)


def compute_rank_changes(current_sorted, prev_snapshot):
    """Annotate each biz with rank_change (positive = moved up in ranking)."""
    if prev_snapshot is None:
        for biz in current_sorted:
            biz["rank_change"] = None
        return
    prev_sorted = sorted(
        prev_snapshot["businesses"], key=lambda b: b["review_count"], reverse=True
    )
    prev_rank = {b["name"].lower(): i + 1 for i, b in enumerate(prev_sorted)}
    for i, biz in enumerate(current_sorted):
        new_rank = i + 1
        old_rank = prev_rank.get(biz["name"].lower())
        biz["rank_change"] = (old_rank - new_rank) if old_rank else None


def render_email(snapshot, prev_snapshot, new_entrants, velocity_leaders, recommendations):
    env = Environment(loader=FileSystemLoader(str(ROOT)))
    template = env.get_template("email_template.html")
    is_baseline = prev_snapshot is None
    top15 = snapshot["businesses"][:15]
    total_reviews_gained = sum(
        (b.get("delta_reviews") or 0) for b in top15
    ) if not is_baseline else 0
    raptor = next(
        (b for b in snapshot["businesses"] if "raptor" in b["name"].lower()), None
    )
    raptor_gain = (raptor.get("delta_reviews") or 0) if (raptor and not is_baseline) else 0
    raptor_rank = next(
        (i + 1 for i, b in enumerate(snapshot["businesses"]) if "raptor" in b["name"].lower()),
        None,
    )
    # Velocity section: show top 5 by delta, but always include Raptor even if not in top 5
    velocity_display = velocity_leaders[:5]
    if raptor and raptor not in velocity_display and raptor.get("delta_reviews") is not None:
        velocity_display = list(velocity_display) + [raptor]
    # Competitive reads on top 5
    competitive_reads = generate_competitive_reads(snapshot, raptor) if not is_baseline or True else []
    # Sanity check: suspiciously large deltas (>50% of total reviews) suggest a data match bug.
    # Flag any and suppress them from velocity to avoid the 1,467 Bone Dry phantom.
    for biz in snapshot["businesses"]:
        dr = biz.get("delta_reviews")
        if dr is not None and biz.get("review_count", 0) > 0 and dr > biz["review_count"] * 0.5 and dr > 20:
            biz["delta_reviews"] = None
            biz["suspect_delta"] = True
    velocity_leaders = [b for b in velocity_leaders if b.get("delta_reviews") is not None]
    velocity_display = [b for b in velocity_display if b.get("delta_reviews") is not None]
    return template.render(
        snapshot=snapshot,
        prev_snapshot=prev_snapshot,
        is_baseline=is_baseline,
        new_entrants=new_entrants,
        velocity_leaders=velocity_leaders,
        velocity_display=velocity_display,
        competitive_reads=competitive_reads,
        recommendations=recommendations,
        total_reviews_gained=total_reviews_gained,
        raptor_gain=raptor_gain,
        raptor_rank=raptor_rank,
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
    prev_snapshot = history["snapshots"][-1] if history["snapshots"] else None
    compute_rank_changes(businesses, prev_snapshot)

    snapshot = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "businesses": businesses,
    }
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
