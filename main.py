"""Raptor Review Digest v3: weekly Indianapolis roofing market intelligence.

Scrapes Google Maps for each competitor's review timestamps (so we can compute
accurate 7/30/90-day counts and owner response rates), then calls Claude Opus
with web search to scan the week's local news, profile rising players, and
synthesize specific actions Raptor should take this week.
"""
import json
import os
import re
import smtplib
import ssl
import sys
from datetime import datetime, timezone, timedelta
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


def _parse_relative_days(text):
    """Parse Google Maps relative date strings like '3 days ago' into days."""
    if not text:
        return None
    t = text.lower().strip()
    m = re.search(r"(a|an|\d+)\s+(day|week|month|year)s?\s+ago", t)
    if not m:
        return None
    count_str, unit = m.groups()
    count = 1 if count_str in ("a", "an") else int(count_str)
    return count * {"day": 1, "week": 7, "month": 30, "year": 365}[unit]


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


ANTHROPIC_MODEL_SYNTHESIS = "claude-opus-4-7"
ANTHROPIC_MODEL_RESEARCH = "claude-sonnet-4-6"  # Opus 4.7 refuses tool-using research tasks
WEB_SEARCH_TOOL = {"type": "web_search_20250305", "name": "web_search", "max_uses": 15}


def _anthropic_client():
    """Return an Anthropic client if the API key is set, else None."""
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        return None
    try:
        from anthropic import Anthropic
    except ImportError:
        return None
    return Anthropic(api_key=key)


def _extract_text(response):
    """Return concatenated text blocks from a Claude response that come AFTER the
    last tool call. For pure-synthesis calls (no tools) this is all text. For
    web-search calls, this is the final answer Claude wrote once research finished,
    without the interleaved "I'll search for..." narration before each tool call."""
    last_tool_idx = -1
    for i, block in enumerate(response.content):
        t = getattr(block, "type", None)
        if t in ("server_tool_use", "tool_use", "web_search_tool_result"):
            last_tool_idx = i
    parts = []
    for i, block in enumerate(response.content):
        if i <= last_tool_idx:
            continue
        if getattr(block, "type", None) == "text" and getattr(block, "text", None):
            parts.append(block.text)
    return "\n\n".join(parts).strip()


def ai_market_news(client, competitors):
    """Scan web for newsworthy events involving Indy roofing companies in last 7 days."""
    names = sorted({c["name"] for c in competitors[:20]})
    names_list = ", ".join(names)
    prompt = f"""You are doing market intelligence for Cameron Blakely, owner of Raptor Roofing in Greenwood, Indiana (663 Google reviews, 5.0 stars, ranked #6 by volume in Indianapolis).

Using web search, find specific newsworthy events from the last 7 days involving ANY Indianapolis-area roofing company. Look for:
- Awards, "Best of" rankings (Indianapolis Business Journal, IndyStar, Indy's Best, Center Grove Magazine, etc.)
- Community events, sponsorships, charity work, veteran giveaways, school partnerships
- New office openings, expansions, acquisitions, market entries
- Ad campaigns, marketing launches, website redesigns, rebrands
- Key hires (especially marketing, sales leadership, growth roles)
- Press releases, local news features

Known competitors (but also look beyond this list): {names_list}.

For EACH real event you find, output this EXACT format (no other prose):

---
COMPANY: [name]
EVENT: [one sentence on what happened]
WHEN: [approximate date]
WHY IT MATTERS: [one specific sentence on what Cameron should do or note]
SOURCE: [URL]
---

If you find nothing substantive in the last 7 days, output "No significant news this week." and stop. Do not invent events. Skip generic content like "they offer roofing" or "they have a website"."""
    response = client.messages.create(
        model=ANTHROPIC_MODEL_RESEARCH,
        max_tokens=4000,
        tools=[WEB_SEARCH_TOOL],
        messages=[{"role": "user", "content": prompt}],
    )
    return _extract_text(response)


def ai_rising_profile(client, player):
    """Research a single rising player; small-base high-velocity account."""
    prompt = f"""Research "{player['name']}", an Indianapolis-area roofing company.

Their numbers right now: {player['review_count']} total Google reviews at {player.get('rating', 'n/a')} stars, gained {player.get('reviews_30d', 0)} reviews in the last 30 days. That is high velocity relative to their base, which means something is working.

Using web search, tell Cameron Blakely (owner of Raptor Roofing) in 4-6 sentences:
1. Who they are (ownership, founded, scale, service area)
2. What is driving the recent review acceleration (specific marketing moves, referrals, a particular campaign, storm chasing, etc.)
3. One thing Raptor Roofing could learn or copy
4. One weakness or risk factor Raptor could exploit

Be specific with facts and URLs. No generic filler."""
    response = client.messages.create(
        model=ANTHROPIC_MODEL_RESEARCH,
        max_tokens=2500,
        tools=[WEB_SEARCH_TOOL],
        messages=[{"role": "user", "content": prompt}],
    )
    return _extract_text(response)


def ai_raptor_actions(client, data_summary):
    """Synthesize 3 concrete actions for Raptor based on the week's data."""
    prompt = f"""You are Cameron Blakely's strategic advisor. Cameron owns Raptor Roofing in Greenwood, Indiana. Context:
- 663 Google reviews at 5.0 stars (highest quality rating in Indy top 10 by volume)
- 16 W-2 employees plus 3 sub crews, ~$1.097M Q1 2026 revenue
- Targeting $10M for 2026, $50M long-term
- Key team: Patrick Kinney (co-founder, ops), Curtis (sales), Dylan (CS), Taylor (PM), Lauryn (finance), Mike Kinney (top seller)
- Positioning: premium, communication-obsessed, "we don't outsource accountability"

This week's market data:

{data_summary}

Write 3 specific actions Raptor should take THIS WEEK based on what you see. Each:
- Starts with a verb
- References a specific competitor event or data point
- Names who on Raptor's team owns it
- Is concrete and measurable this week, not a vague suggestion
- 2-3 sentences

Avoid clichés like "focus on reviews" unless the data specifically demands it. If one of the actions is about defending against a specific competitor's move, say so directly."""
    response = client.messages.create(
        model=ANTHROPIC_MODEL_SYNTHESIS,
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}],
    )
    return _extract_text(response)


def detect_rising_players(competitors, min_velocity_ratio=0.08, min_30d=15, max_total=700):
    """Flag small-base high-velocity accounts worth researching."""
    risers = []
    for c in competitors:
        r30 = c.get("reviews_30d") or 0
        total = c.get("review_count") or 0
        if total == 0 or total > max_total:
            continue
        if r30 < min_30d:
            continue
        ratio = r30 / total
        if ratio >= min_velocity_ratio:
            c["_velocity_ratio"] = ratio
            risers.append(c)
    risers.sort(key=lambda c: c["_velocity_ratio"], reverse=True)
    return risers


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
                            "href": page.url,
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


def scrape_review_timeline(page, href):
    """Visit a competitor's detail page, click into Reviews, scroll to load recent
    reviews, and return counts by 7/30/90-day buckets plus owner response rate.

    Returns: {"reviews_7d", "reviews_30d", "reviews_90d", "response_rate", "reviews_loaded"}
    """
    timeline = {"reviews_7d": 0, "reviews_30d": 0, "reviews_90d": 0,
                "response_rate": None, "reviews_loaded": 0}
    try:
        page.goto(href, wait_until="domcontentloaded", timeout=30000)
        page.wait_for_timeout(1500)
        # Click the Reviews tab. Google varies the markup; try several selectors.
        clicked = False
        for selector in [
            'button[role="tab"][aria-label*="Reviews"]',
            'button[role="tab"]:has-text("Reviews")',
            'div[role="tab"][aria-label*="Reviews"]',
            'div[role="tab"]:has-text("Reviews")',
            'a[aria-label*="reviews"]',
            'button:has-text("More reviews")',
        ]:
            try:
                el = page.locator(selector).first
                if el.count() > 0:
                    el.click(timeout=3000)
                    clicked = True
                    break
            except Exception:
                continue
        if not clicked:
            print(f"    reviews tab click failed for {href[:60]}", file=sys.stderr)
            return timeline
        page.wait_for_timeout(2000)
        # Sort by most recent if possible (Google may show by "most relevant" default).
        try:
            sort_btn = page.locator('button[aria-label*="Sort"]').first
            if sort_btn.count() > 0:
                sort_btn.click(timeout=2000)
                page.wait_for_timeout(800)
                newest_opt = page.locator('div[role="menuitemradio"]:has-text("Newest")').first
                if newest_opt.count() > 0:
                    newest_opt.click(timeout=2000)
                    page.wait_for_timeout(1500)
        except Exception:
            pass
        # Scroll the reviews feed to load ~50-100 reviews or until we see >90 day ages.
        scroll_target_js = """
            const el = document.querySelector('div[role="main"] div.m6QErb.DxyBCb, div.m6QErb.DxyBCb');
            if (el) { el.scrollBy(0, 1500); }
        """
        for _ in range(20):
            page.evaluate(scroll_target_js)
            page.wait_for_timeout(500)
        # Extract review cards via JS. Use data-review-id as the canonical wrapper to
        # avoid double-counting when a review has nested divs matching other selectors.
        cards_data = page.evaluate("""() => {
            const cards = Array.from(document.querySelectorAll('[data-review-id]'));
            const seen = new Set();
            const out = [];
            for (const c of cards) {
                const id = c.getAttribute('data-review-id');
                if (!id || seen.has(id)) continue;
                seen.add(id);
                const dateEl = c.querySelector('.rsqaWe, .xRkPPb, [class*="rsqaWe"]');
                const dateText = dateEl ? dateEl.innerText.trim() : '';
                const txt = c.innerText || '';
                const hasResponse = /Response from the owner/i.test(txt);
                if (dateText) out.push({date: dateText, hasResponse});
            }
            return out;
        }""")
        if not cards_data:
            return timeline
        timeline["reviews_loaded"] = len(cards_data)
        responses = 0
        for c in cards_data:
            days = _parse_relative_days(c["date"])
            if days is None:
                continue
            if c["hasResponse"]:
                responses += 1
            if days <= 7:
                timeline["reviews_7d"] += 1
            if days <= 30:
                timeline["reviews_30d"] += 1
            if days <= 90:
                timeline["reviews_90d"] += 1
        timeline["response_rate"] = round(responses / len(cards_data), 2)
    except Exception as e:
        print(f"    timeline error: {e}", file=sys.stderr)
    return timeline


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
            "href": current_url,
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


def render_email(snapshot, prev_snapshot, raptor, businesses, market_news,
                  rising_profiles, raptor_actions, new_entrants):
    """Render the v3 insights-first HTML email."""
    env = Environment(loader=FileSystemLoader(str(ROOT)))
    template = env.get_template("email_template.html")
    is_baseline = prev_snapshot is None
    # Sort businesses for different sections:
    # - by_volume: for the reference leaderboard
    # - by_gain_7d: for the "Biggest gainers this week" table
    by_volume = sorted(businesses, key=lambda b: b["review_count"], reverse=True)
    by_gain_7d = sorted(
        [b for b in businesses if b.get("reviews_7d") is not None],
        key=lambda b: (b.get("reviews_7d") or 0, b.get("reviews_30d") or 0),
        reverse=True,
    )
    raptor_rank = next(
        (i + 1 for i, b in enumerate(by_volume) if "raptor" in b["name"].lower()),
        None,
    )
    return template.render(
        snapshot=snapshot,
        prev_snapshot=prev_snapshot,
        is_baseline=is_baseline,
        raptor=raptor,
        raptor_rank=raptor_rank,
        by_volume=by_volume,
        by_gain_7d=by_gain_7d,
        market_news=market_news,
        rising_profiles=rising_profiles,
        raptor_actions=raptor_actions,
        new_entrants=new_entrants,
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
    load_dotenv(ROOT / ".env", override=True)
    watch = load_watch_list()
    scrape_only = os.getenv("SCRAPE_ONLY") == "1"

    print(f"Scraping '{watch['query']}'...")
    businesses = scrape_google_maps(watch["query"], top_n=watch.get("discover_top_n", 40))
    print(f"scraped {len(businesses)} businesses")
    if not businesses:
        print("no businesses scraped; aborting", file=sys.stderr)
        sys.exit(1)

    # Timeline scraping: for top 20 by volume, load reviews tab and parse timestamps.
    businesses.sort(key=lambda b: b["review_count"], reverse=True)
    print(f"Scraping review timelines for top {min(20, len(businesses))}...")
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
        for i, biz in enumerate(businesses[:20], 1):
            href = biz.get("href")
            if not href:
                continue
            t = scrape_review_timeline(page, href)
            biz.update(t)
            print(f"  [{i}] {biz['name']}: 7d={t['reviews_7d']} 30d={t['reviews_30d']} 90d={t['reviews_90d']} resp={t['response_rate']} loaded={t['reviews_loaded']}")
        browser.close()

    # Snapshot bookkeeping + deltas (kept as secondary signal now that timelines are primary).
    history = load_history()
    businesses = compute_weekly_deltas(businesses, history)
    velocity_leaders = rank_velocity(businesses)
    new_entrants = find_new_entrants(businesses, history)
    prev_snapshot = history["snapshots"][-1] if history["snapshots"] else None
    compute_rank_changes(businesses, prev_snapshot)

    snapshot = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "businesses": businesses,
    }

    # Intelligence layer: call Claude Opus for market news, rising profiles, actions.
    market_news = None
    rising_profiles = []
    raptor_actions = None
    if not scrape_only:
        client = _anthropic_client()
        if client is None:
            print("no ANTHROPIC_API_KEY; skipping intelligence layer")
        else:
            try:
                print("AI: scanning market news...")
                market_news = ai_market_news(client, businesses)
            except Exception as e:
                print(f"  market news failed: {e}", file=sys.stderr)

            risers = detect_rising_players(businesses)
            for r in risers[:2]:
                try:
                    print(f"AI: profiling riser {r['name']}...")
                    profile = ai_rising_profile(client, r)
                    rising_profiles.append({"competitor": r, "profile": profile})
                except Exception as e:
                    print(f"  rising profile failed for {r['name']}: {e}", file=sys.stderr)

            # Synthesis
            try:
                summary = _compose_ai_context(businesses, market_news, rising_profiles)
                print("AI: synthesizing Raptor actions...")
                raptor_actions = ai_raptor_actions(client, summary)
            except Exception as e:
                print(f"  actions synthesis failed: {e}", file=sys.stderr)

    raptor = next(
        (b for b in businesses if "raptor" in b["name"].lower()), None
    )
    html = render_email(
        snapshot=snapshot,
        prev_snapshot=prev_snapshot,
        raptor=raptor,
        businesses=businesses,
        market_news=market_news,
        rising_profiles=rising_profiles,
        raptor_actions=raptor_actions,
        new_entrants=new_entrants,
    )

    date_tag = snapshot["timestamp"][:10]
    is_baseline = prev_snapshot is None
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


def _compose_ai_context(businesses, market_news, rising_profiles):
    """Compact summary string to feed into the raptor_actions synthesis prompt."""
    top10 = businesses[:10]
    rows = []
    for b in top10:
        rows.append(
            f"- {b['name']}: {b['review_count']} total, {b.get('rating', 'n/a')} stars, "
            f"+{b.get('reviews_7d', 0)} last 7d, +{b.get('reviews_30d', 0)} last 30d, "
            f"response rate {b.get('response_rate')}"
        )
    lines = ["**Top 10 by review volume and velocity:**"] + rows
    if market_news:
        lines += ["", "**Market news this week:**", market_news]
    if rising_profiles:
        lines += ["", "**Rising players profiled:**"]
        for rp in rising_profiles:
            lines += [f"- {rp['competitor']['name']}: {rp['profile'][:400]}..."]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
