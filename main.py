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

import markdown as md
from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader
from markupsafe import Markup
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
DOCS_DIR = ROOT / "docs"
DASHBOARD_TEMPLATE_DIR = ROOT / "dashboard"
WATCH_LIST_PATH = DATA_DIR / "watch_list.json"
HISTORY_PATH = DATA_DIR / "history.json"
DASHBOARD_PUBLIC_URL = os.getenv(
    "DASHBOARD_URL",
    "https://camroofsindy.github.io/raptor-review-digest/",
)


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
    last tool call, with narration preambles stripped."""
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
    return _strip_ai_preamble("\n\n".join(parts)).strip()


def ai_market_news(client, competitors):
    """Scan for newsworthy events affecting the Indy roofing market and the broader
    industry over the last 7 days. Mix local roofing-company moves with national
    industry shifts (insurance regulation, code changes, manufacturer announcements,
    supply-chain news) when those are bigger stories."""
    names = sorted({c["name"] for c in competitors[:20]})
    names_list = ", ".join(names)
    prompt = f"""You are the Chief Marketing Officer for Raptor Roofing — think of yourself as a $500K/year CMO who reads everything and surfaces only what matters. Cameron Blakely owns Raptor Roofing in Greenwood Indiana (663 Google reviews at 5.0 stars). He gets a digest twice a week and only wants stories that change a decision.

Use web search aggressively (8-12 searches across different angles). Find up to TEN newsworthy stories from the last 7 days. Mix local Indy roofing events AND national/industry stories that affect the roofing business. Both matter.

LOCAL events to look for:
- Awards or "Best Of" rankings (IBJ, IndyStar, Indy's Best, Center Grove Magazine)
- Community events, sponsorships, charity, veteran giveaways
- Office openings, expansions, acquisitions in Indy metro
- Ad campaign launches, website redesigns, rebrands
- Key hires (marketing, sales, growth)
- Local press features

NATIONAL / INDUSTRY events to look for (these often matter MORE than local):
- Insurance regulation changes (e.g. Fannie Mae / Freddie Mac roof depreciation rule changes, ACV vs RCV policy shifts, state insurance commissioner rulings)
- IRS / tax code changes affecting commercial roofing depreciation
- Manufacturer announcements from GAF, Owens Corning, CertainTeed, IKO (warranty changes, product launches, recalls)
- Trade-association news (NRCA, RCAT, Roofing Contractors Association of Texas, MidWest Roofing)
- Supply-chain shifts (asphalt prices, shingle availability, labor market)
- AI / search shifts that affect how homeowners find roofers (Google AI Overviews, ChatGPT integration with maps, etc)
- Storm and weather events affecting Indiana / Midwest

Known Indy competitors to spot-check: {names_list}.

BIAS: prefer LOCAL Indy roofing stories when they exist, then mix in industry stories. Aim for at least 4 local items if any are available; fill the rest with industry. If local is genuinely empty this week, default to industry-heavy.

For EACH real story you find (max 10), output EXACTLY this format:

---
COMPANY: [company name OR "Industry-wide" for non-company-specific stories]
CATEGORY: [one of: award, press, expansion, hire, ad, regulation, weather, supply, partnership, other]
EVENT: [one concrete sentence on what happened — lead with the most factual hook]
WHEN: [approximate date]
WHY IT MATTERS: [one CMO-grade sentence on what Cameron should DO or NOTE]
DETAIL: [3-5 sentences expanding on the WHY — context, implication, what tactic Cameron could run, what to monitor next. This is what shows when Cameron clicks the news card to read the full story.]
SOURCE: [URL]
---

CRITICAL formatting rules:
- NO EMOJIS anywhere. No 🦅, no 🏆, no decorative symbols. Cameron has been explicit.
- NO eagle imagery (Raptor is a velociraptor, not an eagle).
- Plain text only. Markdown bold and links are fine. No emoji icons.

If you genuinely find nothing substantive this week, output "No significant news this week." Do not invent. Do not pad with generic info."""
    response = client.messages.create(
        model=ANTHROPIC_MODEL_RESEARCH,
        max_tokens=4000,
        tools=[WEB_SEARCH_TOOL],
        messages=[{"role": "user", "content": prompt}],
    )
    return _extract_text(response)


def ai_rising_profile(client, player):
    """Per-rising-player profile. CMO-grade research."""
    prompt = f"""You are Cameron Blakely's Chief Marketing Officer for Raptor Roofing. Treat yourself as a $500K/year CMO doing competitive intelligence. Be insightful, specific, ahead-of-curve. Reference real tactics, name specific platforms and tools, never write generic filler.

Research "{player['name']}", an Indianapolis-area roofing company.

Their numbers: {player['review_count']} total Google reviews at {player.get('rating', 'n/a')} stars, +{player.get('reviews_30d', 0)} in 30 days. {player.get('_rising_reason', '')}

Use web search (5-8 searches). Tell Cameron in 6-8 sentences total:
1. **Who they are.** Ownership, founding year, scale, service area, what makes them distinct.
2. **What's driving the velocity.** Be specific — is it a paid media play? A referral engine? A storm-chasing operation? A particular partnership? A community-event blitz? Point to evidence.
3. **One tactic Raptor should steal.** Name the specific move and how to execute it within 30 days.
4. **One weakness Raptor can exploit in sales conversations.** Be specific (e.g. "their Yelp is 2.5 stars, lean on Raptor's 5.0 in close").

Cite sources inline as markdown links. Do NOT name specific Raptor team members."""
    response = client.messages.create(
        model=ANTHROPIC_MODEL_RESEARCH,
        max_tokens=2500,
        tools=[WEB_SEARCH_TOOL],
        messages=[{"role": "user", "content": prompt}],
    )
    return _extract_text(response)


def ai_competitor_profile(client, biz, raptor):
    """Per-competitor CMO-grade deep research with web search. One Sonnet call per
    competitor that produces an 8-section markdown profile."""
    name = biz["name"]
    total = biz.get("review_count", 0)
    rating = biz.get("rating")
    r7 = biz.get("reviews_7d") or 0
    r30 = biz.get("reviews_30d") or 0
    r90 = biz.get("reviews_90d") or 0
    resp = biz.get("response_rate")
    addr = biz.get("address") or "Indianapolis area"

    raptor_total = raptor["review_count"] if raptor else 663
    raptor_rating = raptor["rating"] if raptor else 5.0

    prompt = f"""You are Cameron Blakely's Chief Marketing Officer at Raptor Roofing — think of yourself as a $500K/year CMO who has worked in residential home-services marketing for 15+ years. You produce competitive intelligence briefs that read like the work of someone who actually knows the playbook (paid media, SEO, AEO, partnerships, sales enablement). No 101-level filler. Be specific, ahead-of-curve, and willing to share contrarian takes when the data supports them.

Refreshed every Tue/Thu on Cameron's dashboard.

Competitor's current numbers:
- Total Google reviews: {total}
- Star rating: {rating}
- Reviews last 7 days: {r7}
- Reviews last 30 days: {r30}
- Reviews last 90 days: {r90}
- Owner response rate: {resp}
- Address: {addr}

Raptor's numbers for comparison:
- {raptor_total} reviews at {raptor_rating} stars
- 5.0 rating is highest in the Indy top 10 by volume
- Premium positioning, "we don't outsource accountability"

Using web search (use it generously, 6-10 searches), produce a markdown brief in EXACTLY this structure. Use bold section headings. Be specific with names, dates, URLs. NO generic filler. NO naming Raptor team members beyond Cameron and Patrick.

## 1. Who They Are
2-3 sentences: founding year, owner(s), service area, what makes them distinct.

## 2. Estimated Scale
Employee count estimate (with **confidence: high/medium/low** label) and rough annual revenue estimate (with confidence). Cite source if available (LinkedIn employee count, ZoomInfo, BuildZoom permits, etc).

## 3. Recent Hiring Signals
What roles they're posting on Indeed, LinkedIn, ZipRecruiter in the last 30 days. Especially flag marketing, sales, ops leadership hires. If nothing recent, say so.

## 4. Off-Google Presence
How they show up on Yelp, Reddit, Facebook (Ad Library too), BBB, Nextdoor. Note discrepancies (e.g., 4.9 on Google but 2.5 on Yelp = obvious review-funneling).

## 5. SEO and AEO Posture
What's their domain authority signal, are they ranking for key Indy roofing queries, do they have schema markup, blog content, backlinks from local press? Anything Raptor could copy or counter.

## 6. Generative Engine Visibility
Search ChatGPT/Claude/Gemini results for "best roofers Indianapolis", "top roofing contractor near Greenwood Indiana", "roof replacement Carmel Indiana", "roofing company Fishers Noblesville Zionsville." Are they cited by name in AI answers? If yes, where and why. If no, that's an opportunity to note.

## 7. Strategic Read vs Raptor
Where they beat Raptor. Where Raptor beats them. One sentence on the most exploitable weakness.

## 8. What Raptor Should Watch or Copy
2-3 specific tactics Cameron should monitor or steal from this competitor.

Aim for 400-650 words total. Cite sources inline as markdown links."""

    response = client.messages.create(
        model=ANTHROPIC_MODEL_RESEARCH,
        max_tokens=4000,
        tools=[WEB_SEARCH_TOOL],
        messages=[{"role": "user", "content": prompt}],
    )
    return _extract_text(response)


def ai_seo_aeo_brief(client, businesses, raptor):
    """One CMO-grade brief covering: (1) competitor SEO/AEO benchmarks vs Raptor,
    (2) one specific actionable tactic Raptor should execute this week, (3) Raptor's
    AEO standing across the metro queries we send to ChatGPT/Claude/Gemini.
    Output is a markdown document with three section headers."""
    top10 = sorted(businesses, key=lambda b: b["review_count"], reverse=True)[:10]
    top_names = ", ".join(b["name"] for b in top10)
    raptor_name = raptor["name"] if raptor else "Raptor Roofing"
    prompt = f"""You are Cameron Blakely's CMO at Raptor Roofing. He pays you $500K/year for ahead-of-the-curve thinking. This is the SEO and AEO (AI Engine Optimization) brief for THIS WEEK's dashboard refresh.

Use web search aggressively (10-15 searches). Your job is three sections, one markdown document.

## 1. Competitor SEO/AEO Benchmarks

Compare {raptor_name} against the top Indy roofing competitors: {top_names}. Look at:
- Domain authority signals (Ahrefs/Moz/Majestic data if surfaceable)
- Backlink profiles — who has local press citations, association memberships, partnership backlinks
- Schema markup on their websites (LocalBusiness, RoofingContractor, FAQ, Review schema)
- Blog cadence and content depth
- Google Business Profile post frequency, photo upload velocity, service area pages
- Programmatic city pages vs Raptor's

For each top-3 competitor, write 2-3 specific findings with the gap to Raptor. Lead with the BIGGEST gap.

## 2. Tactic of the Week

ONE specific SEO/AEO move Raptor should execute this week. Be tactical, not strategic. Examples of the kind of thing I want — these are good prototypes for the format, find a NEW one each week:
- "Post 8 job-site photos to your GBP today. Use a phone with location services on while shooting so EXIF metadata embeds the lat/lon. Google reads that metadata and uses it as a service-area signal. Bone Dry posts 12-15/week. Raptor posted 2 last week. Closing this gap is a 30-day project; start today."
- "Submit a guest column to Indianapolis Business Journal on roofing-cost trends in 2026. IBJ runs 4-6 industry guest pieces per month. The byline backlink is dofollow and IBJ has a domain authority of 67. One placement = 6 months of consistent local-pack improvement."

Write 200-400 words. Include the WHY (what mechanism does this leverage), the HOW (concrete steps), and the SUCCESS METRIC (what to measure in 30 days).

## 3. Raptor's AEO Standing

Run web searches asking ChatGPT-style queries: "best roofers Indianapolis," "top roofing contractor near Greenwood Indiana," "roof replacement Carmel Indiana," "roofing company Fishers Indiana," "best rated roofer Noblesville," "Zionsville roof repair." Look at what Google AI Overviews / Perplexity / Bing AI return for these and similar prompts.

Report:
- Which competitors get cited by NAME in AI search answers? (List them with the queries that surfaced them.)
- Is Raptor cited? Where, and where not?
- What is Raptor missing structurally that gets others cited (review count threshold? Wikipedia entry? specific publication mention? schema markup?)
- ONE specific move to improve Raptor's AEO standing in the next 14 days.

Output the full document in markdown. Source links inline as markdown links. Be direct and specific.

CRITICAL: Do NOT name specific Raptor team members. NO EMOJIS. NO eagle imagery (Raptor is a velociraptor, a dinosaur, not an eagle). Plain text and standard markdown only."""
    response = client.messages.create(
        model=ANTHROPIC_MODEL_RESEARCH,
        max_tokens=5000,
        tools=[WEB_SEARCH_TOOL],
        messages=[{"role": "user", "content": prompt}],
    )
    return _extract_text(response)


def ai_overview_summary(client, market_news, rising_profiles, raptor_actions,
                         seo_aeo_brief, businesses):
    """One-paragraph-per-tab summary so the Overview is the 5-minute version of
    the whole dashboard. Output is markdown with four sections."""
    top5_volume = sorted(businesses, key=lambda b: b["review_count"], reverse=True)[:5]
    leaders_text = ", ".join(f"{b['name']} ({b['review_count']})" for b in top5_volume)
    raptor = next((b for b in businesses if "raptor" in b["name"].lower()), None)
    raptor_text = (
        f"+{raptor['reviews_7d'] or 0} this week, +{raptor['reviews_30d'] or 0} in 30d"
        if raptor else "n/a"
    )
    news_excerpt = (market_news or "")[:1500]
    rising_excerpt = "\n".join(
        f"- {rp['competitor']['name']}: {rp['profile'][:300]}" for rp in rising_profiles[:3]
    ) if rising_profiles else "(none this cycle)"
    seo_excerpt = (seo_aeo_brief or "")[:1500]
    actions_excerpt = (raptor_actions or "")[:1500]
    prompt = f"""You are Cameron's $500K CMO. Cameron has 5 minutes and wants a paragraph each summary of every dashboard tab so he can decide where to dive deeper. Be punchy. Each section is ONE tight paragraph (3-5 sentences). No filler. NO EMOJIS. NO eagle imagery (Raptor is a velociraptor).

Top 5 by volume: {leaders_text}.
Raptor pace: {raptor_text}.

Source material from this cycle (excerpted):

[NEWS]
{news_excerpt}

[RISING]
{rising_excerpt}

[SEO/AEO]
{seo_excerpt}

[ACTIONS]
{actions_excerpt}

Output exactly this markdown (use these exact headings, no other prose):

## In the news this week
[1 paragraph — the most important 1-2 stories, what they mean for Raptor]

## Rising player to watch
[1 paragraph — name the most interesting rising player and why]

## SEO and AEO read
[1 paragraph — the single most important takeaway: what's the gap, what's the move]

## Competitive landscape
[1 paragraph — top 3 named competitors with key context (volume, velocity, vulnerability), where Raptor sits, what's changing]

Plain professional text only. NO EMOJIS."""
    response = client.messages.create(
        model=ANTHROPIC_MODEL_SYNTHESIS,
        max_tokens=2500,
        messages=[{"role": "user", "content": prompt}],
    )
    return _extract_text(response)


def ai_red_team(client, top_competitor, raptor):
    """Red-team analysis. AI plays the named competitor's CMO and writes how they
    would attack Raptor's market position. Surfaces blind spots."""
    name = top_competitor["name"]
    raptor_total = raptor["review_count"] if raptor else 663
    raptor_rating = raptor["rating"] if raptor else 5.0
    prompt = f"""You are about to step into a different role. You are now the Chief Marketing Officer of "{name}", an Indianapolis roofing competitor. You report to {name}'s ownership and your single goal is to take market share from Raptor Roofing in the Indy metro over the next 6 months.

Raptor's known position: {raptor_total} Google reviews at {raptor_rating} stars, top-10 by volume, premium positioning, "we don't outsource accountability" messaging, family-feel community sponsorships in Greenwood and Center Grove, primary lead sources are Google LSA and Google Ads.

Your position as {name}: {top_competitor['review_count']} reviews at {top_competitor.get('rating', 'n/a')} stars.

Using web search (5-8 searches), write a 250-400 word strategic memo to your own ownership covering:
1. Where Raptor is most vulnerable — 3 specific weaknesses you can exploit. Be ruthless and specific.
2. Three campaigns YOU ({name}) would launch in the next 90 days to take share from Raptor.
3. The single most threatening move you could make against them this quarter.

Be specific. Name exact tactics, channels, ad creative angles. The goal: Cameron should read this and feel uncomfortable, then act.

NO EMOJIS. NO eagle imagery. Plain professional memo. Bold the most important sentences."""
    response = client.messages.create(
        model=ANTHROPIC_MODEL_RESEARCH,
        max_tokens=2500,
        tools=[WEB_SEARCH_TOOL],
        messages=[{"role": "user", "content": prompt}],
    )
    return _extract_text(response)


def ai_review_themes(client, biz):
    """Extract praise + complaint themes from a competitor's recent review samples.
    Returns a dict {praise: [{theme, weight}, ...], complaints: [{theme, weight}, ...]}.
    Used for the keyword cloud on the competitor sub-page."""
    samples = biz.get("review_samples") or []
    if not samples or len(samples) < 3:
        return None
    joined = "\n\n".join(f"- {s}" for s in samples[:15])
    prompt = f"""You are extracting customer-language themes from real Google reviews of "{biz['name']}", an Indianapolis roofing competitor. Cameron uses these themes to find sales-call wedges and copy ideas.

Recent reviews (verbatim, customer voice):

{joined}

Output ONLY valid JSON in this exact shape (no other prose, no markdown fences):

{{
  "praise": [
    {{"theme": "short 1-3 word phrase customers actually used", "weight": 1-10}},
    ...up to 8 items
  ],
  "complaints": [
    {{"theme": "short 1-3 word phrase customers actually used", "weight": 1-10}},
    ...up to 5 items
  ]
}}

Rules:
- Themes should be short (1-3 words), in customer voice (e.g. "fast cleanup", "insurance help", "missed appointments")
- Weight 10 = appears in many reviews; weight 1 = appears once
- If reviews are uniformly positive, complaints array can be empty []
- NO emojis, NO eagle imagery, JSON only"""
    try:
        response = client.messages.create(
            model=ANTHROPIC_MODEL_RESEARCH,
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}],
        )
        text = _extract_text(response)
        # Strip code fences if present
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
        return json.loads(text)
    except Exception as e:
        print(f"  themes error: {e}", file=sys.stderr)
        return None


def _slug(name):
    """URL-safe slug from a competitor name."""
    s = re.sub(r"[^a-z0-9]+", "-", name.lower())
    return s.strip("-") or "competitor"


def render_competitor_pages(businesses, raptor, market_news, run_date_str):
    """Render docs/competitor/<slug>/index.html for top 30 by volume."""
    env = Environment(loader=FileSystemLoader(str(DASHBOARD_TEMPLATE_DIR)))
    env.filters["md"] = _md_to_html
    template = env.get_template("competitor.html.j2")
    by_volume = sorted(businesses, key=lambda b: b["review_count"], reverse=True)
    pages_dir = DOCS_DIR / "competitor"
    pages_dir.mkdir(parents=True, exist_ok=True)
    # Normalize to avoid template errors on competitors that didn't get timeline-scraped.
    for biz in by_volume:
        biz.setdefault("reviews_7d", None)
        biz.setdefault("reviews_30d", None)
        biz.setdefault("reviews_90d", None)
        biz.setdefault("response_rate", None)
        biz.setdefault("address", None)
    raptor_norm = raptor or {}
    if raptor_norm:
        raptor_norm.setdefault("reviews_7d", None)
        raptor_norm.setdefault("reviews_30d", None)
        raptor_norm.setdefault("reviews_90d", None)
        raptor_norm.setdefault("response_rate", None)
        raptor_norm.setdefault("rating", None)

    for i, biz in enumerate(by_volume[:50], 1):
        slug = _slug(biz["name"])
        biz["_slug"] = slug
        page_dir = pages_dir / slug
        page_dir.mkdir(parents=True, exist_ok=True)
        profile = biz.get("_deep_profile")
        themes = biz.get("_review_themes")
        html = template.render(biz=biz, raptor=raptor_norm, rank=i, profile=profile,
                                themes=themes, run_date=run_date_str)
        (page_dir / "index.html").write_text(html)


def ai_raptor_actions(client, data_summary):
    """Synthesize 3 specific actions for Raptor based on the week's data."""
    prompt = f"""You are Cameron Blakely's Chief Marketing Officer at Raptor Roofing. Treat yourself as a $500K/year CMO who's seen everything in residential services marketing — direct response, brand, SEO, AEO, partnerships, the works. Cameron owns the business. Context:
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
- Is concrete and measurable this week, not a vague suggestion
- 2-3 sentences

CRITICAL rules:
- NEVER name specific Raptor team members (no "Patrick", "Dylan", "Curtis", "Cameron", "Mike", "Lauryn", etc.). Cameron assigns internally.
- NO EMOJIS. No decorative symbols. No eagle imagery (Raptor is a velociraptor — a dinosaur — not an eagle). Plain text and markdown formatting only.
- Avoid clichés ("focus on reviews," "improve customer service") unless data demands it.
- Bias toward NON-OBVIOUS, ahead-of-curve recommendations a top-tier CMO would surface — partnership plays, content gambits, AEO experiments, ad-creative angles, distribution arbitrage. Cameron has been in roofing for years; no 101-level advice.
- Format each action with: **Bold one-sentence headline.** Then 2-3 paragraphs of detail covering the why, the how, and the success metric. Cameron will read long if the idea is good.
- If an action defends against a specific competitor move, say which competitor and what move directly."""
    response = client.messages.create(
        model=ANTHROPIC_MODEL_SYNTHESIS,
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}],
    )
    return _extract_text(response)


def detect_rising_players(competitors, min_velocity_ratio=0.08, min_30d=15,
                           max_total=700, exclude_recent=None, fallback_rotation=0):
    """Flag small-base high-velocity accounts worth researching.

    Primary rule: gained min_30d+ reviews and ratio of 30d gain to total >= min_velocity_ratio.
    Fallback (if no riser meets the bar): rotate through alternative criteria so the
    section is never empty:
      0 -> lowest response rate in top 15 (vulnerability target)
      1 -> highest absolute weekly gain regardless of base size
      2 -> smallest player still gaining 1+ this week
      3 -> highest 90d gain from a player not in top 5 by volume
    """
    exclude = {n.lower() for n in (exclude_recent or [])}
    primary = []
    for c in competitors:
        if c["name"].lower() in exclude:
            continue
        r30 = c.get("reviews_30d") or 0
        total = c.get("review_count") or 0
        if total == 0 or total > max_total:
            continue
        if r30 < min_30d:
            continue
        ratio = r30 / total
        if ratio >= min_velocity_ratio:
            c["_velocity_ratio"] = ratio
            c["_rising_reason"] = f"+{r30} reviews in 30 days on a base of {total} ({ratio*100:.0f}% velocity)"
            primary.append(c)
    primary.sort(key=lambda c: c["_velocity_ratio"], reverse=True)
    if primary:
        return primary

    # Fallback rotation
    candidates = [c for c in competitors if c["name"].lower() not in exclude
                   and c.get("review_count", 0) >= 50]
    by_volume = sorted(candidates, key=lambda c: c["review_count"], reverse=True)
    if fallback_rotation == 0:
        # Lowest response rate in top 15 — vulnerability target
        pool = [c for c in by_volume[:15] if c.get("response_rate") is not None]
        pool.sort(key=lambda c: c["response_rate"])
        for c in pool[:2]:
            c["_rising_reason"] = f"low {int(c['response_rate']*100)}% response rate, vulnerability worth probing"
        return pool[:2]
    if fallback_rotation == 1:
        # Highest absolute weekly gain
        pool = sorted([c for c in candidates if (c.get("reviews_7d") or 0) > 0],
                       key=lambda c: c.get("reviews_7d") or 0, reverse=True)
        for c in pool[:2]:
            c["_rising_reason"] = f"+{c['reviews_7d']} reviews this week, market leader on momentum"
        return pool[:2]
    if fallback_rotation == 2:
        # Smallest player still gaining 1+ this week
        pool = sorted([c for c in candidates if (c.get("reviews_7d") or 0) >= 1],
                       key=lambda c: c["review_count"])
        for c in pool[:2]:
            c["_rising_reason"] = f"smaller shop ({c['review_count']} total) gaining traction"
        return pool[:2]
    # rotation == 3
    top5_pids = {c.get("place_id") for c in by_volume[:5]}
    pool = sorted([c for c in candidates
                    if c.get("place_id") not in top5_pids and (c.get("reviews_90d") or 0) > 0],
                   key=lambda c: c.get("reviews_90d") or 0, reverse=True)
    for c in pool[:2]:
        c["_rising_reason"] = f"+{c['reviews_90d']} reviews in 90 days outside the top 5 by volume"
    return pool[:2]


def parse_news_blocks(text):
    """Convert AI-generated market news markdown into structured cards. Each card
    exposes a CATEGORY field (award, press, expansion, hire, ad, regulation, etc.)
    that the template uses to pick an SVG icon. No emojis."""
    if not text or "no significant news" in text.lower():
        return []
    blocks = re.split(r"\n\s*---+\s*\n", text)
    cards = []
    multi_field_re = re.compile(
        r"\*?\*?(COMPANY|CATEGORY|EVENT|WHEN|WHY IT MATTERS|DETAIL|SOURCE)\*?\*?:\s*(.+)",
        re.IGNORECASE,
    )
    for block in blocks:
        card = {}
        current_key = None
        for line in block.strip().split("\n"):
            m = multi_field_re.match(line.strip())
            if m:
                key = m.group(1).lower().replace(" ", "_").replace("it_", "")
                card[key] = m.group(2).strip()
                current_key = key
            elif current_key and line.strip():
                # Continuation line for multi-line fields like DETAIL.
                card[current_key] = (card[current_key] + " " + line.strip()).strip()
        if card.get("company") and card.get("event"):
            cat = (card.get("category") or "").strip().lower()
            if not cat:
                event_text = (card.get("event") or "").lower() + " " + (card.get("why_matters") or "").lower()
                if any(w in event_text for w in ["award", "best of", "named", "ranked"]):
                    cat = "award"
                elif any(w in event_text for w in ["press release", "announced"]):
                    cat = "press"
                elif any(w in event_text for w in ["expand", "open", "new office", "location"]):
                    cat = "expansion"
                elif any(w in event_text for w in ["hired", "joined", "appoint"]):
                    cat = "hire"
                elif any(w in event_text for w in ["launch", "campaign", "rebrand"]):
                    cat = "ad"
                elif any(w in event_text for w in ["regulation", "rule", "law", "fannie", "freddie", "irs", "tariff"]):
                    cat = "regulation"
                elif any(w in event_text for w in ["storm", "weather", "hail", "wind"]):
                    cat = "weather"
                elif any(w in event_text for w in ["price", "supply", "shortage"]):
                    cat = "supply"
                else:
                    cat = "other"
            card["category"] = cat
            # Strip emojis from any field that AI may have leaked them into.
            for k in ("event", "why_matters", "detail", "company"):
                if card.get(k):
                    card[k] = _strip_emojis(card[k])
            cards.append(card)
    return cards


def merge_identical_places(businesses):
    """Catch duplicates that escaped place_id dedup (different display names but
    same physical place). If two entries have identical review_count AND rating
    AND non-zero matching reviews_30d, merge them into the entry with the longer
    name (which usually has richer label text)."""
    out = []
    used = set()
    for i, a in enumerate(businesses):
        if i in used:
            continue
        kept = a
        for j in range(i + 1, len(businesses)):
            if j in used:
                continue
            b = businesses[j]
            same_count = a.get("review_count") and a["review_count"] == b.get("review_count")
            same_rating = a.get("rating") == b.get("rating")
            same_30d = (a.get("reviews_30d") or 0) == (b.get("reviews_30d") or 0)
            same_7d = (a.get("reviews_7d") or 0) == (b.get("reviews_7d") or 0)
            if same_count and same_rating and same_30d and same_7d:
                used.add(j)
                # Keep the entry with the SHORTER, cleaner name (typically the pinned one).
                if len(b["name"]) < len(kept["name"]):
                    kept = b
        out.append(kept)
    return out


def _norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower().replace("&", "and"))


# Companies that show up in roofing-adjacent queries but are NOT primarily roofers.
# Bath/kitchen/door/gutter/window specialists who get caught by exterior remodeling
# searches. Keep names normalized (lowercase, alphanumeric only).
NON_ROOFER_BLOCKLIST = {
    "westshorehome",       # Doors, baths, kitchens, no roofing
    "renewalbyandersen",   # Windows only
    "championwindows",     # Windows + sunrooms
    "champion",            # Champion (when it's the windows brand)
    "leaffilter",          # Gutter guards only
    "leafguard",           # Gutter guards only
    "longhomeproducts",    # Bath + kitchen
    "ecoshield",           # Pest control
    "uniquehomesolutions", # Bath + doors per Cameron - excluded pending review
    "powerhomeremodeling", # Windows + siding, not roofing-led
}


def _is_blocked_non_roofer(name):
    n = _norm(name)
    return any(blocked in n for blocked in NON_ROOFER_BLOCKLIST)


def extract_action_headlines(actions_md):
    """Pull just the bold numbered headlines from a Claude-generated actions block."""
    if not actions_md:
        return []
    headlines = re.findall(r"\*\*\s*(\d+\.\s+[^*]+?)\*\*", actions_md)
    return [h.strip().rstrip(".") for h in headlines[:5]]


def parse_actions_into_cards(actions_md):
    """Split a Claude-generated actions block into structured cards. Each card has
    {number, headline, body_md}. Used so the Moves tab can render visual cards
    with icon + headline + collapsible body instead of one wall of prose."""
    if not actions_md:
        return []
    # Split on numbered bold headlines: **N. Headline.**
    parts = re.split(r"\*\*\s*(\d+)\.\s+([^*]+?)\*\*", actions_md)
    # parts looks like: [intro, num1, headline1, body1, num2, headline2, body2, ...]
    cards = []
    for i in range(1, len(parts) - 2, 3):
        try:
            num = int(parts[i])
            headline = parts[i + 1].strip().rstrip(".")
            body = parts[i + 2].strip()
            # Trim trailing markdown headings or separators
            body = re.sub(r"^---+\s*$", "", body, flags=re.MULTILINE).strip()
            cards.append({"number": num, "headline": headline, "body": body})
        except (ValueError, IndexError):
            continue
    return cards


def parse_markdown_sections(md_text):
    """Split markdown text into sections by ## or # headings. Returns a list of
    {heading, body_md} for each section. Used so SEO/AEO and Rising can render
    section-by-section visual cards instead of one long prose block."""
    if not md_text:
        return []
    # Match ## or # headings (level 1-2 only — sub-headings stay in body)
    parts = re.split(r"^(#{1,2})\s+(.+?)$", md_text, flags=re.MULTILINE)
    sections = []
    # parts: [pre, level, heading, body, level, heading, body, ...]
    for i in range(1, len(parts) - 2, 3):
        level = parts[i]
        heading = parts[i + 1].strip()
        body = parts[i + 2].strip()
        # Drop any trailing horizontal rule artifacts
        body = re.sub(r"\n+---+\s*$", "", body).strip()
        sections.append({"level": len(level), "heading": heading, "body": body})
    if not sections and md_text.strip():
        # No markdown headings found — return whole thing as one section
        sections.append({"level": 0, "heading": "", "body": md_text.strip()})
    return sections


def parse_rising_profile(profile_md):
    """Split a rising-player profile into structured fields. The prompt asks for
    four sections (Who/Driving/Steal/Weakness). Parse into a dict so the template
    can render visual cards instead of one prose blob."""
    if not profile_md:
        return None
    fields = {"who": "", "driving": "", "steal": "", "weakness": "", "raw": profile_md}
    # Look for the 4-section structure with bold or markdown sub-headings.
    patterns = [
        ("who", r"(?:^|\n)\*?\*?\s*(?:1\.\s*|##\s*)?\*?\*?\s*Who (?:they|they are|theyre)\*?\*?[:.]?\s*(.+?)(?=\n\s*\*?\*?\s*(?:2\.|##|\*\*[2-9])|\Z)"),
        ("driving", r"(?:^|\n)\*?\*?\s*(?:2\.\s*|##\s*)?\*?\*?\s*What.+?driving.+?velocity\*?\*?[:.]?\s*(.+?)(?=\n\s*\*?\*?\s*(?:3\.|##|\*\*[3-9])|\Z)"),
        ("steal", r"(?:^|\n)\*?\*?\s*(?:3\.\s*|##\s*)?\*?\*?\s*One tactic.+?steal\*?\*?[:.]?\s*(.+?)(?=\n\s*\*?\*?\s*(?:4\.|##|\*\*[4-9])|\Z)"),
        ("weakness", r"(?:^|\n)\*?\*?\s*(?:4\.\s*|##\s*)?\*?\*?\s*One weakness.+?exploit\*?\*?[:.]?\s*(.+?)(?=\n\s*---+|\Z)"),
    ]
    for key, pattern in patterns:
        m = re.search(pattern, profile_md, re.IGNORECASE | re.DOTALL)
        if m:
            fields[key] = m.group(1).strip()
    return fields


_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001F9FF"   # symbols & pictographs (emojis)
    "\U0001FA00-\U0001FAFF"
    "\U00002600-\U000027BF"   # misc symbols + dingbats (eagle, sparkles, arrows)
    "\U0001F000-\U0001F02F"
    "\U0001F100-\U0001F1FF"
    "]+",
    flags=re.UNICODE,
)


def _strip_emojis(text):
    """Remove emojis and decorative pictographs from AI output. Cameron has been
    explicit that the dashboard should not use emojis (no 🦅, 🏆, 📰, etc.)."""
    if not text:
        return text
    return _EMOJI_RE.sub("", text)


def _strip_ai_preamble(text):
    """Strip Claude's narration ('Now I have enough data...', 'I'll search...') that
    sometimes leads AI outputs, plus any decorative emojis. Trim until the first
    real content line."""
    if not text:
        return text
    text = _strip_emojis(text)
    lines = text.split("\n")
    drop_patterns = (
        r"^\s*(now i (have|will|can)|let me (now|produce|compile)|"
        r"based on (my|the) (research|searches?)|here is|here's the (brief|report|analysis|fully)|"
        r"i'?ll (now|produce|compile|search|find|look)|"
        r"i (have|will) (compile|now have|search)|"
        r"(let|allow) me (search|look|gather|do))"
    )
    while lines and re.match(drop_patterns, lines[0], re.IGNORECASE):
        lines.pop(0)
    while lines and (not lines[0].strip() or lines[0].strip() == "---"):
        lines.pop(0)
    return "\n".join(lines).strip()


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

    # Filter out generic Google category listings that are not real businesses.
    # These show up as "Roofing Indianapolis", "Commercial Roofing Indianapolis", etc.
    GENERIC_PATTERN = re.compile(
        r"^(roofing|commercial roofing|residential roofing|roof repair|"
        r"roof replacement|metal roofing|flat roofing)\s+indianapolis$",
        re.IGNORECASE,
    )
    out = []
    for b in merged.values():
        if GENERIC_PATTERN.match(b["name"].strip()):
            continue
        if _is_blocked_non_roofer(b["name"]):
            continue
        out.append(b)
    return out


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
            'button[jsaction*="reviewchart"]',
            'a[href*="/reviews"]',
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
            # Fallback: don't bail. Some place pages render reviews inline without a tab,
            # and some show them after we scroll the main panel. We'll proceed with the
            # extraction and rely on the data-review-id selectors to find anything visible.
            print(f"    reviews tab click failed for {href[:60]} - trying inline scroll", file=sys.stderr)
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
        # Scroll the reviews feed AND the main panel to load reviews. Some place pages
        # don't have the dedicated reviews container — scroll everything.
        scroll_target_js = """
            const el = document.querySelector('div[role="main"] div.m6QErb.DxyBCb, div.m6QErb.DxyBCb, div[role="main"] div.m6QErb');
            if (el) el.scrollBy(0, 1500);
            const main = document.querySelector('div[role="main"]');
            if (main) main.scrollBy(0, 1200);
            window.scrollBy(0, 800);
        """
        for _ in range(22):
            page.evaluate(scroll_target_js)
            page.wait_for_timeout(450)
        # Extract review cards. Capture date, owner-response flag, and review text
        # (truncated). Review text feeds the keyword-cloud / theme-mining pipeline.
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
                // Review body text element classes Google uses
                const textEl = c.querySelector('.wiI7pd, .MyEned, [class*="wiI7pd"]');
                let reviewText = textEl ? textEl.innerText.trim() : '';
                // Strip any "(Translated by Google)" / response section
                const respIdx = c.innerText.indexOf('Response from the owner');
                const fullText = c.innerText || '';
                const hasResponse = /Response from the owner/i.test(fullText);
                if (reviewText.length > 280) reviewText = reviewText.slice(0, 280) + '...';
                if (dateText) out.push({date: dateText, hasResponse, text: reviewText});
            }
            return out;
        }""")
        if not cards_data:
            return timeline
        timeline["reviews_loaded"] = len(cards_data)
        timeline["review_samples"] = []
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
            # Save first 15 review text snippets for theme mining (only ones with text).
            txt = (c.get("text") or "").strip()
            if txt and len(timeline["review_samples"]) < 15:
                timeline["review_samples"].append(txt)
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


def _md_to_html(text):
    """Convert markdown text to HTML for use in email body. Wraps in Markup so
    Jinja does not auto-escape the HTML."""
    if not text:
        return ""
    html = md.markdown(text, extensions=["extra", "nl2br", "sane_lists"])
    return Markup(html)


def _build_chart_data(businesses, raptor):
    """Construct JSON-serializable chart datasets for the dashboard."""
    raptor_name = raptor["name"] if raptor else None

    # Gainers: top 12 by 7d gain (include Raptor even if not in top 12).
    by_gain = sorted(
        [b for b in businesses if (b.get("reviews_7d") or 0) > 0],
        key=lambda b: b.get("reviews_7d") or 0, reverse=True
    )[:12]
    if raptor and raptor not in by_gain and raptor.get("reviews_7d"):
        by_gain.append(raptor)
    gainers = [
        {"name": b["name"], "gain_7d": b.get("reviews_7d") or 0,
         "is_raptor": b["name"] == raptor_name}
        for b in by_gain
    ]

    # Scatter: every competitor with reviews_7d data plotted as total vs 7d gain.
    scatter = [
        {"name": b["name"], "total": b["review_count"],
         "gain_7d": b.get("reviews_7d") or 0,
         "is_raptor": b["name"] == raptor_name}
        for b in businesses if b.get("reviews_7d") is not None and b["review_count"] > 0
    ]

    # Response rate: top 12 by total volume that have a response_rate measured.
    by_volume_with_resp = sorted(
        [b for b in businesses if b.get("response_rate") is not None],
        key=lambda b: b["review_count"], reverse=True
    )[:12]
    response = [
        {"name": b["name"], "rate": int((b["response_rate"] or 0) * 100),
         "is_raptor": b["name"] == raptor_name}
        for b in by_volume_with_resp
    ]

    # Share donut: who got what slice of THIS WEEK's reviews across tracked competitors.
    total_week_gains = sum((b.get("reviews_7d") or 0) for b in businesses)
    share = []
    if total_week_gains > 0:
        sorted_share = sorted(
            [b for b in businesses if (b.get("reviews_7d") or 0) > 0],
            key=lambda b: b.get("reviews_7d") or 0, reverse=True
        )
        # Top 7 by share, rest grouped as "Others".
        top_share = sorted_share[:7]
        rest_share = sum((b.get("reviews_7d") or 0) for b in sorted_share[7:])
        for b in top_share:
            share.append({
                "name": b["name"], "gain": b.get("reviews_7d") or 0,
                "is_raptor": b["name"] == raptor_name,
            })
        if rest_share > 0:
            share.append({"name": "Others", "gain": rest_share, "is_raptor": False})

    return {"gainers": gainers, "scatter": scatter, "response": response, "share": share}


def render_dashboard(snapshot, prev_snapshot, raptor, businesses, market_news,
                      rising_profiles, raptor_actions, seo_aeo_brief=None,
                      overview_summary=None, red_team=None):
    """Render the static dashboard HTML to docs/index.html."""
    env = Environment(loader=FileSystemLoader(str(DASHBOARD_TEMPLATE_DIR)))
    env.filters["md"] = _md_to_html
    template = env.get_template("index.html.j2")

    # Normalize: ensure timeline fields exist for every business so template doesn't crash
    # on competitors that didn't get timeline-scraped this run.
    for b in businesses:
        b.setdefault("reviews_7d", None)
        b.setdefault("reviews_30d", None)
        b.setdefault("reviews_90d", None)
        b.setdefault("response_rate", None)
        b.setdefault("address", None)
        b["slug"] = _slug(b["name"])

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
    chart_data = _build_chart_data(businesses, raptor)
    news_cards = parse_news_blocks(market_news)
    by_volume_leaderboard = [b for b in by_volume if (b.get("review_count") or 0) >= 50][:50]
    by_gain_for_table = [b for b in by_gain_7d if (b.get("review_count") or 0) >= 50][:50]
    action_headlines = extract_action_headlines(raptor_actions)
    action_cards = parse_actions_into_cards(raptor_actions)
    seo_sections = parse_markdown_sections(seo_aeo_brief or "")
    # Annotate each rising profile with structured fields for card layout.
    rising_with_fields = []
    for rp in rising_profiles or []:
        rp_struct = dict(rp)
        rp_struct["fields"] = parse_rising_profile(rp.get("profile", ""))
        rising_with_fields.append(rp_struct)

    # Compute hero-line numbers for the dynamic position+threat headline.
    raptor_count = raptor["review_count"] if raptor else 0
    next_above = None
    next_above_gap = None
    for b in by_volume:
        if b["review_count"] > raptor_count and "raptor" not in b["name"].lower():
            next_above = b
            next_above_gap = b["review_count"] - raptor_count
    # Threat snippet from market news (top news event)
    top_threat = news_cards[0] if news_cards else None

    html = template.render(
        snapshot=snapshot,
        is_baseline=prev_snapshot is None,
        raptor=raptor,
        raptor_rank=raptor_rank,
        next_above=next_above,
        next_above_gap=next_above_gap,
        top_threat=top_threat,
        by_volume=by_volume_leaderboard,
        by_gain_7d=by_gain_for_table,
        market_news=market_news,
        news_cards=news_cards,
        rising_profiles=rising_with_fields,
        raptor_actions=raptor_actions,
        action_headlines=action_headlines,
        action_cards=action_cards,
        seo_aeo_brief=seo_aeo_brief,
        seo_sections=seo_sections,
        overview_summary=overview_summary,
        red_team=red_team,
        chart_data=chart_data,
        run_date=datetime.now(timezone.utc).strftime("%B %d, %Y"),
    )
    DOCS_DIR.mkdir(exist_ok=True)
    (DOCS_DIR / "index.html").write_text(html)
    # Also drop a no-build marker so GH Pages does not try to run Jekyll.
    (DOCS_DIR / ".nojekyll").write_text("")
    return html


def _truncate_actions_preview(raptor_actions, max_chars=600):
    """Pull a short preview from the actions text for the email notification.
    Strips markdown markers and grabs the first ~max_chars."""
    if not raptor_actions:
        return ""
    text = re.sub(r"#+\s*", "", raptor_actions)
    text = re.sub(r"\*\*", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(". ", 1)[0] + "..."
    return text


def render_email(snapshot, prev_snapshot, raptor, businesses, market_news,
                  rising_profiles, raptor_actions, new_entrants):
    """Render the notification email — link to the dashboard, plus a preview
    of Raptor's Moves and a couple of headline stats. The full content lives
    on the dashboard."""
    env = Environment(loader=FileSystemLoader(str(ROOT)))
    env.filters["md"] = _md_to_html
    template = env.get_template("email_template.html")

    by_gain_7d = sorted(
        [b for b in businesses if b.get("reviews_7d") is not None],
        key=lambda b: (b.get("reviews_7d") or 0), reverse=True
    )
    by_volume = sorted(businesses, key=lambda b: b["review_count"], reverse=True)
    raptor_rank = next(
        (i + 1 for i, b in enumerate(by_volume) if "raptor" in b["name"].lower()),
        None,
    )
    actions_preview = _truncate_actions_preview(raptor_actions, max_chars=900)
    top_gainers = by_gain_7d[:5]

    return template.render(
        is_baseline=prev_snapshot is None,
        raptor=raptor,
        raptor_rank=raptor_rank,
        actions_preview=actions_preview,
        top_gainers=top_gainers,
        dashboard_url=DASHBOARD_PUBLIC_URL,
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

    # Multi-query discovery: cast a wider net across categories and Indy-metro cities
    # so we catch competitors whose primary GBP category isn't "Roofing" and accounts
    # in higher-income suburbs (Carmel, Noblesville, Zionsville, Fishers, Greenwood).
    discovery_queries = watch.get("discovery_queries", [
        watch.get("query", "roofing Indianapolis"),
        "exterior remodeling Indianapolis",
        "roof replacement Indianapolis",
        "roofing Carmel Indiana",
        "roofing Fishers Indiana",
        "roofing Noblesville Indiana",
        "roofing Zionsville Indiana",
        "roofing Greenwood Indiana",
    ])
    all_results = []
    for q in discovery_queries:
        print(f"Scraping '{q}'...")
        try:
            chunk = scrape_google_maps(q, top_n=watch.get("discover_top_n", 25))
            print(f"  +{len(chunk)} from '{q}'")
            all_results.extend(chunk)
        except Exception as e:
            print(f"  query '{q}' failed: {e}", file=sys.stderr)
    # Re-dedup the combined results by place_id and name.
    deduped = {}
    for b in all_results:
        key = b.get("place_id") or f"name:{_norm(b['name'])}"
        if key not in deduped or b["review_count"] > deduped[key]["review_count"]:
            deduped[key] = b
    businesses = list(deduped.values())
    print(f"merged across queries: {len(businesses)} unique businesses")
    if not businesses:
        print("no businesses scraped; aborting", file=sys.stderr)
        sys.exit(1)

    # Timeline scraping: for top 30 by volume, load reviews tab and parse timestamps.
    businesses.sort(key=lambda b: b["review_count"], reverse=True)
    timeline_n = min(30, len(businesses))
    print(f"Scraping review timelines for top {timeline_n}...")
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
        for i, biz in enumerate(businesses[:timeline_n], 1):
            href = biz.get("href")
            if not href:
                continue
            t = scrape_review_timeline(page, href)
            biz.update(t)
            print(f"  [{i}/{timeline_n}] {biz['name']}: 7d={t['reviews_7d']} 30d={t['reviews_30d']} 90d={t['reviews_90d']} resp={t['response_rate']} loaded={t['reviews_loaded']}")
        browser.close()

    # Post-timeline merge of identical-data duplicates. These slip past place_id
    # dedup when two records have different display names but the scraper landed
    # on the same physical place (common for multi-location chains).
    businesses = merge_identical_places(businesses)
    print(f"after identical-data dedup: {len(businesses)} unique places")

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
        "profiled_this_week": [],  # populated below after rising profiling
    }

    # Intelligence layer: call Claude Opus/Sonnet for market news, rising profiles,
    # actions, SEO+AEO brief, and per-competitor deep profiles.
    market_news = None
    rising_profiles = []
    raptor_actions = None
    seo_aeo_brief = None
    overview_summary = None
    red_team = None
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

            # Read profiled list from the most recent older snapshot (>= 5 days ago)
            # so the digest rotates week-to-week without filtering itself out when
            # multiple test runs happen the same day.
            cutoff = datetime.now(timezone.utc) - timedelta(days=5)
            recent_profiled = []
            for snap in reversed(history.get("snapshots", [])):
                try:
                    ts = datetime.fromisoformat(snap["timestamp"].replace("Z", "+00:00"))
                except Exception:
                    continue
                if ts <= cutoff:
                    recent_profiled = snap.get("profiled_this_week", [])
                    break
            # Rotate fallback criteria across runs based on snapshot count.
            rotation = len(history.get("snapshots", [])) % 4
            risers = detect_rising_players(businesses, exclude_recent=recent_profiled,
                                            fallback_rotation=rotation)
            for r in risers[:5]:  # 3-5 rising players per Cameron's request
                try:
                    print(f"AI: profiling riser {r['name']}...")
                    profile = ai_rising_profile(client, r)
                    rising_profiles.append({"competitor": r, "profile": profile})
                except Exception as e:
                    print(f"  rising profile failed for {r['name']}: {e}", file=sys.stderr)
            # Save the names so next week's run skips them.
            for rp in rising_profiles:
                snapshot["profiled_this_week"].append(rp["competitor"]["name"])

            # SEO/AEO brief
            try:
                print("AI: generating SEO/AEO brief...")
                raptor_for_seo = next((b for b in businesses if "raptor" in b["name"].lower()), None)
                seo_aeo_brief = ai_seo_aeo_brief(client, businesses, raptor_for_seo)
            except Exception as e:
                print(f"  SEO/AEO brief failed: {e}", file=sys.stderr)

            # Red-team analysis (rotate through top competitors over time, one per run).
            try:
                top_non_raptor = [b for b in sorted(businesses, key=lambda b: b["review_count"], reverse=True)
                                   if "raptor" not in b["name"].lower()]
                rotation_idx = len(history.get("snapshots", [])) % min(5, len(top_non_raptor))
                target = top_non_raptor[rotation_idx] if top_non_raptor else None
                if target:
                    print(f"AI: red-team analysis as {target['name']}'s CMO...")
                    raptor_for_rt = next((b for b in businesses if "raptor" in b["name"].lower()), None)
                    red_team = {"competitor": target, "memo": ai_red_team(client, target, raptor_for_rt)}
            except Exception as e:
                print(f"  red-team failed: {e}", file=sys.stderr)

            # Synthesis
            try:
                summary = _compose_ai_context(businesses, market_news, rising_profiles)
                print("AI: synthesizing Raptor actions...")
                raptor_actions = ai_raptor_actions(client, summary)
            except Exception as e:
                print(f"  actions synthesis failed: {e}", file=sys.stderr)

            # Per-competitor deep profiles for top N by volume.
            # First-time: profile everyone in scope. Subsequent runs: re-profile each
            # competitor at most once per 14 days (rolling) to control cost. We track
            # by reading the previous snapshot's profiles_dates map.
            raptor_for_profile = next((b for b in businesses if "raptor" in b["name"].lower()), None)
            top_for_profile = sorted(businesses, key=lambda b: b["review_count"], reverse=True)[:20]
            prev_profile_dates = {}
            for snap in reversed(history.get("snapshots", [])):
                pd = snap.get("profile_dates")
                if pd:
                    prev_profile_dates = pd
                    break
            now_iso = datetime.now(timezone.utc).date().isoformat()
            new_profile_dates = dict(prev_profile_dates)
            stale_cutoff = (datetime.now(timezone.utc) - timedelta(days=14)).date().isoformat()
            for biz in top_for_profile:
                key = biz.get("place_id") or _norm(biz["name"])
                last_done = prev_profile_dates.get(key)
                if last_done and last_done > stale_cutoff:
                    # Reuse the cached profile from the previous snapshot.
                    prev_profile = None
                    for snap in reversed(history.get("snapshots", [])):
                        for pb in snap.get("businesses", []):
                            if (pb.get("place_id") and pb.get("place_id") == biz.get("place_id")) \
                                    or _norm(pb["name"]) == _norm(biz["name"]):
                                prev_profile = pb.get("_deep_profile")
                                break
                        if prev_profile:
                            break
                    if prev_profile:
                        biz["_deep_profile"] = prev_profile
                        continue
                try:
                    print(f"AI: deep profile for {biz['name']}...")
                    biz["_deep_profile"] = ai_competitor_profile(client, biz, raptor_for_profile)
                    new_profile_dates[key] = now_iso
                except Exception as e:
                    print(f"  profile for {biz['name']} failed: {e}", file=sys.stderr)
                # Themes: only if we have review_samples and either no cached themes
                # or themes are stale (refresh on the same 14-day cycle).
                if biz.get("review_samples"):
                    try:
                        print(f"AI: review themes for {biz['name']}...")
                        themes = ai_review_themes(client, biz)
                        if themes:
                            biz["_review_themes"] = themes
                    except Exception as e:
                        print(f"  themes for {biz['name']} failed: {e}", file=sys.stderr)
            snapshot["profile_dates"] = new_profile_dates

            # Overview summary — one paragraph per tab. Done LAST so it can reference
            # all the other AI outputs from this cycle.
            try:
                print("AI: generating overview summary...")
                overview_summary = ai_overview_summary(client, market_news,
                                                        rising_profiles, raptor_actions,
                                                        seo_aeo_brief, businesses)
            except Exception as e:
                print(f"  overview summary failed: {e}", file=sys.stderr)

    raptor = next(
        (b for b in businesses if "raptor" in b["name"].lower()), None
    )
    # Render the static dashboard FIRST. The email points to it.
    print("Rendering dashboard to docs/index.html...")
    render_dashboard(
        snapshot=snapshot,
        prev_snapshot=prev_snapshot,
        raptor=raptor,
        businesses=businesses,
        market_news=market_news,
        rising_profiles=rising_profiles,
        raptor_actions=raptor_actions,
        seo_aeo_brief=seo_aeo_brief,
        overview_summary=overview_summary,
        red_team=red_team,
    )
    print("Rendering per-competitor sub-pages...")
    run_date_str = datetime.now(timezone.utc).strftime("%B %d, %Y")
    render_competitor_pages(businesses, raptor, market_news, run_date_str)
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
