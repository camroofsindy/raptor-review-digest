# Raptor Review Digest

Weekly competitive intelligence digest for Raptor Roofing. Scrapes top Indianapolis roofing competitors from Google Maps, tracks week-over-week review velocity, and emails an HTML digest every Friday at 8am Eastern.

## What's in the digest

1. **Leaderboard**: top 15 Indy roofers by Google Business Profile review count
2. **Velocity rank**: who accelerated fastest this week (rising threats)
3. **New entrants**: companies appearing that were not here last week
4. **Recommendations**: specific actions based on the data

v2 adds: theme mining via Claude, cross-platform audit (Yelp, Facebook, Nextdoor, BBB), and Raptor-specific coaching.

## Local setup

```
cd ~/projects/raptor-review-digest
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
playwright install chromium
SKIP_EMAIL=1 python main.py   # dry run, writes preview.html
python main.py                # real run, sends email
```

## Files

- `main.py`: orchestrator (scrape, diff, render, send)
- `email_template.html`: Jinja2 HTML email template
- `data/watch_list.json`: search query plus pinned competitor list
- `data/history.json`: week-over-week snapshot history (created on first run)
- `.github/workflows/digest.yml`: weekly cron trigger

## Deploying to GitHub Actions

1. Create a new PRIVATE GitHub repo (e.g. `raptor-review-digest`)
2. Push this project:
   ```
   cd ~/projects/raptor-review-digest
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin git@github.com:YOUR_USER/raptor-review-digest.git
   git push -u origin main
   ```
3. In the GitHub repo, go to Settings, Secrets and variables, Actions, New repository secret, and add:
   - `GMAIL_USER` = Cameron@RaptorRoofing.com
   - `GMAIL_APP_PASSWORD` = the 16 character app password
   - `RECIPIENT_EMAIL` = Cameron@RaptorRoofing.com
   - `ANTHROPIC_API_KEY` = (optional for v1, required for v2 theme mining)

4. The workflow fires every Friday at 8am Eastern. To test immediately, go to Actions, Weekly Digest, Run workflow.

## Timing

- First baseline run: today (no WoW data yet)
- First real week-over-week digest: next Friday
