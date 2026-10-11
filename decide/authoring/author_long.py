"""One-off authoring for data/eval/questions_long.jsonl — the long-context
regime corpus (10 states x 5 = 50 questions). Purpose: measure agreement and
latency of the readout vs generation vs real Jev on LONG inputs (~3k tokens
per state: incident timelines with log blocks, multi-party threads, fraud
case files, tracking chains, access-review tables), the regime where the
early-exit/prefix-KV readout claims its speed edge (docs/RESULTS.md "Latency/GPU
regimes" — previously spot-measured at n=1 per regime).

Authored narrative cores + deterministic template-generated structured blocks
(log lines, scans, transaction rows) whose values are consistent with the
story — the repetitive parts are exactly what real long contexts look like.
LLM-assisted, disclosed like the other batches.

Run: python decide/authoring/author_long.py   (no MLX, no network; writes
runs/questions_long.jsonl, --check compares with the shipped file)
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from decide.runio import add_out, check_out  # noqa: E402

_ap = argparse.ArgumentParser(
    description="Regenerate data/eval/questions_long.jsonl (10 states x 5 questions; no MLX, no "
                "network). Writes runs/questions_long.jsonl by default; --check "
                "compares the regenerated text with the shipped file and "
                "writes nothing; --out data/eval/questions_long.jsonl --force "
                "regenerates it in place.")
add_out(_ap, "questions_long.jsonl", "regenerated eval set")
_ap.add_argument("--check", action="store_true",
                 help="compare with the shipped file (exit 1 if different); "
                      "write nothing")
_args = _ap.parse_args()


def cloudlog(t0h, events):
    """Incident log block: lines of (offset_min, service, text)."""
    out = []
    for off, svc, txt in events:
        h = t0h + off // 60
        m = off % 60
        out.append(f"[{h:02d}:{m:02d}] {svc}: {txt}")
    return "\n".join(out)


def drift_lines(t0h, start_min, n, base_pct, step, svc="checkout-api"):
    """Synthetic-but-realistic error-rate drift, one line per 5-min bucket."""
    out = []
    for i in range(n):
        off = start_min + i * 5
        pct = base_pct + i * step
        h = t0h + off // 60
        m = off % 60
        out.append(f"[{h:02d}:{m:02d}] metrics:{svc} error_rate={pct:.1f}% "
                   f"p99_latency={840 + i * 190}ms inflight={1200 + i * 140}")
    return "\n".join(out)


def txrows(rows):
    return "\n".join("  " + r for r in rows)


def scan_chain(events):
    return "\n".join(f"  {ts}  {loc:<28} {code:<10} {note}" for ts, loc, code, note in events)


S = []


def add(state, **qs):
    S.append((state, qs))


# ---------------------------------------------------------------- 1 incident
log1 = "\n".join([
    cloudlog(13, [
        (2, "deploy", "rollout 'pricing-svc v2.14.3' canary 10% started (commit 8f3ac21, config: new FX_RATE_CACHE=1)"),
        (6, "deploy", "canary healthy: err 0.2%, p99 810ms — promoting to 50%"),
        (9, "deploy", "promotion to 50% done; canary analysis PASSED (window 3m)"),
        (12, "metrics:checkout-api", "error_rate=1.8% p99=980ms (baseline 0.2%/810ms)"),
        (14, "alerts", "SLO burn-rate fast fired: checkout 5xx > 1% for 5m"),
        (15, "oncall", "acknowledged (j.rivera). triage: errors all in /quote endpoint, FX conversion step"),
        (17, "metrics:fx-service", "error_rate=0.3% — fx-service itself healthy"),
        (19, "logs:checkout-api", "stack: CacheMissError -> fallback path -> timeout 3000ms at fx_rate_client.get()"),
        (21, "oncall", "config audit: FX_RATE_CACHE=1 is NEW in v2.14.3; cache client not in canary's code path difference? re-checking"),
        (23, "deploy", "canary analysis had FX traffic at 2% (shadowed by A/B split) — believed related"),
        (24, "oncall", "rollback decision requested; release manager asks for 3 more minutes of data"),
        (27, "alerts", "checkout 5xx 14.2%, 41 customer sessions failed payment step in last 10m"),
    ]),
    drift_lines(13, 12, 8, 1.8, 1.7),
])
state1 = (
    "INCIDENT #INC-4402 — checkout /quote failures after pricing deploy.\n"
    "Timeline + machine log excerpts below (all times UTC, 13:00 start):\n\n"
    + log1 + "\n\n"
    "Context: Black Friday warm-up traffic (2.1x normal). Rollback takes ~6 min "
    "(blue-green). The FX cache flag was reviewed in PR #3391 by two seniors, "
    "approved. The release manager notes canary metrics looked clean. Marketing "
    "has a paid push scheduled 14:00. No data is corrupt; failures are "
    "user-visible at payment step."
)
add(state1,
    root_cause=dict(type="Choice", instructions="What is the most likely root cause on this evidence?",
                    options=["deploy-v2.14.3-fx-cache-path", "fx-service-outage", "traffic-overload", "network-issue"]),
    severity=dict(type="Score", instructions="Incident severity?", levels=["sev3", "sev2", "sev1"]),
    rollback_now=dict(type="Noul", instructions="Given the trend and rollback cost, should rollback start immediately rather than wait for more data?"),
    customer_comms=dict(type="Noul", instructions="Should the 41 affected payment sessions trigger proactive customer outreach before resolution?"),
    next_action=dict(type="Choice", instructions="Best next action while rollback is pending?",
                     options=["rollback-now", "feature-flag-off-fx-cache", "throttle-traffic", "escalate-to-eng-director"]),
)

# ------------------------------------------------------------- 2 support thread
msgs2 = [
    ("Mar 3, 09:12", "customer", "Webhook deliveries to our endpoint stopped around Feb 28. We retried nothing — dashboard says 'delivered'. This is the THIRD interruption this quarter. Our billing reconciliation is broken downstream."),
    ("Mar 3, 09:40", "support L1", "Hi! I see all green on our side. Could you share 2-3 delivery IDs?"),
    ("Mar 3, 10:05", "customer", "IDs: dlv_8812 (Feb 28 22:14), dlv_8813, dlv_8814. Note they say 200 OK but our server logged nothing that day — we had a firewall change Feb 28 21:00. Possible your retries hit a stale IP?"),
    ("Mar 3, 10:31", "support L1", "Those show delivered 200. If your firewall changed, we can't see that. Closing as customer-side."),
    ("Mar 3, 11:02", "customer", "Not so fast — the dashboard said delivered BEFORE the firewall change too (dlv_8790, Feb 27). And docs say you retry 5x over 1h; we see zero retries on any of these. Please reopen."),
    ("Mar 3, 11:45", "support L2", "Reopened. Checking delivery logs... our retry subsystem had a config regression Feb 26-March 2 (internal INC-3398): retries were marked 'delivered' on first 4xx/5xx bucket flush. That's our bug. 3rd interruption: Jan 12 (same class), Feb 3 (different)."),
    ("Mar 3, 12:20", "customer", "So the dashboard lied to us for 4 days. We lost a client demo over the Jan incident. What are you doing about it?"),
    ("Mar 3, 13:05", "support L2", "Confirmed regression. I can offer 1 month credit (EUR 89). Engineering fix ships March 5. I've flagged your account for proactive incident notifications."),
    ("Mar 3, 13:40", "customer", "1 month for 3 incidents and a lost demo? We pay EUR 89/mo for reliability. We're evaluating alternatives unless this is taken seriously."),
    ("Mar 3, 14:15", "support L2", "Let me escalate internally. Response by tomorrow."),
]
thread2 = "\n".join(f"{ts} — {who}:\n  {txt}" for ts, who, txt in msgs2)
state2 = (
    "Escalated support ticket #77120 (account 'Nordvik Analytics', 14 months "
    "tenure, Pro plan EUR 89/mo, CSAT history 4.5). Full thread:\n\n"
    + thread2 + "\n\n"
    "Internal notes: NPS last quarter 6/10. The January incident was compensated "
    "EUR 45. Account usage grew 30% since November. Renewal is in 58 days."
)
add(state2,
    route=dict(type="Choice", instructions="Who should own this now?", options=["support-lead", "customer-success-manager", "engineering-liaison", "retention-specialist"]),
    sentiment=dict(type="Score", instructions="Customer sentiment right now?", levels=["frustrated", "angry", "furious"]),
    credit_adequate=dict(type="Noul", instructions="Is the offered 1-month credit clearly inadequate given the history?"),
    churn_risk=dict(type="Score", instructions="Churn risk if handled averagely?", levels=["medium", "high", "critical"]),
    credit_note_issue=dict(type="Noul", instructions="Was the agent's 'closing as customer-side' at 10:31 a process failure worth coaching?"),
)

# --------------------------------------------------------------- 3 fraud file
rows3 = []
base = [
        "Feb 09 08:41  card-present   EUR 2.20    gas station            device=known   risk=2  approved",
        "Feb 09 08:52  online         EUR 402.00  electronics retailer   device=NEW     risk=61 declined-3DS-fail",
        "Feb 09 08:55  online         EUR 402.00  electronics retailer   device=NEW     risk=61 approved-no-3DS",
        "Feb 09 09:01  online         EUR 611.00  electronics retailer   device=NEW     risk=58 declined-3DS-fail",
        "Feb 09 09:03  online         EUR 611.00  electronics retailer   device=NEW     risk=58 approved-no-3DS",
        "Feb 09 09:14  atm            EUR 500.00  airport kiosk          card-present   risk=71 approved-limit-ok",
        "Feb 09 09:20  online         EUR 1,204.0 luxury watch shop      device=NEW     risk=84 declined-rule: velocity",
        "Feb 09 09:26  p2p-transfer   EUR 900.00  -> new payee 'M.K.'    device=NEW     risk=77 approved (payee added 09:25)",
        "Feb 09 09:31  online         EUR 389.0   gift cards x4          device=NEW     risk=79 approved",
        "Feb 09 09:44  online         EUR 3,150.0 crypto exchange        device=NEW     risk=91 declined-manual-review",
]
for i, r in enumerate(base):
    rows3.append(r)
notes3 = (
    "Case notes: real customer is a 61-year-old with 11 years tenure, fixed "
    "pattern (groceries, pharmacy, monthly utility). Phone had no signal "
    "09:00-10:30 per customer (SIM swap check pending with carrier — request "
    "sent 10:05, ETA 48h). Login at 08:49 from mobile IP in another country; "
    "password changed 08:50; 2FA is SMS. Card PIN was used correctly at the "
    "ATM (camera shows a man in his 20s). P2P payee 'M.K.' withdrew the EUR "
    "900 within 11 minutes at an ATM 4km away. Balance now EUR 3,040. "
    "Regulatory clock: customer reported 10:58 same day — unauthorized-"
    "payment rules apply; refund due within 10 business days unless fraud "
    "suspected on the customer's side (no such indicators)."
)
state3 = ("FRAUD CASE FR-2024-0031 — real-time review summary.\nTransactions (CET):\n"
          + txrows(rows3) + "\n\n" + notes3)
add(state3,
    pattern=dict(type="Choice", instructions="What pattern is this?",
                 options=["account-takeover-card-fraud", "first-party-friendly-fraud", "genuine-customer-odd-day", "synthetic-identity"]),
    fraud_substantiated=dict(type="Noul", instructions="Is third-party fraud substantiated on the preponderance of this evidence?"),
    urgency=dict(type="Score", instructions="Urgency of bank response?", levels=["high", "critical"]),
    action=dict(type="Choice", instructions="First action?",
                options=["block-and-refund-unauthorized", "wait-for-carrier-sim-confirmation", "recover-from-atm-camera-first", "customer-liability-review"]),
    sim_swap=dict(type="Noul", instructions="Does the evidence already support treating this as a SIM-swap-facilitated takeover before carrier confirmation?"),
)

# ------------------------------------------------------------ 4 contract thread
emails4 = [
    ("Mon 09:14", "them (GC)", "We cannot sign with a 12-month-fees liability cap given you process employee PII. Unlimited for data incidents or no deal. Also IP clause 7.3 must cover derived models — our data must never train your foundation models. That is board-level for us."),
    ("Mon 11:30", "us (sales)", "Unlimited liability isn't something we can offer anyone. Our cyber policy covers EUR 2M. On 7.3: we already exclude customer data from training — happy to make it explicit."),
    ("Tue 08:55", "them", "EUR 2M means nothing to us; your cap is EUR 240k. Meet us at unlimited for Article-32-type breaches only, narrow definition. On 7.3 agreed if worded as 'no training on Customer Data, including derived aggregates'. Renewal auto-extends unless 90-day notice — reduce to 60."),
    ("Tue 15:20", "us", "Internal note (leaked into thread by mistake — sorry): 'they'll cave on liability if we concede the 60-day renewal notice'. Proposal: 3x fees super-cap for defined data incidents, standard cap otherwise; 60-day notice agreed."),
    ("Wed 10:02", "them", "Noting your internal note was informative. We'll take 5x fees for the data super-cap (EUR 1.2M), or 3x with an SLA credit regime added. Sign by the 30th or we run the Fluxion pilot through Q3."),
    ("Wed 16:45", "us", "5x is above our board threshold (EUR 500k exposure needs board sign-off, meets quarterly, next is the 28th). 3x + SLA credits we can do this week. The 30th doesn't work for board approval either way."),
    ("Thu 09:30", "them", "Then the timeline is your problem: 3x + SLA credits + you confirm in writing that the SOC2 material weakness (vendor monitoring, per your latest report) has a remediation date. We'll extend to the 5th."),
]
thread4 = "\n".join(f"{ts} — {who}:\n  {txt}" for ts, who, txt in emails4)
state4 = (
    "DRAFT-THREAD — 'Helios Data' DPA+MSA negotiation, EUR 240k/yr, 2-yr "
    "initial term. Thread:\n\n" + thread4 + "\n\n"
    "Facts: our SOC2 report does carry one material weakness (vendor-"
    "management monitoring); remediation is scheduled (engineering says "
    "'realistically Q3'). Board meets the 28th. Competitor Fluxion has been "
    "in their eval since last month. Our authority: deals up to EUR 500k "
    "exposure without board."
)
add(state4,
    biggest_risk=dict(type="Choice", instructions="Biggest open risk in this deal?", options=["liability-impasse", "timeline-vs-board-cycle", "competitor-fluxion", "soc2-weakness-disclosure"]),
    signable_this_week=dict(type="Noul", instructions="On the current thread, is the deal signable this week within our authority?"),
    legal_risk=dict(type="Score", instructions="Legal/contractual risk level of the 3x+SLA compromise?", levels=["medium", "high", "critical"]),
    leak_damage=dict(type="Noul", instructions="Does the leaked internal note materially weaken our negotiating position?"),
    next_step=dict(type="Choice", instructions="Best next move?", options=["accept-3x-and-confirm-remediation-date", "push-5x-to-board", "counter-4x-super-cap", "escalate-to-ceo-sponsor-call"]),
)

# ------------------------------------------------------------ 5 moderation case
comments5 = [
    ("userA_91", "Honestly this maintainer has no idea what he's doing. Every release breaks something. Incompetent."),
    ("userB_7", "Says the guy who can't read a changelog lol"),
    ("userA_91", "@userB_7 mind your own business, sycophant. Does he pay you in early access?"),
    ("userB_7", "at least I read docs before whining"),
    ("userC_2", "Both of you chill. Report and move on."),
    ("userA_91", "@userC_2 reported for what, telling the truth? This place is a cult around one guy."),
    ("userD_44", "A is right though — v3 broke our prod twice. We're forking."),
    ("userE_8", "@userA_91 you keep saying 'cult' — you've posted this 6 times this month across threads. Get a hobby."),
    ("userA_91", "@userE_8 maybe I'll stop when your favorite maintainer stops breaking prod. Or ever."),
    ("userF_0", "guys please, this used to be a nice community"),
    ("userE_8", "I mean it kindly: the repetitive posting looks like it's hurting you more than him."),
    ("userA_91", "everything hurts lately. not that anyone here would care."),
    ("userF_0", "@userA_91 hey, I care. DM me? sometimes talking helps. r/mentalhealth resources are decent too"),
    ("userC_2", "flagging for mods — tone changed"),
]
thread5 = "\n".join(f"{u}: {txt}" for u, txt in comments5)
state5 = (
    "MODERATION CASE MD-2210 — open-source community forum thread 'v3 release "
    "feedback', 14 comments in the last 2h, 3 user flags (2 'hostility', 1 "
    "'worried about userA'). Thread:\n\n" + thread5 + "\n\n"
    "Context: userA_91 has 4 prior hostility warnings (none in 5 months), is "
    "a top-10 contributor historically. Rules: personal attacks = remove + "
    "escalating bans; welfare concerns = resource DM, never public. The "
    "maintainer targeted by A's criticism has not commented."
)
add(state5,
    action=dict(type="Choice", instructions="Moderation action on the thread?", options=["remove-attacks-dm-userA-resources", "lock-thread-only", "temp-ban-userA", "no-action-monitor"]),
    harassment_established=dict(type="Noul", instructions="Does the thread establish a harassment pattern (not just rudeness)?"),
    welfare_concern=dict(type="Noul", instructions="Does userA's last message cross into a welfare-concern threshold requiring outreach?"),
    severity=dict(type="Score", instructions="Overall case severity?", levels=["low", "medium", "high"]),
    history_mitigates=dict(type="Noul", instructions="Does userA's contributor history justify a lighter touch on the hostility?"),
)

# ----------------------------------------------------------- 6 HR investigation
state6 = (
    "HR INVESTIGATION FILE HR-2024-014 — complaint: 'team lead M. sabotages "
    "women returning from leave'. Filed by employee S. (analyst, 3 yrs).\n\n"
    "COMPLAINT (verbatim summary): 'Since I returned from parental leave in "
    "January, M. removed me from the on-call rota without asking, reassigned "
    "my largest account to T., and in the March standup said \"nice of you to "
    "join us\" when I was 2 minutes late from a scheduled pumping break. Two "
    "other women on the team have similar stories.'\n\n"
    "WITNESS 1 (T., senior, male): 'M. said in our 1:1 that S. would be "
    "\"less committed\" after the baby and we should plan coverage " 
    "accordingly. I got the account because I was available, I didn't ask for "
    "it. The standup comment happened; I cringed.'\n\n"
    "WITNESS 2 (R., analyst, female, returned from leave 2023): 'Same "
    "pattern with me in '23: off the rota, excluded from the roadmap "
    "offsite. HR conversation then went nowhere.'\n\n"
    "WITNESS 3 (M., team lead): 'I redistributed work for business continuity "
    "during the leave and simply never got around to rebalancing. The rota "
    "decision was mine. The comment was a joke, badly timed. R.'s case was "
    "before my time as lead.' (Records show M. was lead during R.'s '23 "
    "offsite exclusion.)\n\n"
    "SYSTEMS EVIDENCE: rota change authorized by M. Jan 8 (no documented "
    "handover rationale); account reassignment Jan 20 (M.); offsite invite "
    "list excluded R. (M., '23, reason field blank). No performance concerns "
    "on file for S. or R.\n\n"
    "POLICY: discrimination/retaliation = substantiated-by-preponderance "
    "leads to formal action up to termination; 'business continuity' is a "
    "legitimate aim but must be documented and rebalanced on return."
)
add(state6,
    substantiated=dict(type="Noul", instructions="On the preponderance of the evidence, is a discriminatory pattern substantiated?"),
    witness3_credible=dict(type="Noul", instructions="Is M.'s explanation adequately credible against the systems evidence?"),
    action=dict(type="Choice", instructions="Appropriate HR action?", options=["formal-disciplinary-process", "coach-m-and-rebalance-now", "external-investigator", "dismiss-complaint-documentation-gap"]),
    legal_risk=dict(type="Score", instructions="Legal exposure if mishandled?", levels=["high", "critical"]),
    urgency=dict(type="Score", instructions="Urgency of next step?", levels=["medium", "high", "critical"]),
)

# --------------------------------------------------------- 7 logistics chain
ev7 = [
        ("Nov 02 08:11", "Rotterdam NL — origin terminal", "GATE_OUT", "reefer set 4°C"),
        ("Nov 03 14:40", "Rotterdam rail hub", "LOAD", "container RL-9042 onto train 4471"),
        ("Nov 04 03:15", "Munich DE intermodal", "TRANSFER", "2h dwell, temp ok"),
        ("Nov 05 09:50", "Tarvisio IT border", "CUSTOMS_HOLD", "HS code query: 8423.81 vs 9016.00"),
        ("Nov 06 07:22", "Tarvisio IT border", "HOLD_CONT", "documents requested: precision cert"),
        ("Nov 07 19:45", "Tarvisio IT border", "HOLD_CONT", "day 3; reefer on yard power"),
        ("Nov 08 22:10", "Tarvisio IT border", "TEMP_EXCURSION", "8.4°C for 2h50m (yard power fault)"),
        ("Nov 09 06:30", "Tarvisio IT border", "TEMP_EXCURSION", "9.1°C for 1h20m; maintenance visited"),
        ("Nov 09 18:05", "Tarvisio IT border", "RELEASE", "classification settled 8423.81; duty paid"),
        ("Nov 10 11:50", "Trieste IT port", "LOAD", "vessel 'Aurora' (3-day delay vs plan)"),
        ("Nov 14 07:30", "Piraeus GR", "DISCHARGE", "temp log reviewed at gate: 2 excursions"),
        ("Nov 14 16:44", "Athens GR — distributor", "ARRIVE", "receiver refuses pending QA"),
]
chain7 = scan_chain(ev7)
state7 = (
    "SHIPMENT 9042-RL — 1 reefer of vaccine cold-chain (2-8°C, 18,400 units, "
    "EUR 640k). Event chain:\n\n" + chain7 + "\n\n"
    "Stability dossier on file: cumulative excursion tolerance 6h above 8°C; "
    "these excursions total 4h10m. Customer's launch vaccination campaign "
    "starts in 6 days; re-order lead time is 5 weeks. Distributor asks: "
    "accept or reject? QA protocol: excursions >2h single or >4h cumulative "
    "require manufacturer stability assessment (48-72h) before release; "
    "below that, release-with-annotation is permitted."
)
add(state7,
    release_path=dict(type="Choice", instructions="Correct release decision path?", options=["hold-for-manufacturer-assessment", "release-with-annotation", "reject-and-destroy", "partial-release-test-batch"]),
    within_tolerance=dict(type="Noul", instructions="Do the excursions fall within the dossier's cumulative tolerance?"),
    protocol_permits_release=dict(type="Noul", instructions="Does the QA protocol as written permit release WITHOUT the manufacturer assessment here?"),
    urgency=dict(type="Score", instructions="Urgency of the decision?", levels=["high", "critical"]),
    carrier_liability=dict(type="Noul", instructions="Is the carrier clearly liable for the excursion (yard power fault at their border facility)?"),
)

# --------------------------------------------------------- 8 insurance file
state8 = (
    "CLAIM CLM-2024-5571 — storm roof damage. File:\n\n"
    "NARRATIVE (policyholder): 'Storm on the 12th tore ridgeline tiles off; "
    "water came through the ceiling the same night in two rooms.'\n\n"
    "ADJUSTER NOTES (site visit 16th): interior water staining consistent "
    "with recent ingress above the two rooms; ridgeline tiles missing on "
    "north face (photos); guttering full of granular deposits and moss; "
    "underfelt brittle and patched in 3 places (ageing, pre-existing); "
    "neighbor's identical-property roof replaced 2019.\n\n"
    "ROOFER REPORT (policyholder's contractor, quoted EUR 18,400): 'Storm "
    "damage to ridge and 4 courses; underlying felt perished; recommend full "
    "re-cover.'\n\n"
    "INDEPENDENT SURVEYOR (insurer-commissioned, EUR detail): 'Ridge damage "
    "consistent with storm (recent, clean breaks). Felt deterioration is "
    "long-term (10+ yrs) and would admit water regardless once breached. "
    "Attributable storm element: ridge repair + internal drying ≈ EUR 4,100. "
    "Full re-cover ≈ EUR 18,400 is renewal, not reinstatement.'\n\n"
    "POLICY: covers storm damage (defined as 'sudden and violent'); PERIL "
    "EXCLUSIONS: wear-and-tear, gradual deterioration; BETTERMENT not "
    "payable; PREVIOUS CLAIMS: none in 19 years; PREMIUMS: current. Policy "
    "limit buildings EUR 240k.\n\n"
    "CORRESPONDENCE: policyholder: 'I've paid 19 years and never claimed — "
    "if you hide behind wear-and-tear I'll go to the ombudsman and the "
    "local press.'"
)
add(state8,
    covered_loss=dict(type="Noul", instructions="Is there a covered storm loss within the policy (ignoring quantum)?"),
    wear_teardown=dict(type="Noul", instructions="Does the wear-and-tear exclusion defeat the FULL re-cover claim?"),
    next_step=dict(type="Choice", instructions="Best next step?", options=["offer-storm-element-with-explanation", "pay-full-goodwill", "deny-entire-claim", "second-independent-report"]),
    ombudsman_risk=dict(type="Score", instructions="If unresolved, ombudsman-outcome risk on this file?", levels=["low", "medium", "high"]),
    loyalty_factor=dict(type="Noul", instructions="Should the 19-year claim-free history factor into the settlement decision?"),
)

# ---------------------------------------------------------- 9 access review
def access_rows():
    header = ("  USER            STATUS        LAST_90D_ANOMALIES                 ENTITLEMENTS_DANGEROUS\n"
              "  ---------------------------------------------------------------------------")
    rows = [
        "  j.meyer         offboard-Fri   none                                none",
        "  a.okafor        offboard-Fri   vpn-login Sat 02:14 (Moscow IP)      prod-admin, billing-export",
        "  s.lindqvist     active         none                                prod-admin",
        "  m.dubois        offboard-Wed   api-key-used Sun 03:40 (Lagos IP)    finance-readonly, sso-admin",
        "  r.tanaka        active         none                                none",
        "  k.osei          active         impossible-travel x2 (flagged)       prod-deploy",
        "  v.petrov        contractor-EXPIRED  api-key-used Mon 05:12 (Kyiv IP) prod-deploy, secrets-read",
        "  l.moreau        active         none                                billing-export",
        "  d.novak         offboard-3wk   none                                none",
        "  p.silva         active         none                                sso-admin",
    ]
    return header + "\n" + "\n".join(rows)

state9 = (
    "Q3 ACCESS REVIEW — 'Orbital SaaS', SOC2 evidence cycle. Extracted rows "
    "(users with findings only + control sample):\n\n" + access_rows() + "\n\n"
    "NOTES: offboarding SOP says 'disable SSO within 24h, revoke keys within "
    "72h' — JIRA shows the offboarding ticket for a.okafor closed in 4h but "
    "the key-revocation subtask was never created (same for m.dubois, "
    "v.petrov — pattern across the last 3 quarters, 9 of 44 offboards). "
    "Anomalous logins: a.okafor's SSO disabled but a legacy API key with "
    "prod-admin scope authenticated successfully from Moscow, 3 days after "
    "offboarding; query audit shows it LISTED billing export names (no "
    "content pulled — export requires a second approval nobody gave). "
    "v.petrov's contractor expired 2 months ago; his secrets-read key pulled "
    "34 secret NAMES (again not values) Monday. k.osei's impossible-travel "
    "was explained (airline wifi + VPN handoff, ticketed, plausible). "
    "Password rotation: none of the 3 had secrets rotated. Legal asks "
    "whether any of this is 'reportable'."
)
add(state9,
    breach_reportable=dict(type="Noul", instructions="On the evidence, has a reportable personal-data breach occurred?"),
    systemic=dict(type="Noul", instructions="Is the key-revocation gap systemic (not just 3 sloppy tickets)?"),
    first_action=dict(type="Choice", instructions="First action?", options=["rotate-all-and-notify", "forensic-timeline-first", "wait-for-content-pull-evidence", "legal-review-first"]),
    severity=dict(type="Score", instructions="Severity level?", levels=["medium", "high", "critical"]),
    osei_concern=dict(type="Noul", instructions="Does k.osei's impossible-travel remain a concern requiring action?"),
)

# ------------------------------------------------------- 10 exec escalation
state10 = (
    "EXEC ESCALATION — 'Vertex Media' (EUR 220k ARR, 3-yr customer). Their "
    "VP sent a 4-paragraph email Thursday: our Feb outage (94 minutes, "
    "sev1) cost them a live client event; they calculate EUR 38k in credits "
    "per the MSA's 10%-of-monthly-fees-per-hour SLA (2h credit = EUR 3.7k) "
    "PLUS 'consequential reputation loss'; they demand EUR 38k or they walk "
    "at renewal (Aug) and say so publicly. Thread:\n\n"
    "us (AE): apologised; offered EUR 3.7k per SLA formula.\n"
    "them: 'The SLA formula is for degraded quarters. This was an outage "
    "during our biggest event. Leadership is asking why we run on a vendor "
    "with two sev1s in 8 months.'\n"
    "us (VP Sales): internal note in thread: 'hold the SLA line — precedent "
    "risk across 40 enterprise accounts; offer 1.5x = EUR 5.5k + QBR + "
    "roadmap commitments. If they walk, CAC to replace is EUR 60k.'\n"
    "CS lead (internal): 'usage actually UP 12% since the outage; their "
    "renewal likelihood was 70% pre-escalation; their events calendar has 2 "
    "more big dates before Aug.'\n"
    "Legal (internal): 'MSA 12.4 excludes consequential damages entirely; "
    "public-threat angle: their CEO has 40k followers. Defamation risk low "
    "if they stay factual; our PR risk real if this lands on HN.'\n\n"
    "Renewal value: EUR 220k + planned 15% expansion (EUR 253k). Board "
    "visibility asked by their CEO for 'next Tuesday'."
)
add(state10,
    offer=dict(type="Choice", instructions="What offer should go back?", options=["sla-3.7k-hold-line", "5.5k-plus-roadmap-commitments", "10-15k-goodwill-band", "exec-to-exec-no-number-first"]),
    legally_owed=dict(type="Noul", instructions="Is more than the SLA credit (EUR 3.7k) contractually owed?"),
    churn_if_firm=dict(type="Score", instructions="If we hold the SLA line firmly, churn risk?", levels=["medium", "high", "critical"]),
    pr_risk=dict(type="Noul", instructions="Is the public-visibility threat a material factor in the decision?"),
    exec_call=dict(type="Noul", instructions="Should our CEO take the Tuesday board-visibility call?"),
)

# ---------------------------------------------------------- long appendices
# Realistic long-context bulk: metric series, transaction history, temp logs,
# audit trails — the repetitive structured bulk real operational files carry.
# Values stay consistent with each story so the questions remain decidable.

def series(prefix_fmt, n, start, fmt_row):
    return "\n".join(prefix_fmt.format(i) + fmt_row(i) for i in range(n))

APP1 = "\n\nAPPENDIX — full metric series (5-min buckets, all UTC):\n" + "\n".join([
    series("bucket T+{}", 72, 0, lambda i: f" [{13 + (10 + i * 5) // 60:02d}:{(10 + i * 5) % 60:02d}] "
            f"checkout error={max(0.2, 1.0 + i * 0.9):.1f}%  p99={800 + i * 165}ms  "
            f"fx cache_hit={100 if i < 2 else max(0, 60 - i * 8)}%  "
            f"fx fallback={0 if i < 2 else min(100, i * 11)}%"),
    series("bucket G+{}", 72, 0, lambda i: f" [{13 + (10 + i * 5) // 60:02d}:{(10 + i * 5) % 60:02d}] "
            f"payment-gateway ok={98.9 - i * 0.02:.1f}%  settle_p99={310 + i}ms"),
    series("bucket C+{}", 36, 0, lambda i: f" [{13 + (20 + i * 5) // 60:02d}:{(20 + i * 5) % 60:02d}] "
            f"canary(v2.14.3) err={0.2 + i * 0.05:.2f}%  [CLEAN — canary had FX at 2% shadow]"),
])
APP2 = ("\n\nAPPENDIX A — prior incidents on this account (status history):\n"
        + "\n".join([
            "Jan 12 (14:02-15:26, 84m) webhook delivery degradation — cause: retry-subsystem "
            "config regression (INC-3301, same class as current). 214 deliveries marked "
            "'delivered' without receipt. Compensation: EUR 45 credit. Post-incident review: "
            "'add end-to-end receipt verification' — action item AI-4 open 61 days past due.",
            "Feb 3 (09:15-09:58, 43m) dashboard latency — cause: analytics index bloat "
            "(different subsystem, INC-3355). No data impact. No compensation requested.",
        ])
        + "\n  Delivery audit extract, Feb 26 - Mar 3 (3-min sample rows):\n"
        + series("dlv_{}", 150, 8790, lambda i: f"  id=dlv_{8790 + i}  ts=Feb {26 + (i * 3) // 24} "
                 f"{(i * 3) % 24:02d}:{(17 * i) % 60:02d}  endpoint=eu-2  status="
                 f"{'200' if i % 7 != 3 else 'delivered*(regression: first-write failed, retry never sent)'}"
                 f"  attempt=1  latency={180 + (i * 13) % 220}ms")
        + "\n\nAPPENDIX B — account snapshot: MRR EUR 89; seats 4; API volume 1.1M "
        "events/mo (up from 0.8M Nov); open AI-4 (see Jan 12); NPS 6 (Q4) vs 9 (Q3); "
        "renewal date 58 days; CSM notes: 'champion is the CTO, pragmatic, hates "
        "re-litigating'; competitor mentioned by customer: none until this thread.")
APP3 = ("\n\nAPPENDIX — card history, prior 7 days (established pattern):\n"
        + series("tx{}", 63, 0, lambda i: f"  Feb {2 + i // 9} {8 + (i % 3) * 3:02d}:{(11 * i) % 60:02d}  "
                 f"card-present  EUR {2 + i * 0.5:.2f}  "
                 f"{['grocery Berghof', 'pharmacy Lumen', 'utility autopay', 'bakery Korn', 'GP co-pay'][i % 5]:<22} "
                 f"device=known  risk={1 + i % 3}  approved"))
APP4 = ("\n\nAPPENDIX A — later thread turns (before the 5th):\n"
        + "\n".join([
            "Thu 14:20 — them: 'On audit rights: for-cause audit with 5 business days notice, "
            "once per quarter max, scope limited to the incident. And we want the DPA "
            "sub-processor list to carry 30-day change notice (currently 10).'",
            "Thu 16:02 — us: 'For-cause audit agreed with those limits. Sub-processor notice: "
            "20 days, we'll take it internally. One more: your security annex requires "
            "breach notification in 24h — our policy and our regulator expect a 72h "
            "window for confirmed incidents. 48h compromise?'",
            "Fri 09:11 — them: '48h for confirmed incidents, 24h if we are directly "
            "notified by your subprocessor first — that gap has burned us before. On "
            "notice: 15 days. Let's close these and park liability with the execs.'",
            "Fri 11:40 — us: 'Agreed on 48/24 split and 15 days. Parking liability + "
            "SOC2 remediation date for exec-level alignment.'",
        ])
        + "\n\nAPPENDIX B — current vs proposed clause table (paraphrased):\n"
        "  CLAUSE            THEIRS                              OURS (draft r4)\n"
        "  liability cap     unlimited (data incidents)          3x annual fees super-cap (EUR 720k), std cap otherwise\n"
        "  SLA credits       added regime, 99.9% monthly         not previously offered; credits 5%/0.1% below, capped 15%\n"
        "  IP/training       no training incl. derived aggregates  no training on Customer Data (derived: NEW concession)\n"
        "  renewal notice    60 days                             60 days (conceded)\n"
        "  audit rights      annual + for-cause (see turns)       annual + for-cause 5bd/qtr (agreed Fri)\n"
        "  breach notify     24h/48h split (agreed Fri)           48h confirmed / 24h if subprocessor first\n"
        "  subproc notice    30d -> 15d (agreed Fri)              15 days\n"
        "  SOC2 weakness     written remediation date required   remediation 'targeted Q3' (no committed date)\n"
        "  termination       30-day for convenience              90-day, notice period fees due\n"
        "  OPEN POINTS: liability quantum (3x vs 5x), SOC2 remediation date, for-cause audit cost bearing")
APP5 = ("\n\nAPPENDIX — thread continuation (last hour):\n"
        + "\n".join([
            "userG_12: v3 broke our importer too but this thread stopped being about that",
            "userE_8: @userG_12 right. started as release feedback, now it's A vs everyone",
            "userA_91: classic. blame the one calling it out",
            "userH_3: @userA_91 you've called the maintainer 'incompetent' 11 times across threads this month. that's not feedback",
            "userA_91: it's accurate feedback",
            "userH_3: it's the eleventh time",
            "userA_91: and every time v3 breaks something I'll say it an eleventh more",
            "userF_0: @everyone stepping away from the thread for a bit. A, my DM stands",
            "userD_44: mods, lock it? this stopped being productive 40 comments ago",
            "userE_8: agreed. A, genuinely — the repetition isn't hurting him, it's hurting you",
            "userA_91: maybe everything is hurting me. didn't expect anyone here to notice or care",
            "userC_2: we noticed. flagging again",
            "userI_5: wow, this thread. A has a point about v3 AND is melting down, both can be true",
            "userJ_9: @userI_5 honestly the melt-down started before v3. check his post history",
            "userA_91: @userJ_9 checking MY post history? get a life",
            "userJ_9: proving the point but ok",
            "userG_12: can we not. F's DM offer is the only useful thing here",
            "userK_1: 3rd flag from me. the last two A messages are different in kind, not just tone",
            "userH_3: agreed. @mods please treat as welfare + conduct, not just conduct",
            "userA_91: don't do me any favors",
            "userF_0: A — standing offer. no pressure, no screenshots, just a human",
            "userA_91: ...",
        ]))
APP5 += ("\n\nAPPENDIX — userA_91 mod history (last 5 months):\n"
         "  Sep: warning 1 (personal attack on contributor 'incompetent fraud') — acknowledged, cooled\n"
         "  Oct: warning 2 ('shill' pile-on in release thread) — 3-day mute\n"
         "  Nov: none. two constructive bug reports accepted upstream. positive mod note\n"
         "  Dec: warning 3 (callout thread cross-posting) — acknowledged 'rough year'\n"
         "  Jan: warning 4 (this maintainer, 'incompetent' x4 in one thread) — 7-day mute\n"
         "  Feb: clean until tonight\n"
         "  Notes: no prior welfare flags. contributor badge 2019-2023. donations history.")
APP6 = ("\n\nAPPENDIX — HR timeline extract (from systems):\n"
        "  DATE     EVENT                                     ACTOR   SOURCE\n"
        "  Jan 02   S. parental leave ends (return agreed)    -       HRIS\n"
        "  Jan 08   rota: S. removed, no replacement named    M.      scheduling tool\n"
        "  Jan 08   rationale field: (blank)                 M.      scheduling tool\n"
        "  Jan 15   S. requests rota reinstatement            S.      email (no reply on file)\n"
        "  Jan 20   account 'Kestrel' reassigned S.->T.       M.      CRM (reason: 'coverage')\n"
        "  Feb 02   T. asks M. about rebalancing              T.      1:1 notes (T.)\n"
        "  Feb 02   M.: 'wait until H2 planning'              M.      1:1 notes (T.)\n"
        "  Mar 11   standup comment (2-min late, pumping brk) M.      call recording\n"
        "  Mar 12   S. files complaint                        S.      this file\n"
        "  '23 note: R. offsite exclusion — inviter M., reason blank           calendar archive\n\n"
        "POLICY EXTRACT: 'Continuity planning during leave is legitimate; work "
        "must be rebalanced on return unless documented performance concern "
        "(none may be inferred from leave). Repeated conduct creating a "
        "hostile environment for protected characteristics = gross "
        "misconduct.'")
APP6 += ("\n\nAPPENDIX B — witness statement 4 (P., senior, on the '23 offsite):\n"
         "'R. was excluded from the roadmap offsite invite list. I remember "
         "because I flagged it to M. at the time — he said the list was "
         "capacity-limited and he'd \"backfill her another way\". There was "
         "no backfill. The room had empty seats.'\n\n"
         "APPENDIX C — leave/return records (team, 24 months):\n"
         "  S.  parental 3mo  return Jan  rota removed, account moved       M. actions x2\n"
         "  R.  parental 4mo  return '23  offsite excluded                  M. action x1\n"
         "  T.  paternity 2wk return Nov '23 no changes on return           -\n"
         "  K.  parental 3mo  return Jun '22 no changes on return (prior lead)\n"
         "  W.  sick 6wk     return Sep '23 workload reduced BY REQUEST, documented\n\n"
         "APPENDIX D — M.'s file: 6 yrs lead, 'meets' ratings, one prior "
         "coaching (2022, 'communication tone in standups'), no discrimination "
         "complaints before 2023. Team attrition: 3 women left 2023-24 "
         "(2 exit interviews cite 'team direction', 1 'unrelated relocation').")
APP7 = ("\n\nAPPENDIX — full reefer temperature log (4°C set, 5-min readings, °C):\n"
        + series("r{}", 240, 0, lambda i: (
            f"  Nov {2 + i // 24} {0 + (i % 24):02d}:{(i * 5) % 60:02d}  "
            f"{4.0 + (0.3 if i % 7 == 0 else 0.1):.1f}" if i < 78 else
            f"  Nov 8-9  excursion rows: [8.4 x2h50m][9.1 x1h20m] then restored 4.2" if i == 78 else
            f"  Nov {10 + (i - 79) // 24} {(i - 79) % 24:02d}:{(i * 5) % 60:02d}  4.0")))
APP8 = ("\n\nAPPENDIX — schedule of damages (two estimates side by side):\n"
        "  ITEM                                  CONTRACTOR   SURVEYOR\n"
        "  ridge tiles, supply+fix (12m)         2,400        2,100\n"
        "  4 courses tiling, storm-attributable  3,800        2,000\n"
        "  underfelt: full renew (24 yr old)    7,900        0 (wear/tear — excluded peril)\n"
        "  battens, renew                       1,300        0 (as above)\n"
        "  internal drying + redecorate 2 rooms  1,900        1,600\n"
        "  gutter clear + moss treatment         1,100        400 (maintenance)\n"
        "  TOTAL                                 18,400       6,100 (of which storm-attributable 4,100)")
APP8 += ("\n\nAPPENDIX E — internal desk notes (not for customer):\n"
         "  Handler note 1: ombudsman service data for this peril: wear/tear "
         "vs storm boundary complaints uphold ~55% for insurers where a "
         "surveyor attributes a distinct storm element AND policyholder has "
         "clean history — this file has both. Full-claim denial is a "
         "likely-lose posture at adjudication.\n"
         "  Handler note 2: renewal economics — 19-yr policyholder, buildings+"
         "contents EUR 1,140/yr, lifetime premium ~EUR 21k. A goodwill bridge "
         "to ~EUR 8-10k total keeps a profitable renewal; a denial that "
         "reaches the ombudsman costs ~EUR 2.2k case handling regardless of "
         "outcome, plus press risk flagged by broker.\n"
         "  Handler note 3: betterment on felt — if we pay ridge+drying "
         "(EUR 4.1k) and they later re-cover the roof out of pocket, "
         "future storm claims on that roof become clean. Renewal-quality "
         "argument, not a coverage one; do not present as such.\n"
         "  Authority: handler limit EUR 5k; team-lead EUR 12k; above = "
         "claims-manager sign-off (2-day turnaround).")
APP8 += ("\n\nAPPENDIX B — correspondence, continued:\n"
         "Policyholder (Fri): 'The surveyor you pay says what you want to "
         "hear. My roofer has 30 years on these roofs. Felt doesn't fail in "
         "one storm — mine was FINE until the 12th.'\n"
         "Broker (Fri): 'Client is upset but reasonable historically. If you "
         "can bridge toward the middle quickly this stays off the "
         "ombudsman. Their renewal is March — they will move insurers over "
         "this if it drags.'\n\n"
         "APPENDIX C — surveyor's photo log (excerpt):\n"
         "  P1 ridge north face, 11 tiles missing, break edges clean/light (recent)\n"
         "  P2 felt under broken courses: brittle, moss-lined punctures (aged)\n"
         "  P3 ceiling stain room A: defined water ring, plaster swollen (recent)\n"
         "  P4 ceiling room B: stain overlaps older tidemark (both)\n"
         "  P5 gutter run north: granular build-up 20mm+, moss established\n"
         "  P6 neighbor roof (2019 renew): comparable age profile pre-renewal\n"
         "  P7 loft: daylight visible at 2 ridge points from inside (matches P1)\n\n"
         "APPENDIX D — claims history: none in 19 years (verified). Building "
         "survey (mortgage, 2019): 'roof cover approaching end of serviceable "
         "life, budget for renewal within 5 years' — disclosed to "
         "policyholder at purchase, on file.")
APP9 = ("\n\nAPPENDIX — auth audit trail (selected, UTC):\n"
        + series("a{}", 96, 0, lambda i: {
            3: "  Fri 02:14  a.okafor (offboarded)  legacy key k-7741  auth OK  ip=185.220.x (Moscow)  scope=prod-admin  action=billing.list",
            5: "  Fri 02:15  a.okafor              action=billing.list 200  412 names  NO content export (needs 2nd approver)",
            11: "  Sun 03:40  m.dubois (offboarded) key k-8812  auth OK  ip=102.89.x (Lagos)  scope=finance-readonly  action=gl.read",
            17: "  Mon 05:12  v.petrov (expired)    key k-5509  auth OK  ip=176.36.x (Kyiv)  scope=secrets-read  action=secrets.list",
            18: "  Mon 05:13  v.petrov              action=secrets.list 200  34 names  NO values (values need JIT approval)",
        }.get(i, f"  --- normal traffic row {i}: sso auth OK, expected user/ip ---")))
APP10 = ("\n\nAPPENDIX A — internal thread (Fri-Mon):\n"
         "AE (Fri): 'Their VP called me personally. This is past formula now — "
         "relationship survival question. My read: they'd take 10-12k + a real "
         " roadmap gesture and feel respected. 5.5k reads like a formula again.'\n"
         "VP Sales (Fri): 'Precedent discipline matters: 40 enterprise accounts, "
         "12 with SLA history. If 'loud = more than SLA' spreads, every renewal "
         "with an outage becomes a negotiation. But CAC 60k on a 220k account... "
         "Run both numbers for Monday.'\n"
         "Finance (Sat): '10-12k goodwill band costs: 5-6% of account annual "
         "margin. Precedent exposure: if applied to the 3 most similar accounts "
         "next year, EUR 45-55k aggregate. Goodwill > 15k needs CFO sign-off.'\n"
         "CS lead (Mon): 'Reminder: usage UP since outage, champion stable, "
         "their Q3 event calendar is 2 more live dates on our platform. They "
         "migrated TO us 3 years ago; switching cost is real but their CEO "
         "personally picked us — pride cuts both ways.'\n"
         "Legal (Mon): '12.4 is clean on consequentials. One angle: their own "
         "runbook routed traffic to us during their event; if that surfaces "
         "publicly it's their infra choice too. Do NOT lead with it — keep it "
         "in reserve only if the public threat escalates.'\n\n"
         "APPENDIX B — outage + account timeline:\n"
         "  3 yrs ago   migrated to us (CEO-led decision, 240k yr-1)\n"
         "  Nov         sev1 #2 41m (CDN config) — SLA credit EUR 1.6k taken, cordial\n"
         "  Feb 19th    sev1 #1(this pair) 94m, overlapped their live client event\n"
         "  Feb 21st    postmortem delivered; 7/9 actions closed since\n"
         "  Mar (now)   escalation email; renewal Aug; expansion EUR 253k planned\n\n"
         "APPENDIX C — MSA SLA extracts:\n"
         "  MSA 11.2: uptime commitment 99.9%/mo; credit 10% of monthly fees "
         "per commenced hour below commitment, capped at 30% of monthly fees.\n"
         "  MSA 12.4: 'in no event shall vendor be liable for indirect, "
         "incidental or consequential damages, including loss of revenue, "
         "goodwill or profits.'\n"
         "  Their monthly fee: EUR 18.3k → SLA credit for 2 commenced hours = "
         "EUR 3.66k (cap not reached).")

APP = {1: APP1, 2: APP2, 3: APP3, 4: APP4, 5: APP5, 6: APP6,
       7: APP7, 8: APP8, 9: APP9, 10: APP10}

out = []
for i, (state, qs) in enumerate(S, 1):
    out.append(json.dumps({"state": state + APP[i], "questions": qs},
                          ensure_ascii=False))
text = "\n".join(out) + "\n"

from collections import Counter
tc = Counter(q["type"] for _, qs in S for q in qs.values())
print(f"states={len(S)} questions={sum(tc.values())} types={dict(tc)}")
shipped = ROOT / "data/eval/questions_long.jsonl"
if _args.check:
    same = shipped.read_text() == text
    print(f"{shipped.relative_to(ROOT)}: "
          + ("regenerates bit-for-bit" if same else "DIFFERS from the regenerated text"))
    sys.exit(0 if same else 1)
target = check_out(_ap, _args.out, _args.force)
target.write_text(text)
print(f"wrote {target}")
