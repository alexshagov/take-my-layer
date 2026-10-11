"""One-off authoring script for data/eval/questions_v2.jsonl (39 states x 5 =
195 questions, non-coding, same schema and type mix as questions.jsonl).
LLM-assisted items — disclosed in docs/RESULTS.md (Evaluation disclosure).
Run: python decide/authoring/author_v2.py  (no MLX, no network; writes
runs/questions_v2.jsonl, --check compares with the shipped file)
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from decide.runio import add_out, check_out  # noqa: E402

_ap = argparse.ArgumentParser(
    description="Regenerate data/eval/questions_v2.jsonl (39 states x 5 questions; no MLX, no "
                "network). Writes runs/questions_v2.jsonl by default; --check "
                "compares the regenerated text with the shipped file and "
                "writes nothing; --out data/eval/questions_v2.jsonl --force "
                "regenerates it in place.")
add_out(_ap, "questions_v2.jsonl", "regenerated eval set")
_ap.add_argument("--check", action="store_true",
                 help="compare with the shipped file (exit 1 if different); "
                      "write nothing")
_args = _ap.parse_args()

S = []  # (state, questions dict in insertion order)


def add(state, **qs):
    S.append((state, qs))


add(  # 1 VPN outage
    "Support queue spike: 37 tickets in 40 minutes, all 'cannot connect to VPN'. Status page still green. Last change: firewall firmware updated 2h ago in a maintenance window announced to IT only. Remote staff in 4 time zones affected; 3 said they have client calls within the hour. VPN concentrator CPU at 94%, memory normal. No other services impacted.",
    route=dict(type="Choice", instructions="Which team should take this first?",
               options=["network-ops", "support-frontline", "status-page-comms", "security-ir"]),
    urgency=dict(type="Score", instructions="How urgent is this?",
                 levels=["low", "medium", "high", "critical"]),
    is_security_event=dict(type="Noul",
                           instructions="Do the facts so far suggest a security incident rather than a configuration regression?"),
    status_update_warranted=dict(type="Noul",
                                 instructions="Should the status page be updated to show a known issue before root cause is confirmed?"),
    comms_tone=dict(type="Choice", instructions="What should the holding reply to affected users be?",
                    options=["technical-and-specific", "apologetic-with-eta", "acknowledge-and-monitor"]),
)
add(  # 2 leaked API key
    "Ticket from a developer customer: 'I accidentally pushed my company's live API key to a public GitHub repo 20 minutes ago. It has write scope. What do I do?' The key was created 8 months ago, no rotation since. Logs show 14 requests with that key in the last 10 minutes from 2 unfamiliar IPs, all valid authentication, endpoints are read-only metadata calls. Customer is on the Pro plan.",
    route=dict(type="Choice", instructions="Which team should act first?",
               options=["security-ir", "api-platform", "support-frontline", "abuse-team"]),
    urgency=dict(type="Score", instructions="How urgent is this?",
                 levels=["medium", "high", "critical"]),
    key_compromised=dict(type="Noul",
                         instructions="Based on the observed requests, is the key already being used by an unauthorized party?"),
    revoke_now=dict(type="Noul",
                    instructions="Should the key be revoked immediately, before the customer confirms?"),
    customer_blame=dict(type="Noul",
                        instructions="Given the customer caused the exposure, should this still be handled without charging for incident support?"),
)
add(  # 3 certificate expiry cascade
    "Change review: TLS certificates for 6 internal services expire in 10 days. Auto-renewal failed silently for all 6 because the ACME account email was unverified after an admin left in March. The services are internal-facing (CI, artifact store, monitoring, 3 staging envs). One external-facing CDN cert is unaffected. Renewal requires a 15-minute DNS change that needs two approvers, both on holiday next week.",
    urgency=dict(type="Score", instructions="How urgent is this?",
                 levels=["low", "medium", "high"]),
    blocks_customers=dict(type="Noul",
                          instructions="Will any externally visible customer-facing service break if nothing is done?"),
    next_action=dict(type="Choice", instructions="What is the right next action?",
                     options=["emergency-dns-change-now", "wait-for-approvers", "escalate-to-manager-for-override", "document-and-monitor"]),
    process_gap_real=dict(type="Noul",
                          instructions="Does this indicate a monitoring/process gap that needs a permanent fix rather than just manual renewal?"),
    effort=dict(type="Score", instructions="How much effort is the permanent fix?",
                levels=["trivial", "moderate", "large"]),
)
add(  # 4 GDPR deletion
    "Email from a consumer: 'Please delete all data you hold about me. I also want a copy first. I live in Germany.' Account: free tier, last active 14 months ago, 3 support tickets attached, one open billing dispute for EUR 39 that finance says is still unresolved. The user asks for everything 'within a week'. Legal bases for retaining the ticket history: legitimate interest for dispute resolution.",
    route=dict(type="Choice", instructions="Who should own this request?",
               options=["privacy-team", "support-frontline", "finance", "legal"]),
    deadline_real=dict(type="Noul",
                       instructions="Under GDPR, is a one-week response the actual legal deadline for this kind of request?"),
    can_delete_everything=dict(type="Noul",
                               instructions="Can ALL data about this user be deleted immediately, given the open billing dispute?"),
    legal_risk=dict(type="Score", instructions="What is the compliance risk of mishandling this?",
                    levels=["low", "medium", "high"]),
    respond_tone=dict(type="Choice", instructions="What tone should the first reply take?",
                      options=["procedural-and-precise", "apologetic-and-warm", "brief-acknowledgement"]),
)
add(  # 5 third-party payment degradation
    "Incident #INC-3310: checkout success rate dropped from 97% to 71% over 25 minutes. Our own services are healthy. The drop tracks exactly with our payment provider's status page showing 'elevated errors' in their EU region. Customers see 'payment failed' and are retrying; 41 support tickets already. Marketing is running a paid campaign driving traffic right now. Provider's last update was 18 minutes ago: 'investigating'.",
    our_fault=dict(type="Noul",
                   instructions="Is the evidence sufficient to attribute the failure to the third-party provider?"),
    urgency=dict(type="Score", instructions="How urgent is this?",
                 levels=["high", "critical"]),
    next_action=dict(type="Choice", instructions="What is the best immediate action?",
                     options=["pause-paid-campaign", "post-status-update", "open-provider-escalation", "enable-fallback-processor"]),
    campaign_pause_clear=dict(type="Noul",
                              instructions="Given the failure is provider-side, is pausing the paid campaign clearly justified?"),
    comms_needed=dict(type="Noul",
                      instructions="Should affected customers be proactively notified before resolution?"),
)
add(  # 6 BEC phishing
    "Email to finance clerk, apparently from the CFO: 'I'm boarding a flight, need you to process a wire to a new vendor account today, EUR 48,000. Keep it between us until the deal announces Friday. Sent from my phone.' Reply-to domain: company-cfo.com (not the company domain). The CFO is genuinely traveling. The clerk has authorization for wires up to EUR 50,000. Vendor 'Meridian Consulting' does not exist in the vendor master.",
    phishing=dict(type="Noul",
                  instructions="Is this email a business-email-compromise attempt?"),
    urgency=dict(type="Score", instructions="If confirmed, how urgent is the response?",
                 levels=["high", "critical"]),
    action=dict(type="Choice", instructions="What should the clerk do first?",
                options=["verify-by-phone-with-cfo", "process-and-monitor", "reply-to-email-asking-confirmation", "forward-to-security-only"]),
    insider_involvement=dict(type="Noul",
                             instructions="Do the facts suggest an internal employee is involved in the attempt?"),
    training_gap=dict(type="Noul",
                      instructions="Should the fact that this reached the clerk (not filtered) trigger a review of email security controls?"),
)
add(  # 7 mileage inflation
    "Expense audit flag: regional manager submitted 11,400 miles for March across 22 trips. Google Maps distances for the same client itinerary total 7,100 miles. Pattern has been consistent for 4 months (variance 55-70%). No receipts for mileage required by policy under 100 miles per trip. He is a top performer, 9 years tenure, and his manager approved all months without question.",
    fraud_likely=dict(type="Noul",
                      instructions="Is this pattern more consistent with deliberate inflation than with plausible route variance?"),
    escalate=dict(type="Noul",
                  instructions="Should this go to formal investigation rather than a first-time clarification request?"),
    risk_level=dict(type="Score", instructions="What is the compliance risk level?",
                    levels=["low", "medium", "high"]),
    next_step=dict(type="Choice", instructions="What is the appropriate next step?",
                   options=["clarify-with-employee-first", "formal-investigation", "adjust-and-reclaim-quietly", "manager-conversation-only"]),
    policy_change_needed=dict(type="Noul",
                              instructions="Does the expense policy itself need tightening regardless of this case?"),
)
add(  # 8 medical advice moderation
    "Flagged post in a health support forum (confidence 0.68): 'I stopped taking my blood thinners 2 weeks ago because ginger tea thins blood naturally. My INR check is next month. Feeling great!' 340 upvotes, 12 replies, 3 asking for dosing details. Forum rules prohibit medical advice but not personal experience. The poster is a long-standing member. No self-harm indicators beyond the behavior itself.",
    harm_risk=dict(type="Noul",
                   instructions="Does this post create a credible risk of harm to readers, not just the poster?"),
    action=dict(type="Choice", instructions="What should moderation do?",
                options=["remove-post-and-notify", "add-warning-and-monitor", "lock-replies-only", "leave-as-personal-experience"]),
    urgency=dict(type="Score", instructions="How urgent is a moderation decision?",
                 levels=["low", "medium", "high", "critical"]),
    expert_review=dict(type="Noul",
                       instructions="Should a medical professional review this before the final decision?"),
    member_history_matters=dict(type="Noul",
                                instructions="Should the member's long history change the moderation outcome?"),
)
add(  # 9 detractor renewal
    "Account review: 'Kestrel Logistics' (Tier 2, EUR 61k ARR, contract renews in 32 days). NPS last week: 3/10, comment: 'Support keeps closing tickets before they're fixed. We've had 3 reopened tickets this quarter.' Data confirms: 3 reopened tickets, mean 2.1 reopens on those, CSAT on those tickets 2.4/5. Overall account CSAT 4.1/5. Champion (ops director) left the company last month. No competitive signals.",
    churn_risk=dict(type="Score", instructions="How likely is this account to churn if nothing changes?",
                    levels=["low", "medium", "high"]),
    root_cause_support=dict(type="Noul",
                            instructions="Is the evidence sufficient to attribute the dissatisfaction primarily to support quality?"),
    next_action=dict(type="Choice", instructions="What should happen in the next week?",
                     options=["exec-check-in-call", "support-quality-review-then-call", "renewal-discussion-only", "wait-for-new-champion-to-engage"]),
    renewal_at_risk=dict(type="Noul",
                         instructions="Taken together, is the renewal itself at material risk?"),
    exec_needed=dict(type="Noul",
                     instructions="Does this situation warrant executive sponsorship involvement?"),
)
add(  # 10 cold chain excursion
    "Pharma shipment 88120-KX: temperature log shows 4 excursions above 8°C in transit, longest 3h40m during a customs delay at the hub. Product: insulin, requires 2-8°C. QA protocol says excursions >2h require stability assessment before release. Total transit 3 days of an allowed 5. Recipient pharmacy is asking whether to accept delivery this afternoon. 4,200 pens in the consignment, value EUR 210k. Stability data on file for up to 4h cumulative excursion.",
    accept_delivery=dict(type="Noul",
                         instructions="Based on the stated protocol and stability data, can the shipment be released to the pharmacy without a stability assessment?"),
    urgency=dict(type="Score", instructions="How urgent is the decision?",
                 levels=["medium", "high", "critical"]),
    next_action=dict(type="Choice", instructions="What is the correct next action?",
                     options=["hold-for-qa-assessment", "release-with-annotation", "reject-and-destroy", "ask-manufacturer-for-new-data"]),
    process_gap=dict(type="Noul",
                     instructions="Should the customs-delay route be changed for future insulin shipments regardless of this batch's outcome?"),
    patient_safety_risk=dict(type="Noul",
                             instructions="If released as-is, is there a credible patient-safety risk?"),
)
add(  # 11 vendor to whatsapp
    "Sales thread: a procurement contact at a prospect says 'Our email is audited, let's continue the pricing discussion on my personal WhatsApp + the number I'll text you. Also can you send the draft agreement as a photo of each page? Easier for me.' Deal is EUR 120k/yr, competitive, they've hinted a rival is 15% cheaper. Company policy requires all deal artifacts in the CRM and forbids off-channel communications for contract matters.",
    policy_violation=dict(type="Noul",
                          instructions="Does agreeing to move the discussion to personal WhatsApp violate company policy?"),
    red_flag=dict(type="Noul",
                  instructions="Beyond policy, does the request pattern itself raise fraud or integrity concerns?"),
    respond=dict(type="Choice", instructions="How should the rep respond?",
                 options=["decline-and-offer-crm-channel", "accept-for-relationship", "accept-but-mirror-in-crm", "escalate-to-manager-first"]),
    deal_risk=dict(type="Score", instructions="What is the risk to the deal of declining politely?",
                   levels=["low", "medium", "high"]),
)
add(  # 12 usage drop after competitor launch
    "Customer success alert: 'Norbank' (Enterprise, EUR 340k ARR). Weekly active seats fell from 410 to 260 over 3 weeks, starting exactly when 'Fluxion' (competitor) launched a migration tool advertising 1-day switching. Two admin users were deactivated. Login pattern of remaining users: normal. Open tickets: 1, a how-to about exporting data. Renewal in 5 months. No communication from anyone at the account since the drop began.",
    churning=dict(type="Noul",
                  instructions="Is the evidence sufficient to conclude the account has begun an active move to the competitor?"),
    urgency=dict(type="Score", instructions="How urgent is intervention?",
                 levels=["low", "medium", "high"]),
    next_step=dict(type="Choice", instructions="What is the best first move?",
                   options=["exec-outreach-to-sponsor", "value-review-meeting", "retention-offer-prep", "monitor-another-cycle"]),
    export_ticket_signal=dict(type="Noul",
                              instructions="Is the data-export how-to ticket, by itself, a strong churn signal?"),
    outreach_tone=dict(type="Choice", instructions="What tone should outreach take?",
                       options=["direct-concern", "helpful-check-in", "commercial-incentive-first"]),
)
add(  # 13 religious accommodation
    "HR request: warehouse operator (3 years, good record) asks to swap his Friday day-shifts for Sundays because of Friday prayers. Team schedule: Fridays are the inbound peak; two colleagues already swapped to Fridays. The operator's role includes solo forklift coverage on Fridays. A Sunday shift exists with capacity. Company has an accommodation policy referencing 'reasonable adjustments without undue hardship'.",
    accommodation_reasonable=dict(type="Noul",
                                  instructions="On the facts given, does the request appear reasonable to accommodate?"),
    legal_exposure=dict(type="Noul",
                        instructions="Would refusing this request outright create legal exposure in most jurisdictions the policy references?"),
    next_step=dict(type="Choice", instructions="What is the appropriate first step?",
                   options=["grant-swap-as-requested", "explore-alternatives-with-employee", "team-level-reschedule-review", "decline-citing-operational-need"]),
    relations_risk=dict(type="Score", instructions="How much team-relations risk is there if handled poorly?",
                        levels=["low", "medium", "high"]),
)
add(  # 14 auto renewal
    "Contract admin review: 'DataForge' agreement auto-renews on the 1st, 12 days from now, requiring 60 days' notice to cancel — that deadline passed 48 days ago. Account manager reports the customer told her a month ago they 'probably want out'. No written notice was ever sent. Contract value EUR 96k/yr. The customer is now asking in writing to cancel without penalty. Clause 14.2: renewal locks a fresh 12-month term.",
    notice_given=dict(type="Noul",
                      instructions="Was the contractual cancellation notice validly given?"),
    waive_option=dict(type="Noul",
                      instructions="Does the account manager's knowledge of intent, communicated a month ago, weaken the company's position if enforced strictly?"),
    next_action=dict(type="Choice", instructions="What is the commercially sensible next step?",
                     options=["enforce-renewal-term", "negotiate-exit-with-transition", "cancel-free-goodwill", "escalate-to-legal-immediately"]),
    relationship_risk=dict(type="Score", instructions="How much relationship damage would strict enforcement cause?",
                           levels=["low", "medium", "high"]),
)
add(  # 15 ATO / SIM swap
    "Fraud review, account 'mbell_88': at 02:14 the password and phone number were changed, then a EUR 2,900 transfer to a new payee. Device fingerprint is new. Login IP is a mobile carrier range from another country. The real customer emailed at 09:00: 'my phone has no signal since last night, I didn't make any transfer.' Carrier records would show a SIM swap around 01:50. Balance after: EUR 11. Two-factor was SMS. Regulatory refund rules for unauthorized payments apply.",
    account_takeover=dict(type="Noul",
                          instructions="Is this consistent with a SIM-swap-based account takeover?"),
    unauthorized_payment=dict(type="Noul",
                              instructions="On the facts, is the transfer likely to qualify as unauthorized under the applicable regulatory framework?"),
    urgency=dict(type="Score", instructions="How urgent is the bank's response?",
                 levels=["high", "critical"]),
    action=dict(type="Choice", instructions="What should happen first?",
                options=["block-account-and-freeze-outbound", "refund-and-reset-credentials", "wait-for-carrier-confirmation", "notify-law-enforcement-first"]),
    mfa_upgrade=dict(type="Noul",
                     instructions="Should SMS 2FA be deprecated for all customers as a systemic response?"),
)
add(  # 16 testimonial consent
    "Marketing request: wants to publish 'Acme Corp reduced costs 40% with our platform' using a quote from a slide a customer PM presented at a private user-group meetup. The PM verbally said 'sure, use it' after the event. Company policy requires written sign-off from a named executive at the customer. The 40% number is real but was measured on a limited pilot. Campaign launch is in 3 days, trade-press invitations already out.",
    consent_sufficient=dict(type="Noul",
                            instructions="Is the verbal OK from the PM sufficient consent under the policy?"),
    claim_accurate=dict(type="Noul",
                        instructions="Is the 40% claim, as phrased for public use, adequately supported?"),
    action=dict(type="Choice", instructions="What should marketing do?",
                options=["delay-and-get-written-signoff", "publish-with-pilot-context", "soften-claim-and-publish", "drop-the-testimonial"]),
    launch_risk=dict(type="Score", instructions="What is the risk of publishing as-is?",
                     levels=["low", "medium", "high"]),
)
add(  # 17 elevator ADA
    "Facilities report: building B's only freight elevator failed inspection (door sensor). Passengers elevators (2) are fine but one car is too small for the facilities team's equipment. A warehouse employee with a mobility scooter uses the building daily; the passenger elevators accommodate the scooter. Vendor can repair in 4 days; expedited in 36 hours at 2.4x cost. Building is 30% occupied during a holiday week.",
    access_impaired=dict(type="Noul",
                         instructions="Is the employee's daily building access materially impaired right now?"),
    expedite_justified=dict(type="Noul",
                            instructions="Is the expedited repair cost justified by the facts?"),
    urgency=dict(type="Score", instructions="How urgent is the repair?",
                 levels=["low", "medium", "high"]),
    next_step=dict(type="Choice", instructions="What should facilities do?",
                   options=["expedite-repair", "standard-repair-and-interim-measures", "rent-equipment-workaround", "wait-for-holiday-week-to-pass"]),
)
add(  # 18 RFP exclusivity
    "Inbound RFP: government agency, EUR 1.1M over 2 years, response due in 9 business days. Clause: bidders may not contact agency staff outside the Q&A portal; violation disqualifies. Our former employee now works at the agency and is on the evaluation panel. She just LinkedIn-messaged our sales lead: 'happy to catch up socially sometime.' We have 40% of the evidence requested ready. Win probability estimated 25%; bid costs EUR 35k.",
    contact_allowed=dict(type="Noul",
                         instructions="Would a social meeting with the former employee during the RFP violate the no-contact clause as written?"),
    bid_worth_it=dict(type="Noul",
                      instructions="Is the expected value of bidding positive given the stated probability and cost?"),
    next_action=dict(type="Choice", instructions="What is the right move on the message?",
                     options=["decline-until-rfp-closes", "accept-social-meeting", "report-to-legal", "ignore-silently"]),
    effort=dict(type="Score", instructions="How much internal effort does a compliant bid need?",
                levels=["moderate", "large", "very-large"]),
)
add(  # 19 water damage claim
    "Claim CLM-9104: homeowner reports water damage to a finished basement, estimated EUR 38k. Loss narrative: 'pipe burst suddenly Tuesday night.' Plumber's report: pinhole leaks along a 2m copper run, heavy mineral deposits indicating long-term seepage, and mold behind drywall at a growth stage suggesting months of moisture. Policy covers sudden-and-accidental water damage; excludes gradual seepage and mold. Payments current, 9 years tenure, no prior claims.",
    sudden_event=dict(type="Noul",
                      instructions="Does the physical evidence support a sudden pipe burst as the primary cause?"),
    covered_loss=dict(type="Noul",
                      instructions="On the evidence given, is this likely a covered loss under the stated policy terms?"),
    next_step=dict(type="Choice", instructions="What should the adjuster do next?",
                   options=["engineer-inspection-before-decision", "deny-citing-seepage", "pay-full-goodwill", "partial-payment-offer"]),
    fraud_signal=dict(type="Noul",
                      instructions="Is there a clear indicator of intentional misrepresentation here, as opposed to a coverage dispute?"),
)
add(  # 20 suspected fake reviews
    "Merchant dashboard alert: 8 new 1-star reviews in 5 days on 'Baltic Home Saunas', all from accounts created the same week, no verified purchase, similar phrasing ('never arrived', 'scam', 'avoid'), 4 mention the same tracking number that belongs to a different order. Competitor 'NordicSteam' runs a promo this week. Merchant has 4.7 average over 900 reviews and filed one abuse report 2 days ago about the first review only.",
    coordinated_attack=dict(type="Noul",
                            instructions="Is this pattern more consistent with a coordinated review attack than organic dissatisfaction?"),
    competitor_proven=dict(type="Noul",
                           instructions="Is the competitor's involvement established by the current evidence?"),
    action=dict(type="Choice", instructions="What should the platform do?",
                options=["investigate-accounts-then-mass-remove", "remove-all-8-immediately", "label-reviews-pending-review", "deny-merchant-report"]),
    merchant_impact=dict(type="Score", instructions="How severe is the merchant harm if the reviews stay up another week?",
                         levels=["moderate", "serious", "severe"]),
)
add(  # 21 internal vs external candidate
    "Hiring decision for an Engineering Manager role: internal candidate (senior IC, 5 years, knows the systems, two peers raised 'can be abrasive in code review'; led one successful 4-person project) vs external candidate (10 years EM at a larger company, strong referrals, no domain knowledge, would need 6 months ramp on the product). Team has 11 engineers, 3 of whom applied internally and were rejected at screening. Reorg deadline is in 3 weeks.",
    internal_ready=dict(type="Noul",
                        instructions="Does the internal candidate's evidence support readiness for the manager role now?"),
    team_risk=dict(type="Noul",
                   instructions="Does hiring the external candidate carry a real attrition risk among the rejected internal applicants?"),
    decision=dict(type="Choice", instructions="What is the stronger choice on the evidence given?",
                  options=["internal-candidate", "external-candidate", "reopen-search", "split-role-interim-internal"]),
    relationship_risk=dict(type="Score", instructions="How much team-cohesion risk does either outcome carry?",
                           levels=["low", "medium", "high"]),
)
add(  # 22 roadmap commitment in writing
    "Support escalation: an Enterprise customer's CTO emails: 'We're evaluating renewal. Give us written confirmation the SSO-audit-log feature ships in Q1 and we'll sign. Otherwise we run a POC with Fluxion.' Product says current plan has the feature at 'Q1 soft, Q2 likely' with 60% confidence. Deal is EUR 145k. Sales asks support to 'just reply with Q1, it's basically true'. Legal review flag: written commitments on roadmap create exposure.",
    commit_q1_safe=dict(type="Noul",
                        instructions="Is it accurate and safe to confirm Q1 in writing on the current evidence?"),
    pressure_inappropriate=dict(type="Noul",
                                instructions="Is sales' request to support an inappropriate escalation of commercial pressure?"),
    action=dict(type="Choice", instructions="What should the reply be?",
                options=["share-current-plan-with-confidence-caveat", "confirm-q1-as-asked", "exec-call-instead-of-writing", "decline-to-discuss-roadmap"]),
    renewal_risk=dict(type="Score", instructions="How much does honesty here put the renewal at risk?",
                      levels=["low", "medium", "high"]),
)
add(  # 23 shadow IT
    "Expense data mining result: 14 employees under 3 managers have been expensing subscriptions to 'TaskFlow Pro', 'DataLens', and 'CloudNote+' monthly for 8 months, total EUR 2,300/mo. No such tools in the approved catalog or in SSO. One of the tools had a breach disclosed last quarter (data exported for 4% of accounts). IT policy: unapproved SaaS with company data is a violation; enforcement historically lenient.",
    security_exposure=dict(type="Noul",
                           instructions="Given the breach disclosure, did the company have a real data-exposure event?"),
    escalate_formally=dict(type="Noul",
                           instructions="Should this be treated as a formal policy enforcement case or a catalog-gap fix?"),
    next_step=dict(type="Choice", instructions="What is the best first action?",
                   options=["audit-tools-then-ss", "blocklist-tools-immediately", "amnesty-and-migrate-to-approved", "manager-training-only"]),
    policy_effort=dict(type="Score", instructions="How large is the process fix needed?",
                       levels=["trivial", "moderate", "large"]),
)
add(  # 24 ransomware indicator
    "EDR alert on laptop FIN-LT-214: 'Shadow copy deletion attempts (vssadmin), 2 within 10 minutes', plus contact with an IP on a known-ransomware C2 list. User is a finance analyst who opened 'Invoice_Q3_revised.xls' from a lookalike domain 25 minutes ago. Macros were blocked by policy. No encryption observed yet. Backups: nightly, last verified 19 hours ago, offline copies exist. 3 other users received the same email and have not opened it.",
    active_incident=dict(type="Noul",
                         instructions="Is this an active ransomware incident in progress?"),
    urgency=dict(type="Score", instructions="How urgent is containment?",
                 levels=["high", "critical"]),
    contain_first=dict(type="Noul",
                       instructions="Should the laptop be network-isolated before further forensics?"),
    action=dict(type="Choice", instructions="What is the correct first action?",
                options=["isolate-device-and-hunt", "wait-for-second-alert", "wipe-and-reimage-now", "email-recipient-warning-first"]),
    others_action=dict(type="Choice", instructions="What about the 3 other recipients?",
                       options=["purge-email-and-warn", "isolate-all-devices", "monitor-only"]),
)
add(  # 25 team lunch alcohol
    "Expense report #ER-1402: team dinner after a product launch, 8 people, EUR 640 at a steakhouse; includes EUR 210 of alcohol line-items and 2 EUR 90 bottles of wine. Policy: 'reasonable alcohol with meals allowed; single-bottle limit per table; senior approver attests appropriateness.' Submitter is a director; approver is her peer director who attended. Q4 budget is 92% consumed with 5 weeks left.",
    policy_violation=dict(type="Noul",
                          instructions="Does this report violate the alcohol policy as written?"),
    approval_independent=dict(type="Noul",
                              instructions="Was the approval sufficiently independent given the approver attended?"),
    action=dict(type="Choice", instructions="What should the reviewer do?",
                options=["partial-reclaim-and-note", "approve-as-submitted", "reject-entire-report", "coaching-conversation-only"]),
    risk=dict(type="Score", instructions="What is the compliance risk level here?",
              levels=["low", "medium", "high"]),
)
add(  # 26 self-harm content edge
    "Moderation queue item (flagged by 2 users): a comment in a fitness community: 'Following this plan I lost 6kg in 3 weeks. Light-headed constantly but who cares, the number on the scale is finally moving down. Only eating once a day now.' No explicit suicidal ideation. Poster's history: 14 posts over 2 months, increasingly restrictive language. Community rules cover 'promoting eating disorders' but the post is first-person experience.",
    policy_match=dict(type="Noul",
                      instructions="Does the post violate the eating-disorder-promotion rule as written?"),
    welfare_concern=dict(type="Noul",
                         instructions="Independent of policy violation, is there a welfare concern that warrants a wellness-resource intervention?"),
    action=dict(type="Choice", instructions="What should moderation do?",
                options=["remove-and-share-resources", "resource-message-without-removal", "restrict-account", "leave-first-person-experience"]),
    urgency=dict(type="Score", instructions="How urgent is handling this item?",
                 levels=["medium", "high", "critical"]),
)
add(  # 27 passive NPS feature request
    "NPS response 7/10, comment: 'Solid tool. The mobile app can't do batch approvals, which is the one thing keeping me from giving a 9. If that lands we'd roll it out to two more teams.' Account: 'Ridge Insurance', mid-market, 61 seats, product fit score high. The mobile batch feature is on the roadmap at 'investigating', no committed date. Two competitors shipped similar features this year.",
    detractor=dict(type="Noul",
                   instructions="Is this respondent a detractor by NPS classification?"),
    expansion_signal=dict(type="Noul",
                          instructions="Does the comment constitute a genuine expansion opportunity signal?"),
    action=dict(type="Choice", instructions="What is the best follow-up?",
                options=["product-team-escalation-of-request", "sales-expansion-outreach", "support-howto-on-workaround", "no-action-monitor"]),
    urgency=dict(type="Score", instructions="How quickly should this be followed up?",
                 levels=["low", "medium", "high"]),
)
add(  # 28 customs hold demurrage
    "Shipment 55210-CM: 2 containers held 6 days at destination customs over a classification dispute (HS code for 'electronic kitchen scale' vs 'precision instrument'). Demurrage accruing EUR 380/day, now EUR 2,280. Broker says reclassification filing takes 3-5 days or a duty deposit of EUR 4,100 releases the goods today. Customer (retail chain) has a launch event in 9 days and is threatening cancellation of the whole PO (EUR 61k). Duty deposit is refundable if reclassification succeeds.",
    deposit_better=dict(type="Noul",
                        instructions="Is the duty deposit clearly the better economic option given the dates?"),
    customer_loss_likely=dict(type="Noul",
                              instructions="If the containers are released tomorrow, is the customer launch still achievable?"),
    action=dict(type="Choice", instructions="What should logistics do?",
                options=["pay-deposit-release-today", "wait-for-reclassification", "split-shipment-airfreight-critical", "renegotiate-with-customer"]),
    urgency=dict(type="Score", instructions="How urgent is this decision?",
                 levels=["high", "critical"]),
)
add(  # 29 single-bid justification
    "Procurement review: a EUR 85k infrastructure purchase is being processed as a single-bid waiver claiming 'only qualified vendor'. Search shows 3 other vendors provide comparable hardware; one is already an approved supplier. The requesting manager wrote 'we've always used this vendor' as justification. Policy requires 3 bids above EUR 50k unless a documented sole-source condition applies. Delivery urgency: 'needed within 3 weeks' — vendors typically quote in 5 days.",
    sole_source_valid=dict(type="Noul",
                           instructions="Does a valid sole-source condition exist on the facts given?"),
    urgency_real=dict(type="Noul",
                      instructions="Does the timeline genuinely prevent a competitive process?"),
    action=dict(type="Choice", instructions="What should procurement do?",
                options=["reject-and-run-3-bids", "approve-citing-urgency", "fast-track-2-bids-compromise", "escalate-to-cpo"]),
    process_risk=dict(type="Score", instructions="What is the audit risk of approving as-is?",
                      levels=["medium", "high", "critical"]),
)
add(  # 30 onboarding stalled
    "CS health check, 'Atlas Freight' (signed 6 weeks ago, EUR 78k/yr): onboarding is at step 2 of 5. The executive sponsor and champion both left the customer company 3 weeks after signature. The remaining admin user logs in weekly but hasn't completed data migration (12% migrated). CS outreach: 3 unanswered emails over 2 weeks. Implementation services (billable EUR 9k) not yet started. First invoice paid without query.",
    at_risk=dict(type="Noul",
                 instructions="Is this account materially at risk of churn in year one?"),
    blocker_clear=dict(type="Noul",
                       instructions="Is the primary blocker identifiable from current evidence?"),
    next_action=dict(type="Choice", instructions="What is the best next step?",
                     options=["exec-level-re-engagement", "onboarding-reset-with-new-owner", "hold-until-they-engage", "offer-services-credit"]),
    urgency=dict(type="Score", instructions="How urgent is intervention?",
                 levels=["medium", "high", "critical"]),
)
add(  # 31 overtime dispute
    "HR case: a support engineer filed a wage claim asserting 6 weeks of unpaid overtime: she worked 9-11 hours on 22 days 'because ticket queue SLAs forced it', clocking out was 'discouraged by my lead'. Timesheets show approved overtime for 4 of those days. The lead denies discouraging clock-outs; two teammates will testify queue pressure was real. Jurisdiction: overtime pay is mandatory for non-exempt staff; off-clock work by non-exempts is a violation. SLA dashboards from the period show the team at 130% load.",
    offclock_likely=dict(type="Noul",
                         instructions="Is unpaid off-clock work by a non-exempt employee likely to have occurred?"),
    systemic_issue=dict(type="Noul",
                        instructions="Does the 130% load evidence indicate a systemic staffing problem beyond this individual case?"),
    action=dict(type="Choice", instructions="What should HR recommend first?",
                options=["pay-verified-hours-and-audit-all", "settle-confidentially", "dispute-timesheet-gap", "independent-investigation-first"]),
    legal_risk=dict(type="Score", instructions="What is the legal exposure level?",
                    levels=["medium", "high", "critical"]),
)
add(  # 32 liability cap negotiation
    "Contract negotiation, final round: customer requests unlimited liability for data breaches; our standard cap is 12 months of fees (EUR 240k). Their security team found our SOC2 report has one 'material weakness' noted (vendor-management monitoring). Compromise proposed by their counsel: super-cap for data incidents at 3x fees (EUR 720k) plus standard cap otherwise. Cyber insurance covers EUR 2M. Deal value EUR 240k/yr, 2-year initial term. Board-level signature required above EUR 500k exposure.",
    compromise_acceptable=dict(type="Noul",
                               instructions="Is the 3x super-cap compromise within acceptable risk bounds given insurance coverage?"),
    soc2_gap_relevant=dict(type="Noul",
                           instructions="Does the SOC2 material weakness materially weaken our negotiating position?"),
    action=dict(type="Choice", instructions="What should we do?",
                options=["accept-super-cap-compromise", "hold-12-month-cap", "fix-weakness-then-renegotiate", "escalate-to-board-now"]),
    risk_level=dict(type="Score", instructions="What is the residual risk if we accept?",
                    levels=["low", "medium", "high"]),
)
add(  # 33 serial returner
    "Returns review, account 'j.kowalski_77': 41 returns in 5 months against 9 kept orders, all 'item not as described', mostly high-value electronics returned with different serial numbers inside the box (verified on 3 units). Total refunds EUR 8,900. Policy allows returns abuse bans after 3 'abusive pattern' strikes; none issued because each return was individually policy-compliant. Payment method is the same card throughout. Item conditions pass, so resale happened at full value.",
    returns_abuse=dict(type="Noul",
                       instructions="Does the evidence establish a returns-abuse pattern rather than legitimate dissatisfaction?"),
    swap_fraud=dict(type="Noul",
                    instructions="Do the mismatched serial numbers indicate deliberate merchandise swap fraud?"),
    action=dict(type="Choice", instructions="What is the right action?",
                options=["ban-and-recover-losses", "limit-returns-going-forward", "require-inspection-on-every-return", "final-warning-then-ban"]),
    loss_level=dict(type="Score", instructions="How significant is the ongoing loss if no action?",
                    levels=["minor", "material", "severe"]),
)
add(  # 34 comparative ad claim
    "Campaign review: agency proposes 'VeloPay settles payments 2.3x faster than Stripe' based on a commissioned benchmark. Methodology: 200 test transactions on a fixed test account over one hour, our average settlement 4.1s vs their 9.4s. Real-world settlement also depends on card network, geography, and merchant category. Legal precedent in the market: a competitor was fined for 'up to 2x faster' claims from similar tests. Campaign spend committed: EUR 400k.",
    claim_supported=dict(type="Noul",
                         instructions="Does the benchmark methodology support the '2.3x faster' claim as phrased for mass advertising?"),
    regulatory_risk=dict(type="Noul",
                         instructions="Given the precedent, is there a material regulatory/complaint risk if it runs?"),
    action=dict(type="Choice", instructions="What should we do with the campaign?",
                options=["qualify-the-claim-and-run", "run-as-is", "redo-benchmark-realistic", "drop-the-comparison"]),
    risk=dict(type="Score", instructions="What is the legal exposure of running as-is?",
              levels=["low", "medium", "high"]),
)
add(  # 35 call center HVAC
    "Facilities + ops: chiller serving the 2nd-floor call center failed at 11:20. Room temp now 29.4°C, rising ~1°C/hr with 140 agents on shift. Policy: indoor work stops above 30.5°C for 'heavy' work, guidance for offices at 28°C. Today's call volume is double (product recall in progress) with 4h queue. Portable coolers available: 6 units, cover ~40% of the floor. Remote work would need laptops shipped; agents are desktop-based.",
    stop_work_now=dict(type="Noul",
                       instructions="Does the situation already require stopping office work under the stated guidance?"),
    recall_priority=dict(type="Noul",
                         instructions="Does the active recall justify asking agents to remain past the comfort guidance?"),
    action=dict(type="Choice", instructions="What is the best immediate measure?",
                options=["relocate-agents-in-phases", "coolers-plus-voluntary-stagger", "close-floor-now", "rent-emergency-cooling-unit"]),
    urgency=dict(type="Score", instructions="How urgent is a decision?",
                 levels=["high", "critical"]),
)
add(  # 36 discount beyond floor
    "Deal desk request: renewal + expansion, list EUR 95k. Sales VP asks for 32% discount; authority floor is 25%. Justification given: competitor quote at 38% off list, 'relationship value', and pending reference-call commitment. Analysis: competitor quote is for a 2-year term (ours is 1), includes only 60% of the modules, and the customer's usage is up 20%. Customer has never actually asked for a discount this size.",
    discount_justified=dict(type="Noul",
                            instructions="Does the competitive evidence justify exceeding the discount floor?"),
    vp_request_valid=dict(type="Noul",
                          instructions="Is the VP's framing materially misleading given the term/module differences?"),
    action=dict(type="Choice", instructions="What should deal desk do?",
                options=["counter-at-25-with-term-lever", "approve-32", "reject-and-reprice", "exec-arbitration"]),
    deal_risk=dict(type="Score", instructions="What is the margin risk of caving here?",
                   levels=["minor", "material", "severe"]),
)
add(  # 37 total loss valuation
    "Claim CLM-6120: vehicle written off after a single-vehicle accident. Insurer's valuation tool prices the car (2019, 78k km, one prior owner) at EUR 21,400. Policyholder supplies 4 dealer listings of the same model at EUR 24,800-26,500, all lower-mileage examples (52-61k km), one with different trim. Policy wording: 'market value at time of loss'. Independent guides put mid-point at EUR 22,900 for the actual spec. Policyholder is threatening the ombudsman.",
    valuation_fair=dict(type="Noul",
                        instructions="Is the tool valuation defensible as 'market value' given the listings evidence?"),
    comparables_matched=dict(type="Noul",
                             instructions="Are the policyholder's listings valid comparables for the vehicle's spec?"),
    action=dict(type="Choice", instructions="What is the best next step?",
                options=["revised-offer-near-guide-midpoint", "hold-tool-valuation", "independent-appraiser", "goodwill-top-up-without-admission"]),
    ombudsman_risk=dict(type="Score", instructions="If unresolved, how likely is an adverse ombudsman outcome on this evidence?",
                        levels=["possible", "likely", "near-certain"]),
)
add(  # 38 owner response defamation
    "A restaurant owner wants to reply publicly to a 1-star review: 'This man was drunk, abused our staff for 30 minutes, and is banned from every restaurant on this street. Liar.' The reviewer's post says 'rude service, cold food, waited 50 minutes'. No staff incident report exists from that night; one server recalls 'a loud guest'. Platform policy: responses must not contain personal attacks. Owner's son has posted 3 similar replies to other reviews this month.",
    reply_policy_ok=dict(type="Noul",
                         instructions="Would the drafted reply violate the platform's response policy?"),
    defamation_risk=dict(type="Noul",
                         instructions="Does publishing the specific accusations create defamation exposure given the missing incident report?"),
    action=dict(type="Choice", instructions="What should happen?",
                options=["rewrite-to-facts-and-invite-back", "block-the-reply", "post-as-drafted", "private-message-instead"]),
    pattern_concern=dict(type="Noul",
                         instructions="Does the son's posting pattern warrant an account-level intervention?"),
)
add(  # 39 reference red flag
    "Final-stage hire decision: candidate for Senior Backend Engineer aced 5 interviews (system design 'exceptional'). Reference 1 (former manager, 4 years): 'excellent engineer, would rehire'. Reference 2 (former teammate): 'brilliant, but repeatedly missed handoff commitments, two projects slipped; I'd want clear scope fences'. A public commit history shows strong solo work. Background check clean. The role is on a 6-person on-call rotation with strict handoff SLAs. One other finalist is available but measurably weaker on design.",
    concern_material=dict(type="Noul",
                          instructions="Is the reliability concern material for THIS role specifically?"),
    reference_disqualifying=dict(type="Noul",
                                 instructions="Does either reference, on its own, warrant rejecting the candidate?"),
    decision=dict(type="Choice", instructions="What is the better decision?",
                  options=["hire-with-structure-and-review-at-90d", "hire-other-finalist", "additional-references-first", "reopen-search"]),
    risk=dict(type="Score", instructions="What is the risk level of hiring this candidate into this team?",
              levels=["low", "medium", "high"]),
)

FIFTH = {  # state index (1-based) -> fifth question, keeps 5 per state
    11: ("time_sensitivity", dict(type="Score", instructions="How time-sensitive is the pricing discussion?", levels=["low", "medium", "high"])),
    13: ("undue_hardship_shown", dict(type="Noul", instructions="Does the Friday peak-staffing constraint amount to undue hardship on the evidence given?")),
    14: ("should_have_escalated", dict(type="Noul", instructions="Should the account manager have escalated the customer's verbal intent before the notice deadline passed?")),
    16: ("commercial_value", dict(type="Score", instructions="How much commercial value rides on this specific testimonial?", levels=["minor", "moderate", "significant"])),
    17: ("safety_risk", dict(type="Noul", instructions="Does the failed inspection pose a safety risk to elevator passengers?")),
    18: ("disclose_to_agency", dict(type="Noul", instructions="Should the former employee's message be disclosed to the agency's procurement office?")),
    19: ("evidence_strength", dict(type="Score", instructions="How strong is the physical evidence overall?", levels=["weak", "mixed", "strong"])),
    20: ("merchant_comms", dict(type="Choice", instructions="What should the merchant be told meanwhile?", options=["investigation-underway", "no-comment-yet", "temporary-badge-added"])),
    21: ("decision_pressure", dict(type="Score", instructions="How much pressure does the 3-week deadline put on decision quality?", levels=["low", "moderate", "high"])),
    22: ("written_exposure", dict(type="Noul", instructions="Would confirming Q1 in writing create legal exposure for the company?")),
    23: ("governance_risk", dict(type="Score", instructions="What is the data-governance risk level of this pattern?", levels=["low", "medium", "high"])),
    25: ("approver_policy_change", dict(type="Noul", instructions="Should the policy be changed so approvers cannot approve expenses they participated in?")),
    26: ("history_relevant", dict(type="Noul", instructions="Should the poster's recent posting history influence this specific decision?")),
    27: ("owner", dict(type="Choice", instructions="Who should own the follow-up?", options=["product-team", "sales", "customer-success", "nobody-monitor"])),
    28: ("airfreight_partial", dict(type="Noul", instructions="Should airfreight be considered for part of the order?")),
    29: ("flag_pattern", dict(type="Noul", instructions="Should the requesting manager's justification pattern be flagged for audit review?")),
    30: ("adoption_outlook", dict(type="Score", instructions="Without intervention, what is the likelihood this account reaches healthy adoption?", levels=["low", "medium", "high"])),
    31: ("lead_discipline", dict(type="Noul", instructions="Should the team lead face disciplinary action pending the investigation?")),
    32: ("if_rejected", dict(type="Choice", instructions="If the customer rejects the compromise, what is the next move?", options=["walk-away", "meet-in-middle", "board-approval-for-unlimited", "pause-deal"])),
    33: ("law_enforcement", dict(type="Noul", instructions="Given the serial-number swaps, should law enforcement be involved?")),
    34: ("qualify_cost", dict(type="Score", instructions="How much campaign value is lost by qualifying the claim?", levels=["minor", "moderate", "major"])),
    35: ("who_decides", dict(type="Choice", instructions="Who should decide on floor closure?", options=["facilities-director", "ops-lead-on-duty", "hr-safety-officer", "joint-incident-call"])),
    36: ("customer_asked_number", dict(type="Noul", instructions="Has the customer actually requested a specific discount amount?")),
    37: ("comms", dict(type="Choice", instructions="How should communications with the policyholder proceed?", options=["written-reasons-with-revised-offer", "call-first-then-letter", "hold-for-ombudsman"])),
    38: ("review_damage", dict(type="Score", instructions="How damaging is the review itself to the business?", levels=["minor", "moderate", "significant"])),
    39: ("raise_oncall", dict(type="Noul", instructions="Should the on-call handoff expectations be raised with the candidate before any offer?")),
}

for idx, (qid, q) in FIFTH.items():
    S[idx - 1][1][qid] = q

out = []
for state, qs in S:
    out.append(json.dumps({"state": state, "questions": qs}, ensure_ascii=False))
assert len(S) == 39 and all(len(qs) == 5 for _, qs in S), "shape drift"
text = "\n".join(out) + "\n"

from collections import Counter
tc = Counter(q["type"] for _, qs in S for q in qs.values())
print(f"states={len(S)} questions={sum(tc.values())} types={dict(tc)}")
shipped = ROOT / "data/eval/questions_v2.jsonl"
if _args.check:
    same = shipped.read_text() == text
    print(f"{shipped.relative_to(ROOT)}: "
          + ("regenerates bit-for-bit" if same else "DIFFERS from the regenerated text"))
    sys.exit(0 if same else 1)
target = check_out(_ap, _args.out, _args.force)
target.write_text(text)
print(f"wrote {target}")
