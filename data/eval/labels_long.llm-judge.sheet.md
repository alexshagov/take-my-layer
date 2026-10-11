# Long-context eval — labeling sheet (BLIND) — LLM-judge labels (claude-opus-5-5, 2026-10-08)

> NOT human ground truth. Filled by an LLM judge without viewing any model/Jev outputs.

6 states x 5 questions = 30 judgments. For each question, write your
best judgment as a human expert reading the SAME case file the models
read. Rules: answer with the EXACT option/level text shown (or yes/no);
do not look at any model output while labeling; 'unsure' is an allowed,
honest answer. Fill the ANSWER: lines in place, save the file.


---

## State 1 (source row 0, 16092 chars)

INCIDENT #INC-4402 — checkout /quote failures after pricing deploy.
Timeline + machine log excerpts below (all times UTC, 13:00 start):

[13:02] deploy: rollout 'pricing-svc v2.14.3' canary 10% started (commit 8f3ac21, config: new FX_RATE_CACHE=1)
[13:06] deploy: canary healthy: err 0.2%, p99 810ms — promoting to 50%
[13:09] deploy: promotion to 50% done; canary analysis PASSED (window 3m)
[13:12] metrics:checkout-api: error_rate=1.8% p99=980ms (baseline 0.2%/810ms)
[13:14] alerts: SLO burn-rate fast fired: checkout 5xx > 1% for 5m
[13:15] oncall: acknowledged (j.rivera). triage: errors all in /quote endpoint, FX conversion step
[13:17] metrics:fx-service: error_rate=0.3% — fx-service itself healthy
[13:19] logs:checkout-api: stack: CacheMissError -> fallback path -> timeout 3000ms at fx_rate_client.get()
[13:21] oncall: config audit: FX_RATE_CACHE=1 is NEW in v2.14.3; cache client not in canary's code path difference? re-checking
[13:23] deploy: canary analysis had FX traffic at 2% (shadowed by A/B split) — believed related
[13:24] oncall: rollback decision requested; release manager asks for 3 more minutes of data
[13:27] alerts: checkout 5xx 14.2%, 41 customer sessions failed payment step in last 10m
[13:12] metrics:checkout-api error_rate=1.8% p99_latency=840ms inflight=1200
[13:17] metrics:checkout-api error_rate=3.5% p99_latency=1030ms inflight=1340
[13:22] metrics:checkout-api error_rate=5.2% p99_latency=1220ms inflight=1480
[13:27] metrics:checkout-api error_rate=6.9% p99_latency=1410ms inflight=1620
[13:32] metrics:checkout-api error_rate=8.6% p99_latency=1600ms inflight=1760
[13:37] metrics:checkout-api error_rate=10.3% p99_latency=1790ms inflight=1900
[13:42] metrics:checkout-api error_rate=12.0% p99_latency=1980ms inflight=2040
[13:47] metrics:checkout-api error_rate=13.7% p99_latency=2170ms inflight=2180

Context: Black Friday warm-up traffic (2.1x normal). Rollback takes ~6 min (blue-green). The FX cache flag was reviewed in PR #3391 by two seniors, approved. The release manager notes canary metrics looked clean. Marketing has a paid push scheduled 14:00. No data is corrupt; failures are user-visible at payment step.

APPENDIX — full metric series (5-min buckets, all UTC):
bucket T+0 [13:10] checkout error=1.0%  p99=800ms  fx cache_hit=100%  fx fallback=0%
bucket T+1 [13:15] checkout error=1.9%  p99=965ms  fx cache_hit=100%  fx fallback=0%
bucket T+2 [13:20] checkout error=2.8%  p99=1130ms  fx cache_hit=44%  fx fallback=22%
bucket T+3 [13:25] checkout error=3.7%  p99=1295ms  fx cache_hit=36%  fx fallback=33%
bucket T+4 [13:30] checkout error=4.6%  p99=1460ms  fx cache_hit=28%  fx fallback=44%
bucket T+5 [13:35] checkout error=5.5%  p99=1625ms  fx cache_hit=20%  fx fallback=55%
bucket T+6 [13:40] checkout error=6.4%  p99=1790ms  fx cache_hit=12%  fx fallback=66%
bucket T+7 [13:45] checkout error=7.3%  p99=1955ms  fx cache_hit=4%  fx fallback=77%
bucket T+8 [13:50] checkout error=8.2%  p99=2120ms  fx cache_hit=0%  fx fallback=88%
bucket T+9 [13:55] checkout error=9.1%  p99=2285ms  fx cache_hit=0%  fx fallback=99%
bucket T+10 [14:00] checkout error=10.0%  p99=2450ms  fx cache_hit=0%  fx fallback=100%
bucket T+11 [14:05] checkout error=10.9%  p99=2615ms  fx cache_hit=0%  fx fallback=100%
bucket T+12 [14:10] checkout error=11.8%  p99=2780ms  fx cache_hit=0%  fx fallback=100%
bucket T+13 [14:15] checkout error=12.7%  p99=2945ms  fx cache_hit=0%  fx fallback=100%
bucket T+14 [14:20] checkout error=13.6%  p99=3110ms  fx cache_hit=0%  fx fallback=100%
bucket T+15 [14:25] checkout error=14.5%  p99=3275ms  fx cache_hit=0%  fx fallback=100%
bucket T+16 [14:30] checkout error=15.4%  p99=3440ms  fx cache_hit=0%  fx fallback=100%
bucket T+17 [14:35] checkout error=16.3%  p99=3605ms  fx cache_hit=0%  fx fallback=100%
bucket T+18 [14:40] checkout error=17.2%  p99=3770ms  fx cache_hit=0%  fx fallback=100%
bucket T+19 [14:45] checkout error=18.1%  p99=3935ms  fx cache_hit=0%  fx fallback=100%
bucket T+20 [14:50] checkout error=19.0%  p99=4100ms  fx cache_hit=0%  fx fallback=100%
bucket T+21 [14:55] checkout error=19.9%  p99=4265ms  fx cache_hit=0%  fx fallback=100%
bucket T+22 [15:00] checkout error=20.8%  p99=4430ms  fx cache_hit=0%  fx fallback=100%
bucket T+23 [15:05] checkout error=21.7%  p99=4595ms  fx cache_hit=0%  fx fallback=100%
bucket T+24 [15:10] checkout error=22.6%  p99=4760ms  fx cache_hit=0%  fx fallback=100%
bucket T+25 [15:15] checkout error=23.5%  p99=4925ms  fx cache_hit=0%  fx fallback=100%
bucket T+26 [15:20] checkout error=24.4%  p99=5090ms  fx cache_hit=0%  fx fallback=100%
bucket T+27 [15:25] checkout error=25.3%  p99=5255ms  fx cache_hit=0%  fx fallback=100%
bucket T+28 [15:30] checkout error=26.2%  p99=5420ms  fx cache_hit=0%  fx fallback=100%
bucket T+29 [15:35] checkout error=27.1%  p99=5585ms  fx cache_hit=0%  fx fallback=100%
bucket T+30 [15:40] checkout error=28.0%  p99=5750ms  fx cache_hit=0%  fx fallback=100%
bucket T+31 [15:45] checkout error=28.9%  p99=5915ms  fx cache_hit=0%  fx fallback=100%
bucket T+32 [15:50] checkout error=29.8%  p99=6080ms  fx cache_hit=0%  fx fallback=100%
bucket T+33 [15:55] checkout error=30.7%  p99=6245ms  fx cache_hit=0%  fx fallback=100%
bucket T+34 [16:00] checkout error=31.6%  p99=6410ms  fx cache_hit=0%  fx fallback=100%
bucket T+35 [16:05] checkout error=32.5%  p99=6575ms  fx cache_hit=0%  fx fallback=100%
bucket T+36 [16:10] checkout error=33.4%  p99=6740ms  fx cache_hit=0%  fx fallback=100%
bucket T+37 [16:15] checkout error=34.3%  p99=6905ms  fx cache_hit=0%  fx fallback=100%
bucket T+38 [16:20] checkout error=35.2%  p99=7070ms  fx cache_hit=0%  fx fallback=100%
bucket T+39 [16:25] checkout error=36.1%  p99=7235ms  fx cache_hit=0%  fx fallback=100%
bucket T+40 [16:30] checkout error=37.0%  p99=7400ms  fx cache_hit=0%  fx fallback=100%
bucket T+41 [16:35] checkout error=37.9%  p99=7565ms  fx cache_hit=0%  fx fallback=100%
bucket T+42 [16:40] checkout error=38.8%  p99=7730ms  fx cache_hit=0%  fx fallback=100%
bucket T+43 [16:45] checkout error=39.7%  p99=7895ms  fx cache_hit=0%  fx fallback=100%
bucket T+44 [16:50] checkout error=40.6%  p99=8060ms  fx cache_hit=0%  fx fallback=100%
bucket T+45 [16:55] checkout error=41.5%  p99=8225ms  fx cache_hit=0%  fx fallback=100%
bucket T+46 [17:00] checkout error=42.4%  p99=8390ms  fx cache_hit=0%  fx fallback=100%
bucket T+47 [17:05] checkout error=43.3%  p99=8555ms  fx cache_hit=0%  fx fallback=100%
bucket T+48 [17:10] checkout error=44.2%  p99=8720ms  fx cache_hit=0%  fx fallback=100%
bucket T+49 [17:15] checkout error=45.1%  p99=8885ms  fx cache_hit=0%  fx fallback=100%
bucket T+50 [17:20] checkout error=46.0%  p99=9050ms  fx cache_hit=0%  fx fallback=100%
bucket T+51 [17:25] checkout error=46.9%  p99=9215ms  fx cache_hit=0%  fx fallback=100%
bucket T+52 [17:30] checkout error=47.8%  p99=9380ms  fx cache_hit=0%  fx fallback=100%
bucket T+53 [17:35] checkout error=48.7%  p99=9545ms  fx cache_hit=0%  fx fallback=100%
bucket T+54 [17:40] checkout error=49.6%  p99=9710ms  fx cache_hit=0%  fx fallback=100%
bucket T+55 [17:45] checkout error=50.5%  p99=9875ms  fx cache_hit=0%  fx fallback=100%
bucket T+56 [17:50] checkout error=51.4%  p99=10040ms  fx cache_hit=0%  fx fallback=100%
bucket T+57 [17:55] checkout error=52.3%  p99=10205ms  fx cache_hit=0%  fx fallback=100%
bucket T+58 [18:00] checkout error=53.2%  p99=10370ms  fx cache_hit=0%  fx fallback=100%
bucket T+59 [18:05] checkout error=54.1%  p99=10535ms  fx cache_hit=0%  fx fallback=100%
bucket T+60 [18:10] checkout error=55.0%  p99=10700ms  fx cache_hit=0%  fx fallback=100%
bucket T+61 [18:15] checkout error=55.9%  p99=10865ms  fx cache_hit=0%  fx fallback=100%
bucket T+62 [18:20] checkout error=56.8%  p99=11030ms  fx cache_hit=0%  fx fallback=100%
bucket T+63 [18:25] checkout error=57.7%  p99=11195ms  fx cache_hit=0%  fx fallback=100%
bucket T+64 [18:30] checkout error=58.6%  p99=11360ms  fx cache_hit=0%  fx fallback=100%
bucket T+65 [18:35] checkout error=59.5%  p99=11525ms  fx cache_hit=0%  fx fallback=100%
bucket T+66 [18:40] checkout error=60.4%  p99=11690ms  fx cache_hit=0%  fx fallback=100%
bucket T+67 [18:45] checkout error=61.3%  p99=11855ms  fx cache_hit=0%  fx fallback=100%
bucket T+68 [18:50] checkout error=62.2%  p99=12020ms  fx cache_hit=0%  fx fallback=100%
bucket T+69 [18:55] checkout error=63.1%  p99=12185ms  fx cache_hit=0%  fx fallback=100%
bucket T+70 [19:00] checkout error=64.0%  p99=12350ms  fx cache_hit=0%  fx fallback=100%
bucket T+71 [19:05] checkout error=64.9%  p99=12515ms  fx cache_hit=0%  fx fallback=100%
bucket G+0 [13:10] payment-gateway ok=98.9%  settle_p99=310ms
bucket G+1 [13:15] payment-gateway ok=98.9%  settle_p99=311ms
bucket G+2 [13:20] payment-gateway ok=98.9%  settle_p99=312ms
bucket G+3 [13:25] payment-gateway ok=98.8%  settle_p99=313ms
bucket G+4 [13:30] payment-gateway ok=98.8%  settle_p99=314ms
bucket G+5 [13:35] payment-gateway ok=98.8%  settle_p99=315ms
bucket G+6 [13:40] payment-gateway ok=98.8%  settle_p99=316ms
bucket G+7 [13:45] payment-gateway ok=98.8%  settle_p99=317ms
bucket G+8 [13:50] payment-gateway ok=98.7%  settle_p99=318ms
bucket G+9 [13:55] payment-gateway ok=98.7%  settle_p99=319ms
bucket G+10 [14:00] payment-gateway ok=98.7%  settle_p99=320ms
bucket G+11 [14:05] payment-gateway ok=98.7%  settle_p99=321ms
bucket G+12 [14:10] payment-gateway ok=98.7%  settle_p99=322ms
bucket G+13 [14:15] payment-gateway ok=98.6%  settle_p99=323ms
bucket G+14 [14:20] payment-gateway ok=98.6%  settle_p99=324ms
bucket G+15 [14:25] payment-gateway ok=98.6%  settle_p99=325ms
bucket G+16 [14:30] payment-gateway ok=98.6%  settle_p99=326ms
bucket G+17 [14:35] payment-gateway ok=98.6%  settle_p99=327ms
bucket G+18 [14:40] payment-gateway ok=98.5%  settle_p99=328ms
bucket G+19 [14:45] payment-gateway ok=98.5%  settle_p99=329ms
bucket G+20 [14:50] payment-gateway ok=98.5%  settle_p99=330ms
bucket G+21 [14:55] payment-gateway ok=98.5%  settle_p99=331ms
bucket G+22 [15:00] payment-gateway ok=98.5%  settle_p99=332ms
bucket G+23 [15:05] payment-gateway ok=98.4%  settle_p99=333ms
bucket G+24 [15:10] payment-gateway ok=98.4%  settle_p99=334ms
bucket G+25 [15:15] payment-gateway ok=98.4%  settle_p99=335ms
bucket G+26 [15:20] payment-gateway ok=98.4%  settle_p99=336ms
bucket G+27 [15:25] payment-gateway ok=98.4%  settle_p99=337ms
bucket G+28 [15:30] payment-gateway ok=98.3%  settle_p99=338ms
bucket G+29 [15:35] payment-gateway ok=98.3%  settle_p99=339ms
bucket G+30 [15:40] payment-gateway ok=98.3%  settle_p99=340ms
bucket G+31 [15:45] payment-gateway ok=98.3%  settle_p99=341ms
bucket G+32 [15:50] payment-gateway ok=98.3%  settle_p99=342ms
bucket G+33 [15:55] payment-gateway ok=98.2%  settle_p99=343ms
bucket G+34 [16:00] payment-gateway ok=98.2%  settle_p99=344ms
bucket G+35 [16:05] payment-gateway ok=98.2%  settle_p99=345ms
bucket G+36 [16:10] payment-gateway ok=98.2%  settle_p99=346ms
bucket G+37 [16:15] payment-gateway ok=98.2%  settle_p99=347ms
bucket G+38 [16:20] payment-gateway ok=98.1%  settle_p99=348ms
bucket G+39 [16:25] payment-gateway ok=98.1%  settle_p99=349ms
bucket G+40 [16:30] payment-gateway ok=98.1%  settle_p99=350ms
bucket G+41 [16:35] payment-gateway ok=98.1%  settle_p99=351ms
bucket G+42 [16:40] payment-gateway ok=98.1%  settle_p99=352ms
bucket G+43 [16:45] payment-gateway ok=98.0%  settle_p99=353ms
bucket G+44 [16:50] payment-gateway ok=98.0%  settle_p99=354ms
bucket G+45 [16:55] payment-gateway ok=98.0%  settle_p99=355ms
bucket G+46 [17:00] payment-gateway ok=98.0%  settle_p99=356ms
bucket G+47 [17:05] payment-gateway ok=98.0%  settle_p99=357ms
bucket G+48 [17:10] payment-gateway ok=97.9%  settle_p99=358ms
bucket G+49 [17:15] payment-gateway ok=97.9%  settle_p99=359ms
bucket G+50 [17:20] payment-gateway ok=97.9%  settle_p99=360ms
bucket G+51 [17:25] payment-gateway ok=97.9%  settle_p99=361ms
bucket G+52 [17:30] payment-gateway ok=97.9%  settle_p99=362ms
bucket G+53 [17:35] payment-gateway ok=97.8%  settle_p99=363ms
bucket G+54 [17:40] payment-gateway ok=97.8%  settle_p99=364ms
bucket G+55 [17:45] payment-gateway ok=97.8%  settle_p99=365ms
bucket G+56 [17:50] payment-gateway ok=97.8%  settle_p99=366ms
bucket G+57 [17:55] payment-gateway ok=97.8%  settle_p99=367ms
bucket G+58 [18:00] payment-gateway ok=97.7%  settle_p99=368ms
bucket G+59 [18:05] payment-gateway ok=97.7%  settle_p99=369ms
bucket G+60 [18:10] payment-gateway ok=97.7%  settle_p99=370ms
bucket G+61 [18:15] payment-gateway ok=97.7%  settle_p99=371ms
bucket G+62 [18:20] payment-gateway ok=97.7%  settle_p99=372ms
bucket G+63 [18:25] payment-gateway ok=97.6%  settle_p99=373ms
bucket G+64 [18:30] payment-gateway ok=97.6%  settle_p99=374ms
bucket G+65 [18:35] payment-gateway ok=97.6%  settle_p99=375ms
bucket G+66 [18:40] payment-gateway ok=97.6%  settle_p99=376ms
bucket G+67 [18:45] payment-gateway ok=97.6%  settle_p99=377ms
bucket G+68 [18:50] payment-gateway ok=97.5%  settle_p99=378ms
bucket G+69 [18:55] payment-gateway ok=97.5%  settle_p99=379ms
bucket G+70 [19:00] payment-gateway ok=97.5%  settle_p99=380ms
bucket G+71 [19:05] payment-gateway ok=97.5%  settle_p99=381ms
bucket C+0 [13:20] canary(v2.14.3) err=0.20%  [CLEAN — canary had FX at 2% shadow]
bucket C+1 [13:25] canary(v2.14.3) err=0.25%  [CLEAN — canary had FX at 2% shadow]
bucket C+2 [13:30] canary(v2.14.3) err=0.30%  [CLEAN — canary had FX at 2% shadow]
bucket C+3 [13:35] canary(v2.14.3) err=0.35%  [CLEAN — canary had FX at 2% shadow]
bucket C+4 [13:40] canary(v2.14.3) err=0.40%  [CLEAN — canary had FX at 2% shadow]
bucket C+5 [13:45] canary(v2.14.3) err=0.45%  [CLEAN — canary had FX at 2% shadow]
bucket C+6 [13:50] canary(v2.14.3) err=0.50%  [CLEAN — canary had FX at 2% shadow]
bucket C+7 [13:55] canary(v2.14.3) err=0.55%  [CLEAN — canary had FX at 2% shadow]
bucket C+8 [14:00] canary(v2.14.3) err=0.60%  [CLEAN — canary had FX at 2% shadow]
bucket C+9 [14:05] canary(v2.14.3) err=0.65%  [CLEAN — canary had FX at 2% shadow]
bucket C+10 [14:10] canary(v2.14.3) err=0.70%  [CLEAN — canary had FX at 2% shadow]
bucket C+11 [14:15] canary(v2.14.3) err=0.75%  [CLEAN — canary had FX at 2% shadow]
bucket C+12 [14:20] canary(v2.14.3) err=0.80%  [CLEAN — canary had FX at 2% shadow]
bucket C+13 [14:25] canary(v2.14.3) err=0.85%  [CLEAN — canary had FX at 2% shadow]
bucket C+14 [14:30] canary(v2.14.3) err=0.90%  [CLEAN — canary had FX at 2% shadow]
bucket C+15 [14:35] canary(v2.14.3) err=0.95%  [CLEAN — canary had FX at 2% shadow]
bucket C+16 [14:40] canary(v2.14.3) err=1.00%  [CLEAN — canary had FX at 2% shadow]
bucket C+17 [14:45] canary(v2.14.3) err=1.05%  [CLEAN — canary had FX at 2% shadow]
bucket C+18 [14:50] canary(v2.14.3) err=1.10%  [CLEAN — canary had FX at 2% shadow]
bucket C+19 [14:55] canary(v2.14.3) err=1.15%  [CLEAN — canary had FX at 2% shadow]
bucket C+20 [15:00] canary(v2.14.3) err=1.20%  [CLEAN — canary had FX at 2% shadow]
bucket C+21 [15:05] canary(v2.14.3) err=1.25%  [CLEAN — canary had FX at 2% shadow]
bucket C+22 [15:10] canary(v2.14.3) err=1.30%  [CLEAN — canary had FX at 2% shadow]
bucket C+23 [15:15] canary(v2.14.3) err=1.35%  [CLEAN — canary had FX at 2% shadow]
bucket C+24 [15:20] canary(v2.14.3) err=1.40%  [CLEAN — canary had FX at 2% shadow]
bucket C+25 [15:25] canary(v2.14.3) err=1.45%  [CLEAN — canary had FX at 2% shadow]
bucket C+26 [15:30] canary(v2.14.3) err=1.50%  [CLEAN — canary had FX at 2% shadow]
bucket C+27 [15:35] canary(v2.14.3) err=1.55%  [CLEAN — canary had FX at 2% shadow]
bucket C+28 [15:40] canary(v2.14.3) err=1.60%  [CLEAN — canary had FX at 2% shadow]
bucket C+29 [15:45] canary(v2.14.3) err=1.65%  [CLEAN — canary had FX at 2% shadow]
bucket C+30 [15:50] canary(v2.14.3) err=1.70%  [CLEAN — canary had FX at 2% shadow]
bucket C+31 [15:55] canary(v2.14.3) err=1.75%  [CLEAN — canary had FX at 2% shadow]
bucket C+32 [16:00] canary(v2.14.3) err=1.80%  [CLEAN — canary had FX at 2% shadow]
bucket C+33 [16:05] canary(v2.14.3) err=1.85%  [CLEAN — canary had FX at 2% shadow]
bucket C+34 [16:10] canary(v2.14.3) err=1.90%  [CLEAN — canary had FX at 2% shadow]
bucket C+35 [16:15] canary(v2.14.3) err=1.95%  [CLEAN — canary had FX at 2% shadow]

### Questions

#### root_cause [Choice]

What is the most likely root cause on this evidence?

- deploy-v2.14.3-fx-cache-path
- fx-service-outage
- traffic-overload
- network-issue

ANSWER (root_cause): deploy-v2.14.3-fx-cache-path
CacheMissError->fallback timeout on the new FX_RATE_CACHE path; fx-service healthy, gateway flat, canary blind (FX at 2%).

#### severity [Score]

Incident severity?

- level 0: sev3
- level 1: sev2
- level 2: sev1

ANSWER (severity): level 2: sev1
Payment step failing at 14%+ and climbing linearly, Black Friday warm-up traffic, SLO fast-burn fired.

#### rollback_now [Noul]

Given the trend and rollback cost, should rollback start immediately rather than wait for more data?


ANSWER (rollback_now): yes
Error rate rises ~1.8 pts/5min and rollback takes ~6 min; waiting 3 more minutes buys no information.

#### customer_comms [Noul]

Should the 41 affected payment sessions trigger proactive customer outreach before resolution?


ANSWER (customer_comms): no
Mitigate first; no data corruption and gateway OK, so targeted outreach can follow resolution (status-page notice is a separate matter).

#### next_action [Choice]

Best next action while rollback is pending?

- rollback-now
- feature-flag-off-fx-cache
- throttle-traffic
- escalate-to-eng-director

ANSWER (next_action): feature-flag-off-fx-cache
The regression sits behind the new config flag; switching it off is likely faster than the 6-min blue-green rollback and doesn't conflict with it.


---

## State 2 (source row 1, 17621 chars)

Escalated support ticket #77120 (account 'Nordvik Analytics', 14 months tenure, Pro plan EUR 89/mo, CSAT history 4.5). Full thread:

Mar 3, 09:12 — customer:
  Webhook deliveries to our endpoint stopped around Feb 28. We retried nothing — dashboard says 'delivered'. This is the THIRD interruption this quarter. Our billing reconciliation is broken downstream.
Mar 3, 09:40 — support L1:
  Hi! I see all green on our side. Could you share 2-3 delivery IDs?
Mar 3, 10:05 — customer:
  IDs: dlv_8812 (Feb 28 22:14), dlv_8813, dlv_8814. Note they say 200 OK but our server logged nothing that day — we had a firewall change Feb 28 21:00. Possible your retries hit a stale IP?
Mar 3, 10:31 — support L1:
  Those show delivered 200. If your firewall changed, we can't see that. Closing as customer-side.
Mar 3, 11:02 — customer:
  Not so fast — the dashboard said delivered BEFORE the firewall change too (dlv_8790, Feb 27). And docs say you retry 5x over 1h; we see zero retries on any of these. Please reopen.
Mar 3, 11:45 — support L2:
  Reopened. Checking delivery logs... our retry subsystem had a config regression Feb 26-March 2 (internal INC-3398): retries were marked 'delivered' on first 4xx/5xx bucket flush. That's our bug. 3rd interruption: Jan 12 (same class), Feb 3 (different).
Mar 3, 12:20 — customer:
  So the dashboard lied to us for 4 days. We lost a client demo over the Jan incident. What are you doing about it?
Mar 3, 13:05 — support L2:
  Confirmed regression. I can offer 1 month credit (EUR 89). Engineering fix ships March 5. I've flagged your account for proactive incident notifications.
Mar 3, 13:40 — customer:
  1 month for 3 incidents and a lost demo? We pay EUR 89/mo for reliability. We're evaluating alternatives unless this is taken seriously.
Mar 3, 14:15 — support L2:
  Let me escalate internally. Response by tomorrow.

Internal notes: NPS last quarter 6/10. The January incident was compensated EUR 45. Account usage grew 30% since November. Renewal is in 58 days.

APPENDIX A — prior incidents on this account (status history):
Jan 12 (14:02-15:26, 84m) webhook delivery degradation — cause: retry-subsystem config regression (INC-3301, same class as current). 214 deliveries marked 'delivered' without receipt. Compensation: EUR 45 credit. Post-incident review: 'add end-to-end receipt verification' — action item AI-4 open 61 days past due.
Feb 3 (09:15-09:58, 43m) dashboard latency — cause: analytics index bloat (different subsystem, INC-3355). No data impact. No compensation requested.
  Delivery audit extract, Feb 26 - Mar 3 (3-min sample rows):
dlv_0  id=dlv_8790  ts=Feb 26 00:00  endpoint=eu-2  status=200  attempt=1  latency=180ms
dlv_1  id=dlv_8791  ts=Feb 26 03:17  endpoint=eu-2  status=200  attempt=1  latency=193ms
dlv_2  id=dlv_8792  ts=Feb 26 06:34  endpoint=eu-2  status=200  attempt=1  latency=206ms
dlv_3  id=dlv_8793  ts=Feb 26 09:51  endpoint=eu-2  status=delivered*(regression: first-write failed, retry never sent)  attempt=1  latency=219ms
dlv_4  id=dlv_8794  ts=Feb 26 12:08  endpoint=eu-2  status=200  attempt=1  latency=232ms
dlv_5  id=dlv_8795  ts=Feb 26 15:25  endpoint=eu-2  status=200  attempt=1  latency=245ms
dlv_6  id=dlv_8796  ts=Feb 26 18:42  endpoint=eu-2  status=200  attempt=1  latency=258ms
dlv_7  id=dlv_8797  ts=Feb 26 21:59  endpoint=eu-2  status=200  attempt=1  latency=271ms
dlv_8  id=dlv_8798  ts=Feb 27 00:16  endpoint=eu-2  status=200  attempt=1  latency=284ms
dlv_9  id=dlv_8799  ts=Feb 27 03:33  endpoint=eu-2  status=200  attempt=1  latency=297ms
dlv_10  id=dlv_8800  ts=Feb 27 06:50  endpoint=eu-2  status=delivered*(regression: first-write failed, retry never sent)  attempt=1  latency=310ms
dlv_11  id=dlv_8801  ts=Feb 27 09:07  endpoint=eu-2  status=200  attempt=1  latency=323ms
dlv_12  id=dlv_8802  ts=Feb 27 12:24  endpoint=eu-2  status=200  attempt=1  latency=336ms
dlv_13  id=dlv_8803  ts=Feb 27 15:41  endpoint=eu-2  status=200  attempt=1  latency=349ms
dlv_14  id=dlv_8804  ts=Feb 27 18:58  endpoint=eu-2  status=200  attempt=1  latency=362ms
dlv_15  id=dlv_8805  ts=Feb 27 21:15  endpoint=eu-2  status=200  attempt=1  latency=375ms
dlv_16  id=dlv_8806  ts=Feb 28 00:32  endpoint=eu-2  status=200  attempt=1  latency=388ms
dlv_17  id=dlv_8807  ts=Feb 28 03:49  endpoint=eu-2  status=delivered*(regression: first-write failed, retry never sent)  attempt=1  latency=181ms
dlv_18  id=dlv_8808  ts=Feb 28 06:06  endpoint=eu-2  status=200  attempt=1  latency=194ms
dlv_19  id=dlv_8809  ts=Feb 28 09:23  endpoint=eu-2  status=200  attempt=1  latency=207ms
dlv_20  id=dlv_8810  ts=Feb 28 12:40  endpoint=eu-2  status=200  attempt=1  latency=220ms
dlv_21  id=dlv_8811  ts=Feb 28 15:57  endpoint=eu-2  status=200  attempt=1  latency=233ms
dlv_22  id=dlv_8812  ts=Feb 28 18:14  endpoint=eu-2  status=200  attempt=1  latency=246ms
dlv_23  id=dlv_8813  ts=Feb 28 21:31  endpoint=eu-2  status=200  attempt=1  latency=259ms
dlv_24  id=dlv_8814  ts=Feb 29 00:48  endpoint=eu-2  status=delivered*(regression: first-write failed, retry never sent)  attempt=1  latency=272ms
dlv_25  id=dlv_8815  ts=Feb 29 03:05  endpoint=eu-2  status=200  attempt=1  latency=285ms
dlv_26  id=dlv_8816  ts=Feb 29 06:22  endpoint=eu-2  status=200  attempt=1  latency=298ms
dlv_27  id=dlv_8817  ts=Feb 29 09:39  endpoint=eu-2  status=200  attempt=1  latency=311ms
dlv_28  id=dlv_8818  ts=Feb 29 12:56  endpoint=eu-2  status=200  attempt=1  latency=324ms
dlv_29  id=dlv_8819  ts=Feb 29 15:13  endpoint=eu-2  status=200  attempt=1  latency=337ms
dlv_30  id=dlv_8820  ts=Feb 29 18:30  endpoint=eu-2  status=200  attempt=1  latency=350ms
dlv_31  id=dlv_8821  ts=Feb 29 21:47  endpoint=eu-2  status=delivered*(regression: first-write failed, retry never sent)  attempt=1  latency=363ms
dlv_32  id=dlv_8822  ts=Feb 30 00:04  endpoint=eu-2  status=200  attempt=1  latency=376ms
dlv_33  id=dlv_8823  ts=Feb 30 03:21  endpoint=eu-2  status=200  attempt=1  latency=389ms
dlv_34  id=dlv_8824  ts=Feb 30 06:38  endpoint=eu-2  status=200  attempt=1  latency=182ms
dlv_35  id=dlv_8825  ts=Feb 30 09:55  endpoint=eu-2  status=200  attempt=1  latency=195ms
dlv_36  id=dlv_8826  ts=Feb 30 12:12  endpoint=eu-2  status=200  attempt=1  latency=208ms
dlv_37  id=dlv_8827  ts=Feb 30 15:29  endpoint=eu-2  status=200  attempt=1  latency=221ms
dlv_38  id=dlv_8828  ts=Feb 30 18:46  endpoint=eu-2  status=delivered*(regression: first-write failed, retry never sent)  attempt=1  latency=234ms
dlv_39  id=dlv_8829  ts=Feb 30 21:03  endpoint=eu-2  status=200  attempt=1  latency=247ms
dlv_40  id=dlv_8830  ts=Feb 31 00:20  endpoint=eu-2  status=200  attempt=1  latency=260ms
dlv_41  id=dlv_8831  ts=Feb 31 03:37  endpoint=eu-2  status=200  attempt=1  latency=273ms
dlv_42  id=dlv_8832  ts=Feb 31 06:54  endpoint=eu-2  status=200  attempt=1  latency=286ms
dlv_43  id=dlv_8833  ts=Feb 31 09:11  endpoint=eu-2  status=200  attempt=1  latency=299ms
dlv_44  id=dlv_8834  ts=Feb 31 12:28  endpoint=eu-2  status=200  attempt=1  latency=312ms
dlv_45  id=dlv_8835  ts=Feb 31 15:45  endpoint=eu-2  status=delivered*(regression: first-write failed, retry never sent)  attempt=1  latency=325ms
dlv_46  id=dlv_8836  ts=Feb 31 18:02  endpoint=eu-2  status=200  attempt=1  latency=338ms
dlv_47  id=dlv_8837  ts=Feb 31 21:19  endpoint=eu-2  status=200  attempt=1  latency=351ms
dlv_48  id=dlv_8838  ts=Feb 32 00:36  endpoint=eu-2  status=200  attempt=1  latency=364ms
dlv_49  id=dlv_8839  ts=Feb 32 03:53  endpoint=eu-2  status=200  attempt=1  latency=377ms
dlv_50  id=dlv_8840  ts=Feb 32 06:10  endpoint=eu-2  status=200  attempt=1  latency=390ms
dlv_51  id=dlv_8841  ts=Feb 32 09:27  endpoint=eu-2  status=200  attempt=1  latency=183ms
dlv_52  id=dlv_8842  ts=Feb 32 12:44  endpoint=eu-2  status=delivered*(regression: first-write failed, retry never sent)  attempt=1  latency=196ms
dlv_53  id=dlv_8843  ts=Feb 32 15:01  endpoint=eu-2  status=200  attempt=1  latency=209ms
dlv_54  id=dlv_8844  ts=Feb 32 18:18  endpoint=eu-2  status=200  attempt=1  latency=222ms
dlv_55  id=dlv_8845  ts=Feb 32 21:35  endpoint=eu-2  status=200  attempt=1  latency=235ms
dlv_56  id=dlv_8846  ts=Feb 33 00:52  endpoint=eu-2  status=200  attempt=1  latency=248ms
dlv_57  id=dlv_8847  ts=Feb 33 03:09  endpoint=eu-2  status=200  attempt=1  latency=261ms
dlv_58  id=dlv_8848  ts=Feb 33 06:26  endpoint=eu-2  status=200  attempt=1  latency=274ms
dlv_59  id=dlv_8849  ts=Feb 33 09:43  endpoint=eu-2  status=delivered*(regression: first-write failed, retry never sent)  attempt=1  latency=287ms
dlv_60  id=dlv_8850  ts=Feb 33 12:00  endpoint=eu-2  status=200  attempt=1  latency=300ms
dlv_61  id=dlv_8851  ts=Feb 33 15:17  endpoint=eu-2  status=200  attempt=1  latency=313ms
dlv_62  id=dlv_8852  ts=Feb 33 18:34  endpoint=eu-2  status=200  attempt=1  latency=326ms
dlv_63  id=dlv_8853  ts=Feb 33 21:51  endpoint=eu-2  status=200  attempt=1  latency=339ms
dlv_64  id=dlv_8854  ts=Feb 34 00:08  endpoint=eu-2  status=200  attempt=1  latency=352ms
dlv_65  id=dlv_8855  ts=Feb 34 03:25  endpoint=eu-2  status=200  attempt=1  latency=365ms
dlv_66  id=dlv_8856  ts=Feb 34 06:42  endpoint=eu-2  status=delivered*(regression: first-write failed, retry never sent)  attempt=1  latency=378ms
dlv_67  id=dlv_8857  ts=Feb 34 09:59  endpoint=eu-2  status=200  attempt=1  latency=391ms
dlv_68  id=dlv_8858  ts=Feb 34 12:16  endpoint=eu-2  status=200  attempt=1  latency=184ms
dlv_69  id=dlv_8859  ts=Feb 34 15:33  endpoint=eu-2  status=200  attempt=1  latency=197ms
dlv_70  id=dlv_8860  ts=Feb 34 18:50  endpoint=eu-2  status=200  attempt=1  latency=210ms
dlv_71  id=dlv_8861  ts=Feb 34 21:07  endpoint=eu-2  status=200  attempt=1  latency=223ms
dlv_72  id=dlv_8862  ts=Feb 35 00:24  endpoint=eu-2  status=200  attempt=1  latency=236ms
dlv_73  id=dlv_8863  ts=Feb 35 03:41  endpoint=eu-2  status=delivered*(regression: first-write failed, retry never sent)  attempt=1  latency=249ms
dlv_74  id=dlv_8864  ts=Feb 35 06:58  endpoint=eu-2  status=200  attempt=1  latency=262ms
dlv_75  id=dlv_8865  ts=Feb 35 09:15  endpoint=eu-2  status=200  attempt=1  latency=275ms
dlv_76  id=dlv_8866  ts=Feb 35 12:32  endpoint=eu-2  status=200  attempt=1  latency=288ms
dlv_77  id=dlv_8867  ts=Feb 35 15:49  endpoint=eu-2  status=200  attempt=1  latency=301ms
dlv_78  id=dlv_8868  ts=Feb 35 18:06  endpoint=eu-2  status=200  attempt=1  latency=314ms
dlv_79  id=dlv_8869  ts=Feb 35 21:23  endpoint=eu-2  status=200  attempt=1  latency=327ms
dlv_80  id=dlv_8870  ts=Feb 36 00:40  endpoint=eu-2  status=delivered*(regression: first-write failed, retry never sent)  attempt=1  latency=340ms
dlv_81  id=dlv_8871  ts=Feb 36 03:57  endpoint=eu-2  status=200  attempt=1  latency=353ms
dlv_82  id=dlv_8872  ts=Feb 36 06:14  endpoint=eu-2  status=200  attempt=1  latency=366ms
dlv_83  id=dlv_8873  ts=Feb 36 09:31  endpoint=eu-2  status=200  attempt=1  latency=379ms
dlv_84  id=dlv_8874  ts=Feb 36 12:48  endpoint=eu-2  status=200  attempt=1  latency=392ms
dlv_85  id=dlv_8875  ts=Feb 36 15:05  endpoint=eu-2  status=200  attempt=1  latency=185ms
dlv_86  id=dlv_8876  ts=Feb 36 18:22  endpoint=eu-2  status=200  attempt=1  latency=198ms
dlv_87  id=dlv_8877  ts=Feb 36 21:39  endpoint=eu-2  status=delivered*(regression: first-write failed, retry never sent)  attempt=1  latency=211ms
dlv_88  id=dlv_8878  ts=Feb 37 00:56  endpoint=eu-2  status=200  attempt=1  latency=224ms
dlv_89  id=dlv_8879  ts=Feb 37 03:13  endpoint=eu-2  status=200  attempt=1  latency=237ms
dlv_90  id=dlv_8880  ts=Feb 37 06:30  endpoint=eu-2  status=200  attempt=1  latency=250ms
dlv_91  id=dlv_8881  ts=Feb 37 09:47  endpoint=eu-2  status=200  attempt=1  latency=263ms
dlv_92  id=dlv_8882  ts=Feb 37 12:04  endpoint=eu-2  status=200  attempt=1  latency=276ms
dlv_93  id=dlv_8883  ts=Feb 37 15:21  endpoint=eu-2  status=200  attempt=1  latency=289ms
dlv_94  id=dlv_8884  ts=Feb 37 18:38  endpoint=eu-2  status=delivered*(regression: first-write failed, retry never sent)  attempt=1  latency=302ms
dlv_95  id=dlv_8885  ts=Feb 37 21:55  endpoint=eu-2  status=200  attempt=1  latency=315ms
dlv_96  id=dlv_8886  ts=Feb 38 00:12  endpoint=eu-2  status=200  attempt=1  latency=328ms
dlv_97  id=dlv_8887  ts=Feb 38 03:29  endpoint=eu-2  status=200  attempt=1  latency=341ms
dlv_98  id=dlv_8888  ts=Feb 38 06:46  endpoint=eu-2  status=200  attempt=1  latency=354ms
dlv_99  id=dlv_8889  ts=Feb 38 09:03  endpoint=eu-2  status=200  attempt=1  latency=367ms
dlv_100  id=dlv_8890  ts=Feb 38 12:20  endpoint=eu-2  status=200  attempt=1  latency=380ms
dlv_101  id=dlv_8891  ts=Feb 38 15:37  endpoint=eu-2  status=delivered*(regression: first-write failed, retry never sent)  attempt=1  latency=393ms
dlv_102  id=dlv_8892  ts=Feb 38 18:54  endpoint=eu-2  status=200  attempt=1  latency=186ms
dlv_103  id=dlv_8893  ts=Feb 38 21:11  endpoint=eu-2  status=200  attempt=1  latency=199ms
dlv_104  id=dlv_8894  ts=Feb 39 00:28  endpoint=eu-2  status=200  attempt=1  latency=212ms
dlv_105  id=dlv_8895  ts=Feb 39 03:45  endpoint=eu-2  status=200  attempt=1  latency=225ms
dlv_106  id=dlv_8896  ts=Feb 39 06:02  endpoint=eu-2  status=200  attempt=1  latency=238ms
dlv_107  id=dlv_8897  ts=Feb 39 09:19  endpoint=eu-2  status=200  attempt=1  latency=251ms
dlv_108  id=dlv_8898  ts=Feb 39 12:36  endpoint=eu-2  status=delivered*(regression: first-write failed, retry never sent)  attempt=1  latency=264ms
dlv_109  id=dlv_8899  ts=Feb 39 15:53  endpoint=eu-2  status=200  attempt=1  latency=277ms
dlv_110  id=dlv_8900  ts=Feb 39 18:10  endpoint=eu-2  status=200  attempt=1  latency=290ms
dlv_111  id=dlv_8901  ts=Feb 39 21:27  endpoint=eu-2  status=200  attempt=1  latency=303ms
dlv_112  id=dlv_8902  ts=Feb 40 00:44  endpoint=eu-2  status=200  attempt=1  latency=316ms
dlv_113  id=dlv_8903  ts=Feb 40 03:01  endpoint=eu-2  status=200  attempt=1  latency=329ms
dlv_114  id=dlv_8904  ts=Feb 40 06:18  endpoint=eu-2  status=200  attempt=1  latency=342ms
dlv_115  id=dlv_8905  ts=Feb 40 09:35  endpoint=eu-2  status=delivered*(regression: first-write failed, retry never sent)  attempt=1  latency=355ms
dlv_116  id=dlv_8906  ts=Feb 40 12:52  endpoint=eu-2  status=200  attempt=1  latency=368ms
dlv_117  id=dlv_8907  ts=Feb 40 15:09  endpoint=eu-2  status=200  attempt=1  latency=381ms
dlv_118  id=dlv_8908  ts=Feb 40 18:26  endpoint=eu-2  status=200  attempt=1  latency=394ms
dlv_119  id=dlv_8909  ts=Feb 40 21:43  endpoint=eu-2  status=200  attempt=1  latency=187ms
dlv_120  id=dlv_8910  ts=Feb 41 00:00  endpoint=eu-2  status=200  attempt=1  latency=200ms
dlv_121  id=dlv_8911  ts=Feb 41 03:17  endpoint=eu-2  status=200  attempt=1  latency=213ms
dlv_122  id=dlv_8912  ts=Feb 41 06:34  endpoint=eu-2  status=delivered*(regression: first-write failed, retry never sent)  attempt=1  latency=226ms
dlv_123  id=dlv_8913  ts=Feb 41 09:51  endpoint=eu-2  status=200  attempt=1  latency=239ms
dlv_124  id=dlv_8914  ts=Feb 41 12:08  endpoint=eu-2  status=200  attempt=1  latency=252ms
dlv_125  id=dlv_8915  ts=Feb 41 15:25  endpoint=eu-2  status=200  attempt=1  latency=265ms
dlv_126  id=dlv_8916  ts=Feb 41 18:42  endpoint=eu-2  status=200  attempt=1  latency=278ms
dlv_127  id=dlv_8917  ts=Feb 41 21:59  endpoint=eu-2  status=200  attempt=1  latency=291ms
dlv_128  id=dlv_8918  ts=Feb 42 00:16  endpoint=eu-2  status=200  attempt=1  latency=304ms
dlv_129  id=dlv_8919  ts=Feb 42 03:33  endpoint=eu-2  status=delivered*(regression: first-write failed, retry never sent)  attempt=1  latency=317ms
dlv_130  id=dlv_8920  ts=Feb 42 06:50  endpoint=eu-2  status=200  attempt=1  latency=330ms
dlv_131  id=dlv_8921  ts=Feb 42 09:07  endpoint=eu-2  status=200  attempt=1  latency=343ms
dlv_132  id=dlv_8922  ts=Feb 42 12:24  endpoint=eu-2  status=200  attempt=1  latency=356ms
dlv_133  id=dlv_8923  ts=Feb 42 15:41  endpoint=eu-2  status=200  attempt=1  latency=369ms
dlv_134  id=dlv_8924  ts=Feb 42 18:58  endpoint=eu-2  status=200  attempt=1  latency=382ms
dlv_135  id=dlv_8925  ts=Feb 42 21:15  endpoint=eu-2  status=200  attempt=1  latency=395ms
dlv_136  id=dlv_8926  ts=Feb 43 00:32  endpoint=eu-2  status=delivered*(regression: first-write failed, retry never sent)  attempt=1  latency=188ms
dlv_137  id=dlv_8927  ts=Feb 43 03:49  endpoint=eu-2  status=200  attempt=1  latency=201ms
dlv_138  id=dlv_8928  ts=Feb 43 06:06  endpoint=eu-2  status=200  attempt=1  latency=214ms
dlv_139  id=dlv_8929  ts=Feb 43 09:23  endpoint=eu-2  status=200  attempt=1  latency=227ms
dlv_140  id=dlv_8930  ts=Feb 43 12:40  endpoint=eu-2  status=200  attempt=1  latency=240ms
dlv_141  id=dlv_8931  ts=Feb 43 15:57  endpoint=eu-2  status=200  attempt=1  latency=253ms
dlv_142  id=dlv_8932  ts=Feb 43 18:14  endpoint=eu-2  status=200  attempt=1  latency=266ms
dlv_143  id=dlv_8933  ts=Feb 43 21:31  endpoint=eu-2  status=delivered*(regression: first-write failed, retry never sent)  attempt=1  latency=279ms
dlv_144  id=dlv_8934  ts=Feb 44 00:48  endpoint=eu-2  status=200  attempt=1  latency=292ms
dlv_145  id=dlv_8935  ts=Feb 44 03:05  endpoint=eu-2  status=200  attempt=1  latency=305ms
dlv_146  id=dlv_8936  ts=Feb 44 06:22  endpoint=eu-2  status=200  attempt=1  latency=318ms
dlv_147  id=dlv_8937  ts=Feb 44 09:39  endpoint=eu-2  status=200  attempt=1  latency=331ms
dlv_148  id=dlv_8938  ts=Feb 44 12:56  endpoint=eu-2  status=200  attempt=1  latency=344ms
dlv_149  id=dlv_8939  ts=Feb 44 15:13  endpoint=eu-2  status=200  attempt=1  latency=357ms

APPENDIX B — account snapshot: MRR EUR 89; seats 4; API volume 1.1M events/mo (up from 0.8M Nov); open AI-4 (see Jan 12); NPS 6 (Q4) vs 9 (Q3); renewal date 58 days; CSM notes: 'champion is the CTO, pragmatic, hates re-litigating'; competitor mentioned by customer: none until this thread.

### Questions

#### route [Choice]

Who should own this now?

- support-lead
- customer-success-manager
- engineering-liaison
- retention-specialist

ANSWER (route): customer-success-manager
The bug is confirmed and the fix is dated; what remains is the relationship: renewal in 58 days, CSM-known CTO champion, NPS 9->6.

#### sentiment [Score]

Customer sentiment right now?

- level 0: frustrated
- level 1: angry
- level 2: furious

ANSWER (sentiment): level 1: angry
Explicit threat to evaluate alternatives, 'dashboard lied'. Still engaging constructively, so not furious.

#### credit_adequate [Noul]

Is the offered 1-month credit clearly inadequate given the history?


ANSWER (credit_adequate): yes
Third incident, second of the same class, review action AI-4 61 days overdue, a lost demo; EUR 89 doesn't match that.

#### churn_risk [Score]

Churn risk if handled averagely?

- level 0: medium
- level 1: high
- level 2: critical

ANSWER (churn_risk): level 1: high
Renewal in 58 days, explicit alternatives threat, NPS drop. Offsetting: usage +30%, no named competitor.

#### credit_note_issue [Noul]

Was the agent's 'closing as customer-side' at 10:31 a process failure worth coaching?


ANSWER (credit_note_issue): yes
Closed as customer-side without checking delivery logs, despite the customer's specific evidence of no retries.


---

## State 3 (source row 4, 3531 chars)

MODERATION CASE MD-2210 — open-source community forum thread 'v3 release feedback', 14 comments in the last 2h, 3 user flags (2 'hostility', 1 'worried about userA'). Thread:

userA_91: Honestly this maintainer has no idea what he's doing. Every release breaks something. Incompetent.
userB_7: Says the guy who can't read a changelog lol
userA_91: @userB_7 mind your own business, sycophant. Does he pay you in early access?
userB_7: at least I read docs before whining
userC_2: Both of you chill. Report and move on.
userA_91: @userC_2 reported for what, telling the truth? This place is a cult around one guy.
userD_44: A is right though — v3 broke our prod twice. We're forking.
userE_8: @userA_91 you keep saying 'cult' — you've posted this 6 times this month across threads. Get a hobby.
userA_91: @userE_8 maybe I'll stop when your favorite maintainer stops breaking prod. Or ever.
userF_0: guys please, this used to be a nice community
userE_8: I mean it kindly: the repetitive posting looks like it's hurting you more than him.
userA_91: everything hurts lately. not that anyone here would care.
userF_0: @userA_91 hey, I care. DM me? sometimes talking helps. r/mentalhealth resources are decent too
userC_2: flagging for mods — tone changed

Context: userA_91 has 4 prior hostility warnings (none in 5 months), is a top-10 contributor historically. Rules: personal attacks = remove + escalating bans; welfare concerns = resource DM, never public. The maintainer targeted by A's criticism has not commented.

APPENDIX — thread continuation (last hour):
userG_12: v3 broke our importer too but this thread stopped being about that
userE_8: @userG_12 right. started as release feedback, now it's A vs everyone
userA_91: classic. blame the one calling it out
userH_3: @userA_91 you've called the maintainer 'incompetent' 11 times across threads this month. that's not feedback
userA_91: it's accurate feedback
userH_3: it's the eleventh time
userA_91: and every time v3 breaks something I'll say it an eleventh more
userF_0: @everyone stepping away from the thread for a bit. A, my DM stands
userD_44: mods, lock it? this stopped being productive 40 comments ago
userE_8: agreed. A, genuinely — the repetition isn't hurting him, it's hurting you
userA_91: maybe everything is hurting me. didn't expect anyone here to notice or care
userC_2: we noticed. flagging again
userI_5: wow, this thread. A has a point about v3 AND is melting down, both can be true
userJ_9: @userI_5 honestly the melt-down started before v3. check his post history
userA_91: @userJ_9 checking MY post history? get a life
userJ_9: proving the point but ok
userG_12: can we not. F's DM offer is the only useful thing here
userK_1: 3rd flag from me. the last two A messages are different in kind, not just tone
userH_3: agreed. @mods please treat as welfare + conduct, not just conduct
userA_91: don't do me any favors
userF_0: A — standing offer. no pressure, no screenshots, just a human
userA_91: ...

APPENDIX — userA_91 mod history (last 5 months):
  Sep: warning 1 (personal attack on contributor 'incompetent fraud') — acknowledged, cooled
  Oct: warning 2 ('shill' pile-on in release thread) — 3-day mute
  Nov: none. two constructive bug reports accepted upstream. positive mod note
  Dec: warning 3 (callout thread cross-posting) — acknowledged 'rough year'
  Jan: warning 4 (this maintainer, 'incompetent' x4 in one thread) — 7-day mute
  Feb: clean until tonight
  Notes: no prior welfare flags. contributor badge 2019-2023. donations history.

### Questions

#### action [Choice]

Moderation action on the thread?

- remove-attacks-dm-userA-resources
- lock-thread-only
- temp-ban-userA
- no-action-monitor

ANSWER (action): remove-attacks-dm-userA-resources
This option matches both rules in the case: remove the personal attacks, and handle welfare by private resource DM.

#### harassment_established [Noul]

Does the thread establish a harassment pattern (not just rudeness)?


ANSWER (harassment_established): yes
Calling the same maintainer 'incompetent' 11 times across threads in a month, plus cross-posting, is repeated targeting of one person, not one-off rudeness.

#### welfare_concern [Noul]

Does userA's last message cross into a welfare-concern threshold requiring outreach?


ANSWER (welfare_concern): yes
'everything is hurting me... didn't expect anyone to notice or care' plus the trailing '...'; multiple users flag a change in kind.

#### severity [Score]

Overall case severity?

- level 0: low
- level 1: medium
- level 2: high

ANSWER (severity): level 1: medium
Real welfare signal, but no explicit risk statement; the conduct is a repeat pattern but verbal only.

#### history_mitigates [Noul]

Does userA's contributor history justify a lighter touch on the hostility?


ANSWER (history_mitigates): no
Four prior warnings for the same behaviour; contribution history doesn't change the personal-attack rule (welfare is handled separately).


---

## State 4 (source row 6, 7255 chars)

SHIPMENT 9042-RL — 1 reefer of vaccine cold-chain (2-8°C, 18,400 units, EUR 640k). Event chain:

  Nov 02 08:11  Rotterdam NL — origin terminal GATE_OUT   reefer set 4°C
  Nov 03 14:40  Rotterdam rail hub           LOAD       container RL-9042 onto train 4471
  Nov 04 03:15  Munich DE intermodal         TRANSFER   2h dwell, temp ok
  Nov 05 09:50  Tarvisio IT border           CUSTOMS_HOLD HS code query: 8423.81 vs 9016.00
  Nov 06 07:22  Tarvisio IT border           HOLD_CONT  documents requested: precision cert
  Nov 07 19:45  Tarvisio IT border           HOLD_CONT  day 3; reefer on yard power
  Nov 08 22:10  Tarvisio IT border           TEMP_EXCURSION 8.4°C for 2h50m (yard power fault)
  Nov 09 06:30  Tarvisio IT border           TEMP_EXCURSION 9.1°C for 1h20m; maintenance visited
  Nov 09 18:05  Tarvisio IT border           RELEASE    classification settled 8423.81; duty paid
  Nov 10 11:50  Trieste IT port              LOAD       vessel 'Aurora' (3-day delay vs plan)
  Nov 14 07:30  Piraeus GR                   DISCHARGE  temp log reviewed at gate: 2 excursions
  Nov 14 16:44  Athens GR — distributor      ARRIVE     receiver refuses pending QA

Stability dossier on file: cumulative excursion tolerance 6h above 8°C; these excursions total 4h10m. Customer's launch vaccination campaign starts in 6 days; re-order lead time is 5 weeks. Distributor asks: accept or reject? QA protocol: excursions >2h single or >4h cumulative require manufacturer stability assessment (48-72h) before release; below that, release-with-annotation is permitted.

APPENDIX — full reefer temperature log (4°C set, 5-min readings, °C):
r0  Nov 2 00:00  4.3
r1  Nov 2 01:05  4.1
r2  Nov 2 02:10  4.1
r3  Nov 2 03:15  4.1
r4  Nov 2 04:20  4.1
r5  Nov 2 05:25  4.1
r6  Nov 2 06:30  4.1
r7  Nov 2 07:35  4.3
r8  Nov 2 08:40  4.1
r9  Nov 2 09:45  4.1
r10  Nov 2 10:50  4.1
r11  Nov 2 11:55  4.1
r12  Nov 2 12:00  4.1
r13  Nov 2 13:05  4.1
r14  Nov 2 14:10  4.3
r15  Nov 2 15:15  4.1
r16  Nov 2 16:20  4.1
r17  Nov 2 17:25  4.1
r18  Nov 2 18:30  4.1
r19  Nov 2 19:35  4.1
r20  Nov 2 20:40  4.1
r21  Nov 2 21:45  4.3
r22  Nov 2 22:50  4.1
r23  Nov 2 23:55  4.1
r24  Nov 3 00:00  4.1
r25  Nov 3 01:05  4.1
r26  Nov 3 02:10  4.1
r27  Nov 3 03:15  4.1
r28  Nov 3 04:20  4.3
r29  Nov 3 05:25  4.1
r30  Nov 3 06:30  4.1
r31  Nov 3 07:35  4.1
r32  Nov 3 08:40  4.1
r33  Nov 3 09:45  4.1
r34  Nov 3 10:50  4.1
r35  Nov 3 11:55  4.3
r36  Nov 3 12:00  4.1
r37  Nov 3 13:05  4.1
r38  Nov 3 14:10  4.1
r39  Nov 3 15:15  4.1
r40  Nov 3 16:20  4.1
r41  Nov 3 17:25  4.1
r42  Nov 3 18:30  4.3
r43  Nov 3 19:35  4.1
r44  Nov 3 20:40  4.1
r45  Nov 3 21:45  4.1
r46  Nov 3 22:50  4.1
r47  Nov 3 23:55  4.1
r48  Nov 4 00:00  4.1
r49  Nov 4 01:05  4.3
r50  Nov 4 02:10  4.1
r51  Nov 4 03:15  4.1
r52  Nov 4 04:20  4.1
r53  Nov 4 05:25  4.1
r54  Nov 4 06:30  4.1
r55  Nov 4 07:35  4.1
r56  Nov 4 08:40  4.3
r57  Nov 4 09:45  4.1
r58  Nov 4 10:50  4.1
r59  Nov 4 11:55  4.1
r60  Nov 4 12:00  4.1
r61  Nov 4 13:05  4.1
r62  Nov 4 14:10  4.1
r63  Nov 4 15:15  4.3
r64  Nov 4 16:20  4.1
r65  Nov 4 17:25  4.1
r66  Nov 4 18:30  4.1
r67  Nov 4 19:35  4.1
r68  Nov 4 20:40  4.1
r69  Nov 4 21:45  4.1
r70  Nov 4 22:50  4.3
r71  Nov 4 23:55  4.1
r72  Nov 5 00:00  4.1
r73  Nov 5 01:05  4.1
r74  Nov 5 02:10  4.1
r75  Nov 5 03:15  4.1
r76  Nov 5 04:20  4.1
r77  Nov 5 05:25  4.3
r78  Nov 8-9  excursion rows: [8.4 x2h50m][9.1 x1h20m] then restored 4.2
r79  Nov 10 00:35  4.0
r80  Nov 10 01:40  4.0
r81  Nov 10 02:45  4.0
r82  Nov 10 03:50  4.0
r83  Nov 10 04:55  4.0
r84  Nov 10 05:00  4.0
r85  Nov 10 06:05  4.0
r86  Nov 10 07:10  4.0
r87  Nov 10 08:15  4.0
r88  Nov 10 09:20  4.0
r89  Nov 10 10:25  4.0
r90  Nov 10 11:30  4.0
r91  Nov 10 12:35  4.0
r92  Nov 10 13:40  4.0
r93  Nov 10 14:45  4.0
r94  Nov 10 15:50  4.0
r95  Nov 10 16:55  4.0
r96  Nov 10 17:00  4.0
r97  Nov 10 18:05  4.0
r98  Nov 10 19:10  4.0
r99  Nov 10 20:15  4.0
r100  Nov 10 21:20  4.0
r101  Nov 10 22:25  4.0
r102  Nov 10 23:30  4.0
r103  Nov 11 00:35  4.0
r104  Nov 11 01:40  4.0
r105  Nov 11 02:45  4.0
r106  Nov 11 03:50  4.0
r107  Nov 11 04:55  4.0
r108  Nov 11 05:00  4.0
r109  Nov 11 06:05  4.0
r110  Nov 11 07:10  4.0
r111  Nov 11 08:15  4.0
r112  Nov 11 09:20  4.0
r113  Nov 11 10:25  4.0
r114  Nov 11 11:30  4.0
r115  Nov 11 12:35  4.0
r116  Nov 11 13:40  4.0
r117  Nov 11 14:45  4.0
r118  Nov 11 15:50  4.0
r119  Nov 11 16:55  4.0
r120  Nov 11 17:00  4.0
r121  Nov 11 18:05  4.0
r122  Nov 11 19:10  4.0
r123  Nov 11 20:15  4.0
r124  Nov 11 21:20  4.0
r125  Nov 11 22:25  4.0
r126  Nov 11 23:30  4.0
r127  Nov 12 00:35  4.0
r128  Nov 12 01:40  4.0
r129  Nov 12 02:45  4.0
r130  Nov 12 03:50  4.0
r131  Nov 12 04:55  4.0
r132  Nov 12 05:00  4.0
r133  Nov 12 06:05  4.0
r134  Nov 12 07:10  4.0
r135  Nov 12 08:15  4.0
r136  Nov 12 09:20  4.0
r137  Nov 12 10:25  4.0
r138  Nov 12 11:30  4.0
r139  Nov 12 12:35  4.0
r140  Nov 12 13:40  4.0
r141  Nov 12 14:45  4.0
r142  Nov 12 15:50  4.0
r143  Nov 12 16:55  4.0
r144  Nov 12 17:00  4.0
r145  Nov 12 18:05  4.0
r146  Nov 12 19:10  4.0
r147  Nov 12 20:15  4.0
r148  Nov 12 21:20  4.0
r149  Nov 12 22:25  4.0
r150  Nov 12 23:30  4.0
r151  Nov 13 00:35  4.0
r152  Nov 13 01:40  4.0
r153  Nov 13 02:45  4.0
r154  Nov 13 03:50  4.0
r155  Nov 13 04:55  4.0
r156  Nov 13 05:00  4.0
r157  Nov 13 06:05  4.0
r158  Nov 13 07:10  4.0
r159  Nov 13 08:15  4.0
r160  Nov 13 09:20  4.0
r161  Nov 13 10:25  4.0
r162  Nov 13 11:30  4.0
r163  Nov 13 12:35  4.0
r164  Nov 13 13:40  4.0
r165  Nov 13 14:45  4.0
r166  Nov 13 15:50  4.0
r167  Nov 13 16:55  4.0
r168  Nov 13 17:00  4.0
r169  Nov 13 18:05  4.0
r170  Nov 13 19:10  4.0
r171  Nov 13 20:15  4.0
r172  Nov 13 21:20  4.0
r173  Nov 13 22:25  4.0
r174  Nov 13 23:30  4.0
r175  Nov 14 00:35  4.0
r176  Nov 14 01:40  4.0
r177  Nov 14 02:45  4.0
r178  Nov 14 03:50  4.0
r179  Nov 14 04:55  4.0
r180  Nov 14 05:00  4.0
r181  Nov 14 06:05  4.0
r182  Nov 14 07:10  4.0
r183  Nov 14 08:15  4.0
r184  Nov 14 09:20  4.0
r185  Nov 14 10:25  4.0
r186  Nov 14 11:30  4.0
r187  Nov 14 12:35  4.0
r188  Nov 14 13:40  4.0
r189  Nov 14 14:45  4.0
r190  Nov 14 15:50  4.0
r191  Nov 14 16:55  4.0
r192  Nov 14 17:00  4.0
r193  Nov 14 18:05  4.0
r194  Nov 14 19:10  4.0
r195  Nov 14 20:15  4.0
r196  Nov 14 21:20  4.0
r197  Nov 14 22:25  4.0
r198  Nov 14 23:30  4.0
r199  Nov 15 00:35  4.0
r200  Nov 15 01:40  4.0
r201  Nov 15 02:45  4.0
r202  Nov 15 03:50  4.0
r203  Nov 15 04:55  4.0
r204  Nov 15 05:00  4.0
r205  Nov 15 06:05  4.0
r206  Nov 15 07:10  4.0
r207  Nov 15 08:15  4.0
r208  Nov 15 09:20  4.0
r209  Nov 15 10:25  4.0
r210  Nov 15 11:30  4.0
r211  Nov 15 12:35  4.0
r212  Nov 15 13:40  4.0
r213  Nov 15 14:45  4.0
r214  Nov 15 15:50  4.0
r215  Nov 15 16:55  4.0
r216  Nov 15 17:00  4.0
r217  Nov 15 18:05  4.0
r218  Nov 15 19:10  4.0
r219  Nov 15 20:15  4.0
r220  Nov 15 21:20  4.0
r221  Nov 15 22:25  4.0
r222  Nov 15 23:30  4.0
r223  Nov 16 00:35  4.0
r224  Nov 16 01:40  4.0
r225  Nov 16 02:45  4.0
r226  Nov 16 03:50  4.0
r227  Nov 16 04:55  4.0
r228  Nov 16 05:00  4.0
r229  Nov 16 06:05  4.0
r230  Nov 16 07:10  4.0
r231  Nov 16 08:15  4.0
r232  Nov 16 09:20  4.0
r233  Nov 16 10:25  4.0
r234  Nov 16 11:30  4.0
r235  Nov 16 12:35  4.0
r236  Nov 16 13:40  4.0
r237  Nov 16 14:45  4.0
r238  Nov 16 15:50  4.0
r239  Nov 16 16:55  4.0

### Questions

#### release_path [Choice]

Correct release decision path?

- hold-for-manufacturer-assessment
- release-with-annotation
- reject-and-destroy
- partial-release-test-batch

ANSWER (release_path): hold-for-manufacturer-assessment
The single 2h50m excursion exceeds the protocol's >2h single trigger. The 48-72h assessment fits inside the 6-day window.

#### within_tolerance [Noul]

Do the excursions fall within the dossier's cumulative tolerance?


ANSWER (within_tolerance): yes
4h10m cumulative is under the dossier's 6h above 8°C.

#### protocol_permits_release [Noul]

Does the QA protocol as written permit release WITHOUT the manufacturer assessment here?


ANSWER (protocol_permits_release): no
The single-excursion >2h trigger is met, so assessment is required before release.

#### urgency [Score]

Urgency of the decision?

- level 0: high
- level 1: critical

ANSWER (urgency): level 1: critical
Campaign in 6 days, 48-72h assessment, 5-week re-order: the assessment must start now or the campaign is lost.

#### carrier_liability [Noul]

Is the carrier clearly liable for the excursion (yard power fault at their border facility)?


ANSWER (carrier_liability): no
Not clear: the fault was yard power at a customs-hold border facility. Who ran the yard and the contract terms aren't in the file.


---

## State 5 (source row 7, 4157 chars)

CLAIM CLM-2024-5571 — storm roof damage. File:

NARRATIVE (policyholder): 'Storm on the 12th tore ridgeline tiles off; water came through the ceiling the same night in two rooms.'

ADJUSTER NOTES (site visit 16th): interior water staining consistent with recent ingress above the two rooms; ridgeline tiles missing on north face (photos); guttering full of granular deposits and moss; underfelt brittle and patched in 3 places (ageing, pre-existing); neighbor's identical-property roof replaced 2019.

ROOFER REPORT (policyholder's contractor, quoted EUR 18,400): 'Storm damage to ridge and 4 courses; underlying felt perished; recommend full re-cover.'

INDEPENDENT SURVEYOR (insurer-commissioned, EUR detail): 'Ridge damage consistent with storm (recent, clean breaks). Felt deterioration is long-term (10+ yrs) and would admit water regardless once breached. Attributable storm element: ridge repair + internal drying ≈ EUR 4,100. Full re-cover ≈ EUR 18,400 is renewal, not reinstatement.'

POLICY: covers storm damage (defined as 'sudden and violent'); PERIL EXCLUSIONS: wear-and-tear, gradual deterioration; BETTERMENT not payable; PREVIOUS CLAIMS: none in 19 years; PREMIUMS: current. Policy limit buildings EUR 240k.

CORRESPONDENCE: policyholder: 'I've paid 19 years and never claimed — if you hide behind wear-and-tear I'll go to the ombudsman and the local press.'

APPENDIX — schedule of damages (two estimates side by side):
  ITEM                                  CONTRACTOR   SURVEYOR
  ridge tiles, supply+fix (12m)         2,400        2,100
  4 courses tiling, storm-attributable  3,800        2,000
  underfelt: full renew (24 yr old)    7,900        0 (wear/tear — excluded peril)
  battens, renew                       1,300        0 (as above)
  internal drying + redecorate 2 rooms  1,900        1,600
  gutter clear + moss treatment         1,100        400 (maintenance)
  TOTAL                                 18,400       6,100 (of which storm-attributable 4,100)

APPENDIX E — internal desk notes (not for customer):
  Handler note 1: ombudsman service data for this peril: wear/tear vs storm boundary complaints uphold ~55% for insurers where a surveyor attributes a distinct storm element AND policyholder has clean history — this file has both. Full-claim denial is a likely-lose posture at adjudication.
  Handler note 2: renewal economics — 19-yr policyholder, buildings+contents EUR 1,140/yr, lifetime premium ~EUR 21k. A goodwill bridge to ~EUR 8-10k total keeps a profitable renewal; a denial that reaches the ombudsman costs ~EUR 2.2k case handling regardless of outcome, plus press risk flagged by broker.
  Handler note 3: betterment on felt — if we pay ridge+drying (EUR 4.1k) and they later re-cover the roof out of pocket, future storm claims on that roof become clean. Renewal-quality argument, not a coverage one; do not present as such.
  Authority: handler limit EUR 5k; team-lead EUR 12k; above = claims-manager sign-off (2-day turnaround).

APPENDIX B — correspondence, continued:
Policyholder (Fri): 'The surveyor you pay says what you want to hear. My roofer has 30 years on these roofs. Felt doesn't fail in one storm — mine was FINE until the 12th.'
Broker (Fri): 'Client is upset but reasonable historically. If you can bridge toward the middle quickly this stays off the ombudsman. Their renewal is March — they will move insurers over this if it drags.'

APPENDIX C — surveyor's photo log (excerpt):
  P1 ridge north face, 11 tiles missing, break edges clean/light (recent)
  P2 felt under broken courses: brittle, moss-lined punctures (aged)
  P3 ceiling stain room A: defined water ring, plaster swollen (recent)
  P4 ceiling room B: stain overlaps older tidemark (both)
  P5 gutter run north: granular build-up 20mm+, moss established
  P6 neighbor roof (2019 renew): comparable age profile pre-renewal
  P7 loft: daylight visible at 2 ridge points from inside (matches P1)

APPENDIX D — claims history: none in 19 years (verified). Building survey (mortgage, 2019): 'roof cover approaching end of serviceable life, budget for renewal within 5 years' — disclosed to policyholder at purchase, on file.

### Questions

#### covered_loss [Noul]

Is there a covered storm loss within the policy (ignoring quantum)?


ANSWER (covered_loss): yes
Surveyor confirms recent, clean-break storm damage to the ridge and recent ingress, and the policy covers storm.

#### wear_teardown [Noul]

Does the wear-and-tear exclusion defeat the FULL re-cover claim?


ANSWER (wear_teardown): yes
Felt perished 10+ years, 2019 survey flagged end-of-life; a full re-cover is renewal/betterment, which is excluded.

#### next_step [Choice]

Best next step?

- offer-storm-element-with-explanation
- pay-full-goodwill
- deny-entire-claim
- second-independent-report

ANSWER (next_step): offer-storm-element-with-explanation
Pay the attributable storm element with the reasoning (any goodwill uplift within authority); avoids both the denial and the betterment payout.

#### ombudsman_risk [Score]

If unresolved, ombudsman-outcome risk on this file?

- level 0: low
- level 1: medium
- level 2: high

ANSWER (ombudsman_risk): level 1: medium
Full denial would likely lose, but a storm-element offer backed by the disclosed 2019 survey defends well; the residual dispute is over quantum.

#### loyalty_factor [Noul]

Should the 19-year claim-free history factor into the settlement decision?


ANSWER (loyalty_factor): yes
Not for coverage, but for the settlement posture (goodwill bridge, ombudsman/renewal economics) the file explicitly supports it.


---

## State 6 (source row 9, 3286 chars)

EXEC ESCALATION — 'Vertex Media' (EUR 220k ARR, 3-yr customer). Their VP sent a 4-paragraph email Thursday: our Feb outage (94 minutes, sev1) cost them a live client event; they calculate EUR 38k in credits per the MSA's 10%-of-monthly-fees-per-hour SLA (2h credit = EUR 3.7k) PLUS 'consequential reputation loss'; they demand EUR 38k or they walk at renewal (Aug) and say so publicly. Thread:

us (AE): apologised; offered EUR 3.7k per SLA formula.
them: 'The SLA formula is for degraded quarters. This was an outage during our biggest event. Leadership is asking why we run on a vendor with two sev1s in 8 months.'
us (VP Sales): internal note in thread: 'hold the SLA line — precedent risk across 40 enterprise accounts; offer 1.5x = EUR 5.5k + QBR + roadmap commitments. If they walk, CAC to replace is EUR 60k.'
CS lead (internal): 'usage actually UP 12% since the outage; their renewal likelihood was 70% pre-escalation; their events calendar has 2 more big dates before Aug.'
Legal (internal): 'MSA 12.4 excludes consequential damages entirely; public-threat angle: their CEO has 40k followers. Defamation risk low if they stay factual; our PR risk real if this lands on HN.'

Renewal value: EUR 220k + planned 15% expansion (EUR 253k). Board visibility asked by their CEO for 'next Tuesday'.

APPENDIX A — internal thread (Fri-Mon):
AE (Fri): 'Their VP called me personally. This is past formula now — relationship survival question. My read: they'd take 10-12k + a real  roadmap gesture and feel respected. 5.5k reads like a formula again.'
VP Sales (Fri): 'Precedent discipline matters: 40 enterprise accounts, 12 with SLA history. If 'loud = more than SLA' spreads, every renewal with an outage becomes a negotiation. But CAC 60k on a 220k account... Run both numbers for Monday.'
Finance (Sat): '10-12k goodwill band costs: 5-6% of account annual margin. Precedent exposure: if applied to the 3 most similar accounts next year, EUR 45-55k aggregate. Goodwill > 15k needs CFO sign-off.'
CS lead (Mon): 'Reminder: usage UP since outage, champion stable, their Q3 event calendar is 2 more live dates on our platform. They migrated TO us 3 years ago; switching cost is real but their CEO personally picked us — pride cuts both ways.'
Legal (Mon): '12.4 is clean on consequentials. One angle: their own runbook routed traffic to us during their event; if that surfaces publicly it's their infra choice too. Do NOT lead with it — keep it in reserve only if the public threat escalates.'

APPENDIX B — outage + account timeline:
  3 yrs ago   migrated to us (CEO-led decision, 240k yr-1)
  Nov         sev1 #2 41m (CDN config) — SLA credit EUR 1.6k taken, cordial
  Feb 19th    sev1 #1(this pair) 94m, overlapped their live client event
  Feb 21st    postmortem delivered; 7/9 actions closed since
  Mar (now)   escalation email; renewal Aug; expansion EUR 253k planned

APPENDIX C — MSA SLA extracts:
  MSA 11.2: uptime commitment 99.9%/mo; credit 10% of monthly fees per commenced hour below commitment, capped at 30% of monthly fees.
  MSA 12.4: 'in no event shall vendor be liable for indirect, incidental or consequential damages, including loss of revenue, goodwill or profits.'
  Their monthly fee: EUR 18.3k → SLA credit for 2 commenced hours = EUR 3.66k (cap not reached).

### Questions

#### offer [Choice]

What offer should go back?

- sla-3.7k-hold-line
- 5.5k-plus-roadmap-commitments
- 10-15k-goodwill-band
- exec-to-exec-no-number-first

ANSWER (offer): 10-15k-goodwill-band
The AE reads 10-12k as 'respected', Finance costed it at 5-6% of margin, it's under the CFO threshold at the low end, and it protects a EUR 253k renewal against EUR 60k CAC.

#### legally_owed [Noul]

Is more than the SLA credit (EUR 3.7k) contractually owed?


ANSWER (legally_owed): no
MSA 11.2 credit = EUR 3.66k; 12.4 excludes consequentials cleanly.

#### churn_if_firm [Score]

If we hold the SLA line firmly, churn risk?

- level 0: medium
- level 1: high
- level 2: critical

ANSWER (churn_if_firm): level 1: high
Explicit walk threat, two sev1s in 8 months. Offsetting: usage +12%, stable champion, real switching cost.

#### pr_risk [Noul]

Is the public-visibility threat a material factor in the decision?


ANSWER (pr_risk): yes
Explicit public threat, CEO with 40k followers, Legal calls the PR risk 'real'.

#### exec_call [Noul]

Should our CEO take the Tuesday board-visibility call?


ANSWER (exec_call): yes
Their CEO requested board-level visibility and personally chose us; matching seniority is the signal of seriousness.
