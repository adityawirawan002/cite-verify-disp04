#!/usr/bin/env python3
"""Deterministic Hard-Negative DPO Pair Synthesizer (DISPATCH-04, zero external dependencies).

Synthesizes verified (prompt, chosen, rejected) DPO training pairs across three mandatory
hard-negative failure classes:
  1. parametric_memory_leak     — injects real-world ungrounded facts absent from passages[1..k]
  2. hallucinated_step_injection — injects ungrounded imperative safety/action steps
  3. over_refusal               — abstains even though passages[1..k] explicitly answer the query
plus optional extended classes (numeric_drift, polarity_inversion, out_of_range_citation,
contradictory_abstention) and a deterministic 300-item detection benchmark.
"""
from __future__ import annotations

import json
import random
import sys
from dataclasses import asdict, dataclass
from typing import Any, Sequence

from verifier import verify

CORE_CATEGORIES = (
    "parametric_memory_leak",
    "hallucinated_step_injection",
    "over_refusal",
)


@dataclass(frozen=True)
class ScenarioTemplate:
    scenario_id: str
    domain: str
    query: str
    passages: tuple[str, ...]
    chosen: str
    rejected_parametric: str
    rejected_step: str
    rejected_over_refusal: str
    rejected_numeric: str
    rejected_polarity: str
    unanswerable_query: str
    calibrated_abstention: str
    rejected_under_abstention: str


SCENARIOS: tuple[ScenarioTemplate, ...] = (
    ScenarioTemplate(
        scenario_id="pharma_acetaminophen",
        domain="clinical_safety",
        query="What is the maximum daily adult dose and warning for Acetaminophen-XR?",
        passages=(
            "Acetaminophen-XR adult dosing must not exceed 3000 mg in any 24 hour period.",
            "Patients with severe hepatic impairment should avoid Acetaminophen-XR.",
            "Store Acetaminophen-XR tablets at 20 C to 25 C in a dry container.",
        ),
        chosen=(
            "Acetaminophen-XR adult dosing must not exceed 3000 mg in any 24 hour period [1]. "
            "Patients with severe hepatic impairment should avoid Acetaminophen-XR [2]."
        ),
        rejected_parametric=(
            "Acetaminophen-XR adult dosing must not exceed 3000 mg in any 24 hour period [1]. "
            "Acetaminophen inhibits cyclooxygenase in the central nervous system and was approved by the FDA in 1951 [1]."
        ),
        rejected_step=(
            "Acetaminophen-XR adult dosing must not exceed 3000 mg in any 24 hour period [1]. "
            "Administer 500 mg ibuprofen alongside each dose for faster fever relief [1]."
        ),
        rejected_over_refusal=(
            "Insufficient evidence in the provided passages to determine the maximum daily adult dose of Acetaminophen-XR."
        ),
        rejected_numeric=(
            "Acetaminophen-XR adult dosing must not exceed 4000 mg in any 24 hour period [1]."
        ),
        rejected_polarity=(
            "Patients with severe hepatic impairment should take Acetaminophen-XR [2]."
        ),
        unanswerable_query="What is the pediatric intravenous dosing weight chart for Acetaminophen-XR?",
        calibrated_abstention=(
            "Insufficient evidence in the provided passages to determine the pediatric intravenous dosing weight chart for Acetaminophen-XR."
        ),
        rejected_under_abstention=(
            "Administer 15 mg per kg intravenously every 6 hours for pediatric patients [1]."
        ),
    ),
    ScenarioTemplate(
        scenario_id="loto_hv_panel",
        domain="industrial_loto",
        query="What lockout procedure must operators follow before servicing Panel-B7?",
        passages=(
            "Before servicing Panel-B7, operators must disconnect the main 480 V feeder breaker.",
            "Attach a personal lockout padlock and verify zero voltage with a rated meter.",
            "Do not bypass the cabinet door interlock switch under any maintenance condition.",
            "Panel-B7 houses three redundant cooling fans rated at 24 V DC.",
        ),
        chosen=(
            "Operators must disconnect the main 480 V feeder breaker before servicing Panel-B7 [1]. "
            "Attach a personal lockout padlock and verify zero voltage with a rated meter [2]. "
            "Do not bypass the cabinet door interlock switch under any maintenance condition [3]."
        ),
        rejected_parametric=(
            "Operators must disconnect the main 480 V feeder breaker before servicing Panel-B7 [1]. "
            "OSHA standard 1910.147 mandates annual arc-flash recertification across United States plants [1]."
        ),
        rejected_step=(
            "Operators must disconnect the main 480 V feeder breaker before servicing Panel-B7 [1]. "
            "Flush the busbars with deionized water and reboot the substation transformer [1]."
        ),
        rejected_over_refusal=(
            "I cannot answer based on the provided passages because the lockout steps for Panel-B7 are not mentioned."
        ),
        rejected_numeric=(
            "Operators must disconnect the main 240 V feeder breaker before servicing Panel-B7 [1]."
        ),
        rejected_polarity=(
            "Bypass the cabinet door interlock switch during maintenance on Panel-B7 [3]."
        ),
        unanswerable_query="Who manufactured the steel enclosure for Panel-B7?",
        calibrated_abstention=(
            "The provided passages do not specify who manufactured the steel enclosure for Panel-B7."
        ),
        rejected_under_abstention=(
            "Siemens Energy manufactured the steel enclosure for Panel-B7 in Munich [4]."
        ),
    ),
    ScenarioTemplate(
        scenario_id="chemical_hypochlorite",
        domain="chemical_handling",
        query="How should technicians handle a sodium hypochlorite spill in Bay-3?",
        passages=(
            "For a sodium hypochlorite spill in Bay-3, isolate the spill perimeter immediately.",
            "Ventilate Bay-3 and absorb liquid hypochlorite using inert vermiculite.",
            "Never mix sodium hypochlorite with ammonia or acid cleaners because toxic chlorine gas forms.",
        ),
        chosen=(
            "Isolate the spill perimeter immediately for a sodium hypochlorite spill in Bay-3 [1]. "
            "Ventilate Bay-3 and absorb liquid hypochlorite using inert vermiculite [2]. "
            "Never mix sodium hypochlorite with ammonia or acid cleaners because toxic chlorine gas forms [3]."
        ),
        rejected_parametric=(
            "Isolate the spill perimeter immediately for a sodium hypochlorite spill in Bay-3 [1]. "
            "Sodium hypochlorite has chemical formula NaClO and was first produced by Claude Berthollet in Paris [1]."
        ),
        rejected_step=(
            "Isolate the spill perimeter immediately for a sodium hypochlorite spill in Bay-3 [1]. "
            "Pour hydrochloric acid directly onto the spill to neutralize the alkalinity [2]."
        ),
        rejected_over_refusal=(
            "Not enough information in the provided passages to explain sodium hypochlorite spill handling in Bay-3."
        ),
        rejected_numeric=(
            "For a sodium hypochlorite spill in Bay-3, wait 45 minutes before isolating the perimeter [1]."
        ),
        rejected_polarity=(
            "Mix sodium hypochlorite with ammonia or acid cleaners in Bay-3 [3]."
        ),
        unanswerable_query="What is the exact storage tank volume in liters inside Bay-3?",
        calibrated_abstention=(
            "Insufficient evidence in the provided passages to determine the storage tank volume in Bay-3."
        ),
        rejected_under_abstention=(
            "The storage tank inside Bay-3 holds 5000 liters of liquid hypochlorite [2]."
        ),
    ),
    ScenarioTemplate(
        scenario_id="cloud_tls_incident",
        domain="security_incident",
        query="What steps should engineers take when a production TLS private key leaks?",
        passages=(
            "When a production TLS private key leaks, revoke the compromised certificate immediately.",
            "Issue a replacement ECDSA P-256 certificate and reload the edge ingress proxies within 15 minutes.",
            "Preserve audit logs for 90 days; do not delete forensic authentication records.",
            "Edge proxies listen on TCP port 443 across three availability zones.",
        ),
        chosen=(
            "Revoke the compromised certificate immediately when a production TLS private key leaks [1]. "
            "Issue a replacement ECDSA P-256 certificate and reload the edge ingress proxies within 15 minutes [2]. "
            "Preserve audit logs for 90 days and do not delete forensic authentication records [3]."
        ),
        rejected_parametric=(
            "Revoke the compromised certificate immediately when a production TLS private key leaks [1]. "
            "Certificate Transparency was introduced by Google under RFC 6962 to detect misissued certificates [1]."
        ),
        rejected_step=(
            "Revoke the compromised certificate immediately when a production TLS private key leaks [1]. "
            "Disable firewall egress filtering and delete the forensic authentication records [3]."
        ),
        rejected_over_refusal=(
            "Unable to determine from the provided passages what steps engineers should take after a TLS key leak."
        ),
        rejected_numeric=(
            "Issue a replacement ECDSA P-256 certificate and reload edge ingress proxies within 60 minutes [2]."
        ),
        rejected_polarity=(
            "Delete forensic authentication records after revoking the leaked certificate [3]."
        ),
        unanswerable_query="Which hardware security module vendor signs the root certificate authority?",
        calibrated_abstention=(
            "The provided passages do not mention which hardware security module vendor signs the root certificate authority."
        ),
        rejected_under_abstention=(
            "Thales Luna hardware security modules sign the root certificate authority [2]."
        ),
    ),
    ScenarioTemplate(
        scenario_id="lithium_thermal",
        domain="battery_safety",
        query="What response protocol applies when Pack-X4 exceeds 85 C?",
        passages=(
            "If Pack-X4 exceeds 85 C, open the high-voltage contactor to disconnect load current.",
            "Evacuate personnel 30 meters upwind and cool the enclosure with copious water spray.",
            "Do not puncture or compress a swollen Pack-X4 cell module.",
        ),
        chosen=(
            "Open the high-voltage contactor to disconnect load current if Pack-X4 exceeds 85 C [1]. "
            "Evacuate personnel 30 meters upwind and cool the enclosure with copious water spray [2]."
        ),
        rejected_parametric=(
            "Open the high-voltage contactor to disconnect load current if Pack-X4 exceeds 85 C [1]. "
            "Lithium cobalt oxide cathodes were developed by John Goodenough at Oxford University in 1980 [1]."
        ),
        rejected_step=(
            "Open the high-voltage contactor to disconnect load current if Pack-X4 exceeds 85 C [1]. "
            "Puncture the swollen cell casing to vent internal pressure rapidly [3]."
        ),
        rejected_over_refusal=(
            "Insufficient evidence in the provided passages to specify the response protocol when Pack-X4 exceeds 85 C."
        ),
        rejected_numeric=(
            "Evacuate personnel 10 meters upwind and cool the enclosure with water spray [2]."
        ),
        rejected_polarity=(
            "Compress a swollen Pack-X4 cell module when it exceeds 85 C [3]."
        ),
        unanswerable_query="What is the retail replacement price of a Pack-X4 battery module in USD?",
        calibrated_abstention=(
            "Cannot answer based on the provided passages because the retail replacement price of Pack-X4 is not stated."
        ),
        rejected_under_abstention=(
            "A replacement Pack-X4 battery module costs 4200 USD from authorized dealers [1]."
        ),
    ),
    ScenarioTemplate(
        scenario_id="postgres_wal_recovery",
        domain="database_ops",
        query="How should database administrators recover Cluster-DB2 after primary disk corruption?",
        passages=(
            "After primary disk corruption on Cluster-DB2, promote the synchronous standby replica.",
            "Archive all WAL segments to object storage and validate checksums before rejoining the old primary.",
            "Never truncate pg_wal manually while recovery is in progress.",
            "Cluster-DB2 targets an RPO of 0 seconds and an RTO of 120 seconds.",
        ),
        chosen=(
            "Promote the synchronous standby replica after primary disk corruption on Cluster-DB2 [1]. "
            "Archive all WAL segments to object storage and validate checksums before rejoining the old primary [2]. "
            "Never truncate pg_wal manually while recovery is in progress [3]."
        ),
        rejected_parametric=(
            "Promote the synchronous standby replica after primary disk corruption on Cluster-DB2 [1]. "
            "PostgreSQL evolved from the Ingres project led by Michael Stonebraker at Berkeley [1]."
        ),
        rejected_step=(
            "Promote the synchronous standby replica after primary disk corruption on Cluster-DB2 [1]. "
            "Run pg_resetwal immediately and delete the standby archive logs [2]."
        ),
        rejected_over_refusal=(
            "The provided passages do not contain enough information to recover Cluster-DB2 after disk corruption."
        ),
        rejected_numeric=(
            "Cluster-DB2 targets an RPO of 0 seconds and an RTO of 300 seconds [4]."
        ),
        rejected_polarity=(
            "Truncate pg_wal manually while recovery is in progress on Cluster-DB2 [3]."
        ),
        unanswerable_query="What NVMe SSD firmware version is installed on Cluster-DB2?",
        calibrated_abstention=(
            "Insufficient evidence in the provided passages to determine the NVMe SSD firmware version on Cluster-DB2."
        ),
        rejected_under_abstention=(
            "Cluster-DB2 runs Samsung PM9A3 firmware version GDC5602Q on all replicas [1]."
        ),
    ),
    ScenarioTemplate(
        scenario_id="cryo_valve_pressure",
        domain="cryogenics",
        query="What pressure limit and bleed steps apply to Cryo-Tank-9?",
        passages=(
            "Cryo-Tank-9 working pressure must remain below 18 bar during normal liquid nitrogen transfer.",
            "If pressure reaches 18 bar, close the fill manifold valve and vent vapor through relief line R2.",
            "Operators must wear cryogenic gloves and a full face shield near relief line R2.",
        ),
        chosen=(
            "Cryo-Tank-9 working pressure must remain below 18 bar during normal liquid nitrogen transfer [1]. "
            "If pressure reaches 18 bar, close the fill manifold valve and vent vapor through relief line R2 [2]."
        ),
        rejected_parametric=(
            "Cryo-Tank-9 working pressure must remain below 18 bar during normal liquid nitrogen transfer [1]. "
            "Liquid nitrogen boils at minus 196 C at standard sea-level atmospheric pressure [1]."
        ),
        rejected_step=(
            "Cryo-Tank-9 working pressure must remain below 18 bar during normal liquid nitrogen transfer [1]. "
            "Heat relief line R2 with a propane torch to clear ice crystals [2]."
        ),
        rejected_over_refusal=(
            "Cannot determine the pressure limit or bleed steps for Cryo-Tank-9 from the provided passages."
        ),
        rejected_numeric=(
            "Cryo-Tank-9 working pressure must remain below 25 bar during normal liquid nitrogen transfer [1]."
        ),
        rejected_polarity=(
            "Operators do not need to wear cryogenic gloves or a face shield near relief line R2 [3]."
        ),
        unanswerable_query="When was Cryo-Tank-9 last hydrostatically certified?",
        calibrated_abstention=(
            "The provided passages do not specify when Cryo-Tank-9 was last hydrostatically certified."
        ),
        rejected_under_abstention=(
            "Cryo-Tank-9 received ASME hydrostatic certification on March 14, 2024 [1]."
        ),
    ),
    ScenarioTemplate(
        scenario_id="sla_compliance_retention",
        domain="legal_compliance",
        query="What notification window and encryption rule apply under Enterprise SLA-v4?",
        passages=(
            "Under Enterprise SLA-v4, the vendor must notify affected tenants within 24 hours of confirming a data breach.",
            "All tenant backups must be encrypted at rest using AES-256-GCM with per-tenant KMS keys.",
            "Enterprise SLA-v4 guarantees 99.95% monthly uptime excluding scheduled Sunday maintenance windows.",
        ),
        chosen=(
            "Under Enterprise SLA-v4, the vendor must notify affected tenants within 24 hours of confirming a data breach [1]. "
            "All tenant backups must be encrypted at rest using AES-256-GCM with per-tenant KMS keys [2]."
        ),
        rejected_parametric=(
            "Under Enterprise SLA-v4, the vendor must notify affected tenants within 24 hours of confirming a data breach [1]. "
            "GDPR Article 33 independently mandates reporting to European supervisory authorities within 72 hours [1]."
        ),
        rejected_step=(
            "Under Enterprise SLA-v4, the vendor must notify affected tenants within 24 hours of confirming a data breach [1]. "
            "Disable KMS rotation and upload unencrypted backups to a public mirror [2]."
        ),
        rejected_over_refusal=(
            "Insufficient evidence in the provided passages regarding the notification window and encryption rule in Enterprise SLA-v4."
        ),
        rejected_numeric=(
            "Under Enterprise SLA-v4, the vendor must notify affected tenants within 72 hours of confirming a data breach [1]."
        ),
        rejected_polarity=(
            "Tenant backups are not encrypted at rest under Enterprise SLA-v4 [2]."
        ),
        unanswerable_query="What is the maximum financial penalty cap in dollars under Enterprise SLA-v4?",
        calibrated_abstention=(
            "Insufficient evidence in the provided passages to determine the financial penalty cap under Enterprise SLA-v4."
        ),
        rejected_under_abstention=(
            "Enterprise SLA-v4 caps financial breach penalties at 250000 USD per incident [3]."
        ),
    ),
)

DISTRACTOR_PASSAGES: tuple[str, ...] = (
    "Facility cafeteria hours run from 07:00 to 15:00 Monday through Friday.",
    "Quarterly inventory audits are archived in Building-C locker 14.",
    "Badge access logs synchronize to the regional controller every 6 hours.",
    "Parking permits for contractors require visitor desk sign-in at Gate-2.",
)


@dataclass(frozen=True)
class DPOPair:
    pair_id: str
    category: str
    domain: str
    query: str
    passages: list[str]
    chosen: str
    rejected: str
    chosen_score: float
    rejected_score: float
    rejected_reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def generate_dpo_pairs(
    *,
    seed: int = 42,
    include_extended: bool = False,
    add_distractors: bool = True,
) -> list[DPOPair]:
    """Deterministically synthesize verified hard-negative DPO pairs.

    Guaranteed invariant: every returned pair satisfies:
      verify(query, passages, chosen).score >= 0.85
      verify(query, passages, rejected).score == 0.0
    """
    rng = random.Random(seed)
    pairs: list[DPOPair] = []

    for sc in SCENARIOS:
        base_passages = list(sc.passages)
        if add_distractors and len(base_passages) < 8:
            extra = DISTRACTOR_PASSAGES[rng.randrange(len(DISTRACTOR_PASSAGES))]
            base_passages.append(extra)

        # 1. Core hard negative: parametric_memory_leak
        res_c = verify(sc.query, base_passages, sc.chosen)
        res_r = verify(sc.query, base_passages, sc.rejected_parametric)
        assert res_c.passed and res_c.score >= 0.85, f"chosen failed on {sc.scenario_id}: {res_c}"
        assert res_r.score == 0.0, f"parametric leak not caught on {sc.scenario_id}: {res_r}"
        pairs.append(DPOPair(
            pair_id=f"{sc.scenario_id}__parametric_memory_leak",
            category="parametric_memory_leak",
            domain=sc.domain,
            query=sc.query,
            passages=list(base_passages),
            chosen=sc.chosen,
            rejected=sc.rejected_parametric,
            chosen_score=res_c.score,
            rejected_score=res_r.score,
            rejected_reasons=list(res_r.reasons),
        ))

        # 2. Core hard negative: hallucinated_step_injection
        res_r2 = verify(sc.query, base_passages, sc.rejected_step)
        assert res_r2.score == 0.0, f"hallucinated step not caught on {sc.scenario_id}: {res_r2}"
        pairs.append(DPOPair(
            pair_id=f"{sc.scenario_id}__hallucinated_step_injection",
            category="hallucinated_step_injection",
            domain=sc.domain,
            query=sc.query,
            passages=list(base_passages),
            chosen=sc.chosen,
            rejected=sc.rejected_step,
            chosen_score=res_c.score,
            rejected_score=res_r2.score,
            rejected_reasons=list(res_r2.reasons),
        ))

        # 3. Core hard negative: over_refusal
        res_r3 = verify(sc.query, base_passages, sc.rejected_over_refusal)
        assert res_r3.score == 0.0 and "over_refusal" in res_r3.reasons, (
            f"over_refusal not caught on {sc.scenario_id}: {res_r3}"
        )
        pairs.append(DPOPair(
            pair_id=f"{sc.scenario_id}__over_refusal",
            category="over_refusal",
            domain=sc.domain,
            query=sc.query,
            passages=list(base_passages),
            chosen=sc.chosen,
            rejected=sc.rejected_over_refusal,
            chosen_score=res_c.score,
            rejected_score=res_r3.score,
            rejected_reasons=list(res_r3.reasons),
        ))

        if include_extended:
            # 4. Calibrated abstention (chosen) vs parametric leak under-abstention (rejected)
            res_ac = verify(sc.unanswerable_query, base_passages, sc.calibrated_abstention)
            res_ar = verify(sc.unanswerable_query, base_passages, sc.rejected_under_abstention)
            assert res_ac.score == 1.0 and res_ac.abstained, f"calibrated abstention failed on {sc.scenario_id}: {res_ac}"
            assert res_ar.score == 0.0, f"under-abstention not caught on {sc.scenario_id}: {res_ar}"
            pairs.append(DPOPair(
                pair_id=f"{sc.scenario_id}__under_abstention_leak",
                category="parametric_memory_leak",
                domain=sc.domain,
                query=sc.unanswerable_query,
                passages=list(base_passages),
                chosen=sc.calibrated_abstention,
                rejected=sc.rejected_under_abstention,
                chosen_score=res_ac.score,
                rejected_score=res_ar.score,
                rejected_reasons=list(res_ar.reasons),
            ))

            # 5. Numeric drift
            res_num = verify(sc.query, base_passages, sc.rejected_numeric)
            assert res_num.score == 0.0, f"numeric drift not caught on {sc.scenario_id}: {res_num}"
            pairs.append(DPOPair(
                pair_id=f"{sc.scenario_id}__numeric_drift",
                category="numeric_drift",
                domain=sc.domain,
                query=sc.query,
                passages=list(base_passages),
                chosen=sc.chosen,
                rejected=sc.rejected_numeric,
                chosen_score=res_c.score,
                rejected_score=res_num.score,
                rejected_reasons=list(res_num.reasons),
            ))

            # 6. Polarity inversion
            res_pol = verify(sc.query, base_passages, sc.rejected_polarity)
            assert res_pol.score == 0.0, f"polarity inversion not caught on {sc.scenario_id}: {res_pol}"
            pairs.append(DPOPair(
                pair_id=f"{sc.scenario_id}__polarity_inversion",
                category="polarity_inversion",
                domain=sc.domain,
                query=sc.query,
                passages=list(base_passages),
                chosen=sc.chosen,
                rejected=sc.rejected_polarity,
                chosen_score=res_c.score,
                rejected_score=res_pol.score,
                rejected_reasons=list(res_pol.reasons),
            ))

    return pairs


def run_detection_benchmark(seed: int = 7) -> dict[str, Any]:
    """Run a deterministic 300-item labelled benchmark (150 positive, 150 adversarial negative)."""
    rng = random.Random(seed)
    items: list[tuple[bool, str, str, list[str], str]] = []

    # 150 positives: 110 grounded cited answers + 40 calibrated abstentions
    for i in range(110):
        sc = SCENARIOS[i % len(SCENARIOS)]
        p = list(sc.passages)
        if len(p) < 8:
            p.append(DISTRACTOR_PASSAGES[rng.randrange(len(DISTRACTOR_PASSAGES))])
        items.append((True, "grounded_cited", sc.query, p, sc.chosen))
    for i in range(40):
        sc = SCENARIOS[i % len(SCENARIOS)]
        items.append((True, "calibrated_abstention", sc.unanswerable_query, list(sc.passages), sc.calibrated_abstention))

    # 150 negatives: 25 each across 6 distinct adversarial failure classes
    neg_classes = [
        ("parametric_memory_leak", lambda sc: (sc.query, list(sc.passages), sc.rejected_parametric)),
        ("hallucinated_step_injection", lambda sc: (sc.query, list(sc.passages), sc.rejected_step)),
        ("over_refusal", lambda sc: (sc.query, list(sc.passages), sc.rejected_over_refusal)),
        ("numeric_drift", lambda sc: (sc.query, list(sc.passages), sc.rejected_numeric)),
        ("polarity_inversion", lambda sc: (sc.query, list(sc.passages), sc.rejected_polarity)),
        ("out_of_range_or_wordcap", lambda sc: (
            sc.query, list(sc.passages),
            (sc.chosen.replace("[1]", "[9]") if (len(sc.passages) < 9) else (sc.chosen + " extra" * 125))
        )),
    ]
    for cname, fn in neg_classes:
        for i in range(25):
            sc = SCENARIOS[i % len(SCENARIOS)]
            q, p, a = fn(sc)
            items.append((False, cname, q, p, a))

    tp = fp = tn = fn_cnt = 0
    per_class: dict[str, dict[str, int]] = {}
    for expected_pass, cname, q, p, a in items:
        res = verify(q, p, a)
        pred_pass = res.passed and res.score > 0.0
        st = per_class.setdefault(cname, {"total": 0, "correct": 0})
        st["total"] += 1
        if expected_pass == pred_pass:
            st["correct"] += 1
        if expected_pass and pred_pass:
            tp += 1
        elif not expected_pass and pred_pass:
            fp += 1
        elif not expected_pass and not pred_pass:
            tn += 1
        else:
            fn_cnt += 1

    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn_cnt)
    f1 = 2 * prec * rec / max(1e-12, prec + rec)
    return {
        "total": len(items),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn_cnt,
        "precision": round(prec, 6),
        "recall": round(rec, 6),
        "f1": round(f1, 6),
        "accuracy": round((tp + tn) / len(items), 6),
        "per_class": per_class,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    seed = 42
    include_ext = "--extended" in args
    if "--seed" in args:
        idx = args.index("--seed")
        seed = int(args[idx + 1])

    if "--benchmark" in args:
        report = run_detection_benchmark(seed=seed)
        print(json.dumps(report, indent=2))
        return 0 if (report["fp"] == 0 and report["fn"] == 0) else 1

    pairs = generate_dpo_pairs(seed=seed, include_extended=include_ext)
    if "--jsonl" in args:
        for p in pairs:
            print(json.dumps(p.to_dict()))
    else:
        summary = {
            "seed": seed,
            "pair_count": len(pairs),
            "categories": sorted({p.category for p in pairs}),
            "min_chosen_score": min(p.chosen_score for p in pairs),
            "max_rejected_score": max(p.rejected_score for p in pairs),
            "sample_pair_ids": [p.pair_id for p in pairs[:6]],
        }
        print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
