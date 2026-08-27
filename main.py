"""
POST /quantize — two-phase (freeze / select) candidate-admission API.

Run:
    pip install -r requirements.txt
    uvicorn main:app --host 0.0.0.0 --port 3000
"""

import hashlib
import json
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

# ---- in-memory persistence: freezeId -> {"rawInput": ..., "response": ...} ----
FREEZE_STORE: Dict[str, Dict[str, Any]] = {}


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------

def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def utf8_bytes(s: str) -> int:
    return len(s.encode("utf-8"))


def utf8_key(s: str):
    return s.encode("utf-8")


def is_nonempty_str(x: Any) -> bool:
    return isinstance(x, str) and len(x) > 0


def is_plain_object(x: Any) -> bool:
    return isinstance(x, dict)


def is_finite_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and x == x and abs(x) != float("inf")


def is_safe_nonneg_int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool) and x >= 0


def deep_equal(a: Any, b: Any) -> bool:
    if a is b:
        return True
    if type(a) != type(b) and not (
        isinstance(a, (int, float)) and isinstance(b, (int, float))
    ):
        return False
    if isinstance(a, dict):
        if not isinstance(b, dict):
            return False
        if set(a.keys()) != set(b.keys()):
            return False
        return all(deep_equal(a[k], b[k]) for k in a.keys())
    if isinstance(a, list):
        if not isinstance(b, list) or len(a) != len(b):
            return False
        return all(deep_equal(x, y) for x, y in zip(a, b))
    return a == b


def error_response(status_code: int, code: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"error": code})


# ---------------------------------------------------------------------
# freeze phase
# ---------------------------------------------------------------------

def compute_inventory(files: Dict[str, str]) -> List[Dict[str, Any]]:
    inv = []
    for name, content in files.items():
        inv.append({"name": name, "bytes": utf8_bytes(content), "sha256": sha256_hex(content)})
    inv.sort(key=lambda i: utf8_key(i["name"]))
    return inv


def package_digest_of(inventory: List[Dict[str, Any]]) -> str:
    canonical = json.dumps(
        [{"name": i["name"], "bytes": i["bytes"], "sha256": i["sha256"]} for i in inventory],
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return sha256_hex(canonical)


def candidate_files_valid(cand: Dict[str, Any]) -> bool:
    files = cand.get("files")
    if not is_plain_object(files) or len(files) == 0:
        return False
    return all(isinstance(v, str) for v in files.values())


def candidate_structure_valid(cand: Any) -> bool:
    if not is_plain_object(cand):
        return False
    if not is_nonempty_str(cand.get("name")):
        return False
    if not isinstance(cand.get("loadable"), bool):
        return False
    if not is_nonempty_str(cand.get("calibrationDigest")):
        return False
    if not is_nonempty_str(cand.get("tokenizerDigest")):
        return False
    ur = cand.get("unsupportedReason")
    if ur is not None and not isinstance(ur, str):
        return False
    return True


def process_freeze_candidate(cand: Dict[str, Any], req: Dict[str, Any], allowed_reasons: List[str]) -> Dict[str, Any]:
    name = cand.get("name")
    files_ok = candidate_files_valid(cand)
    struct_ok = candidate_structure_valid(cand)

    inventory: List[Dict[str, Any]] = []
    total_bytes: Optional[int] = None
    package_digest: Optional[str] = None
    if files_ok:
        inventory = compute_inventory(cand["files"])
        total_bytes = sum(i["bytes"] for i in inventory)
        package_digest = package_digest_of(inventory)

    reason_codes: List[str] = []

    if not struct_ok or not files_ok:
        status = "invalid"
        reason_codes.append("INVALID_INPUT")
    elif is_nonempty_str(cand.get("unsupportedReason")):
        if cand["unsupportedReason"] in allowed_reasons:
            status = "unsupported"
        else:
            status = "invalid"
            reason_codes.append("UNALLOWED_UNSUPPORTED_REASON")
    else:
        ok = True
        if cand.get("loadable") is not True:
            reason_codes.append("NOT_LOADABLE")
            ok = False
        if cand.get("calibrationDigest") != req.get("calibrationDigest"):
            reason_codes.append("CALIBRATION_MISMATCH")
            ok = False
        if cand.get("tokenizerDigest") != req.get("tokenizerDigest"):
            reason_codes.append("TOKENIZER_MISMATCH")
            ok = False
        status = "frozen" if ok else "invalid"

    reason_codes.sort(key=utf8_key)

    return {
        "name": name,
        "status": status,
        "inventory": inventory,
        "totalBytes": total_bytes,
        "packageDigest": package_digest,
        "reasonCodes": reason_codes,
    }


def validate_freeze_shape(body: Dict[str, Any]) -> bool:
    if not is_nonempty_str(body.get("freezeId")) or len(body["freezeId"]) > 128:
        return False
    if not is_nonempty_str(body.get("calibrationDigest")):
        return False
    if not is_nonempty_str(body.get("tokenizerDigest")):
        return False
    candidates = body.get("candidates")
    if not isinstance(candidates, list) or len(candidates) == 0:
        return False

    allowed_reasons = body.get("allowedUnsupportedReasons", [])
    if allowed_reasons is None:
        allowed_reasons = []
    if not isinstance(allowed_reasons, list):
        return False
    for r in allowed_reasons:
        if not is_nonempty_str(r):
            return False
    if len(set(allowed_reasons)) != len(allowed_reasons):
        return False

    for c in candidates:
        if not is_plain_object(c) or not is_nonempty_str(c.get("name")):
            return False
    names = [c["name"] for c in candidates]
    if len(set(names)) != len(names):
        return False

    return True


def handle_freeze(body: Dict[str, Any]) -> JSONResponse:
    if not validate_freeze_shape(body):
        return error_response(400, "INVALID_INPUT")

    allowed_reasons = body.get("allowedUnsupportedReasons") or []

    relevant_input = {
        "calibrationDigest": body["calibrationDigest"],
        "tokenizerDigest": body["tokenizerDigest"],
        "allowedUnsupportedReasons": allowed_reasons,
        "candidates": body["candidates"],
    }

    existing = FREEZE_STORE.get(body["freezeId"])
    if existing:
        if deep_equal(existing["rawInput"], relevant_input):
            return JSONResponse(status_code=200, content=existing["response"])
        return error_response(409, "FREEZE_ID_CONFLICT")

    candidates = [
        process_freeze_candidate(c, body, allowed_reasons) for c in body["candidates"]
    ]
    candidates.sort(key=lambda c: utf8_key(c["name"]))

    response = {"freezeId": body["freezeId"], "candidates": candidates}

    FREEZE_STORE[body["freezeId"]] = {"rawInput": relevant_input, "response": response}

    return JSONResponse(status_code=200, content=response)


# ---------------------------------------------------------------------
# select phase
# ---------------------------------------------------------------------

def round12(x: float) -> float:
    return round(x, 12)


def valid_select_shape(body: Dict[str, Any]) -> bool:
    return (
        is_nonempty_str(body.get("freezeId"))
        and isinstance(body.get("candidates"), list)
        and isinstance(body.get("rows"), list)
        and is_plain_object(body.get("policy"))
    )


def handle_select(body: Dict[str, Any]) -> JSONResponse:
    if not valid_select_shape(body):
        return error_response(400, "INVALID_INPUT")

    stored = FREEZE_STORE.get(body["freezeId"])
    stored_candidates = stored["response"]["candidates"] if stored else None
    frozen_matches = stored_candidates is not None and deep_equal(stored_candidates, body["candidates"])

    policy = body["policy"]
    policy_ok = (
        is_safe_nonneg_int(policy.get("maxBytes"))
        and is_finite_number(policy.get("aggregateFloor"))
        and 0 <= policy.get("aggregateFloor", -1) <= 1
        and is_plain_object(policy.get("requiredSlices"))
        and is_finite_number(policy.get("maxLatencyMs"))
        and policy.get("maxLatencyMs", -1) >= 0
        and isinstance(policy.get("candidateOrder"), list)
    )

    cand_names = [c.get("name") for c in body["candidates"] if is_plain_object(c) and is_nonempty_str(c.get("name"))]
    order_set = set(policy.get("candidateOrder") or [])
    name_set = set(cand_names)
    same_set = order_set == name_set and len(order_set) == len(policy.get("candidateOrder") or [])

    latencies = body.get("latencies") if is_plain_object(body.get("latencies")) else {}

    results = []
    for cand in body["candidates"]:
        codes = set()
        name = cand.get("name") if is_plain_object(cand) else None

        if not is_nonempty_str(name):
            codes.add("INVALID_LINEAGE")

        stored_cand = None
        if not frozen_matches:
            codes.add("NOT_FROZEN")
        else:
            stored_cand = next((c for c in stored_candidates if c["name"] == name), None)
            if stored_cand is None or not deep_equal(stored_cand, cand):
                codes.add("INVALID_LINEAGE")
            elif stored_cand["status"] != "frozen":
                codes.add("NOT_FROZEN")

        if not policy_ok or not same_set:
            codes.add("INVALID_POLICY")

        total_bytes = None
        if is_plain_object(cand) and candidate_files_valid(cand):
            inv = compute_inventory(cand["files"])
            total_bytes = sum(i["bytes"] for i in inv)
            digest = package_digest_of(inv)
            manifest_ok = (not cand.get("packageDigest")) or cand.get("packageDigest") == digest
            if not manifest_ok:
                codes.add("INVALID_MANIFEST")
        else:
            codes.add("INVALID_MANIFEST")

        aggregate = None
        slices: Dict[str, float] = {}
        predictions_valid = is_nonempty_str(name)
        total = 0
        correct = 0
        slice_totals: Dict[str, int] = {}
        slice_correct: Dict[str, int] = {}

        for row in body["rows"]:
            if not is_plain_object(row) or not is_plain_object(row.get("predictions")):
                predictions_valid = False
                continue
            pred = row["predictions"].get(name)
            label = row.get("label")
            if pred not in (0, 1) or label not in (0, 1):
                predictions_valid = False
                continue
            total += 1
            is_correct = 1 if pred == label else 0
            correct += is_correct
            slice_name = row.get("slice")
            if is_nonempty_str(slice_name):
                slice_totals[slice_name] = slice_totals.get(slice_name, 0) + 1
                slice_correct[slice_name] = slice_correct.get(slice_name, 0) + is_correct

        if not predictions_valid or total == 0:
            codes.add("INVALID_PREDICTIONS")
            aggregate = None
            slices = {}
        else:
            aggregate = round12(correct / total)
            for s, t in slice_totals.items():
                slices[s] = round12(slice_correct[s] / t)

        required_slices = policy.get("requiredSlices") if policy_ok else {}
        if aggregate is not None:
            for slice_name, floor in (required_slices or {}).items():
                if slice_name not in slices:
                    codes.add(f"MISSING_SLICE:{slice_name}")
                elif slices[slice_name] < floor:
                    codes.add(f"SLICE_FLOOR:{slice_name}")
            agg_floor = policy.get("aggregateFloor") if policy_ok else 1
            if aggregate < agg_floor:
                codes.add("AGGREGATE_FLOOR")

        if policy_ok:
            if total_bytes is None or total_bytes > policy["maxBytes"]:
                codes.add("SIZE_LIMIT")

        latency_ms = latencies.get(name) if is_finite_number(latencies.get(name)) else None
        if policy_ok:
            if latency_ms is None or latency_ms > policy["maxLatencyMs"]:
                codes.add("LATENCY_LIMIT")

        admitted = len(codes) == 0
        sorted_codes = sorted(codes, key=utf8_key)

        results.append(
            {
                "name": name,
                "aggregate": aggregate,
                "slices": slices,
                "totalBytes": total_bytes,
                "latencyMs": latency_ms,
                "admitted": admitted,
                "reasonCodes": sorted_codes,
            }
        )

    order = policy.get("candidateOrder") or []

    def order_key(r):
        try:
            idx = order.index(r["name"])
            return (0, idx)
        except ValueError:
            return (1, utf8_key(r["name"] or ""))

    results.sort(key=order_key)

    admitted_results = [r for r in results if r["admitted"]]
    winner = None
    if admitted_results:
        def winner_key(r):
            idx = order.index(r["name"]) if r["name"] in order else len(order)
            return (r["totalBytes"], r["latencyMs"], idx)

        winner = sorted(admitted_results, key=winner_key)[0]

    package_manifest = None
    if winner and stored_candidates:
        package_manifest = next((c for c in stored_candidates if c["name"] == winner["name"]), None)

    return JSONResponse(
        status_code=200,
        content={
            "freezeId": body["freezeId"],
            "selected": winner["name"] if winner else None,
            "results": results,
            "packageManifest": package_manifest,
        },
    )


# ---------------------------------------------------------------------
# route
# ---------------------------------------------------------------------

@app.post("/quantize")
async def quantize(request: Request):
    try:
        body = await request.json()
    except Exception:
        return error_response(400, "INVALID_INPUT")

    if not is_plain_object(body):
        return error_response(400, "INVALID_INPUT")

    phase = body.get("phase")
    if phase == "freeze":
        return handle_freeze(body)
    if phase == "select":
        return handle_select(body)
    return error_response(400, "INVALID_INPUT")
