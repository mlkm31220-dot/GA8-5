import hashlib
import json
import math
import threading
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

# freezeId -> {
#     "input": canonical freeze input,
#     "response": stored freeze response
# }
FREEZES: dict[str, dict[str, Any]] = {}
LOCK = threading.Lock()

FREEZE_CODES = {
    "INVALID_INPUT",
    "UNALLOWED_UNSUPPORTED_REASON",
    "NOT_LOADABLE",
    "CALIBRATION_MISMATCH",
    "TOKENIZER_MISMATCH",
}

SELECT_CODES = {
    "NOT_FROZEN",
    "INVALID_LINEAGE",
    "INVALID_POLICY",
    "INVALID_PREDICTIONS",
    "INVALID_MANIFEST",
    "AGGREGATE_FLOOR",
    "SIZE_LIMIT",
    "LATENCY_LIMIT",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def compact_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def utf8_key(value: str):
    return value.encode("utf-8")


def is_string(x):
    return isinstance(x, str)


def is_nonempty_string(x):
    return isinstance(x, str) and len(x) > 0


def is_safe_integer(x):
    return (
        isinstance(x, int)
        and not isinstance(x, bool)
        and x >= 0
        and x <= 9007199254740991
    )


def is_finite_number(x):
    return (
        isinstance(x, (int, float))
        and not isinstance(x, bool)
        and math.isfinite(float(x))
    )


def round12(x):
    return float(f"{x:.12f}")


def unique_nonempty_strings(values):
    if not isinstance(values, list):
        return False

    seen = set()

    for value in values:
        if not is_nonempty_string(value):
            return False
        if value in seen:
            return False
        seen.add(value)

    return True


def canonical_copy(value):
    """
    JSON-compatible deep copy preserving object insertion order.
    """
    return json.loads(json.dumps(value, ensure_ascii=False))


def canonicalize_freeze_input(body):
    return canonical_copy(body)


def validate_file_inventory(files):
    """
    Returns:
        (valid, inventory, total_bytes, package_digest)
    """

    if not isinstance(files, dict) or not files:
        return False, [], None, None

    # Object keys are unique by JSON definition.
    inventory = []

    for filename, text in files.items():
        if not isinstance(filename, str):
            return False, [], None, None

        if filename == "":
            return False, [], None, None

        if not isinstance(text, str):
            return False, [], None, None

        try:
            data = text.encode("utf-8")
        except UnicodeEncodeError:
            return False, [], None, None

        inventory.append(
            {
                "name": filename,
                "bytes": len(data),
                "sha256": sha256_bytes(data),
            }
        )

    inventory.sort(key=lambda x: utf8_key(x["name"]))

    total = sum(item["bytes"] for item in inventory)

    manifest = [
        {
            "name": item["name"],
            "bytes": item["bytes"],
            "sha256": item["sha256"],
        }
        for item in inventory
    ]

    package_digest = sha256_bytes(compact_json_bytes(manifest))

    return True, inventory, total, package_digest


def sort_codes(codes):
    return sorted(set(codes), key=lambda x: x.encode("utf-8"))


def validate_freeze_request(body):
    if not isinstance(body, dict):
        return False

    if body.get("phase") != "freeze":
        return False

    freeze_id = body.get("freezeId")

    if (
        not isinstance(freeze_id, str)
        or len(freeze_id) == 0
        or len(freeze_id) > 128
    ):
        return False

    calibration = body.get("calibrationDigest")
    tokenizer = body.get("tokenizerDigest")

    if not is_nonempty_string(calibration):
        return False

    if not is_nonempty_string(tokenizer):
        return False

    allowed = body.get("allowedUnsupportedReasons")

    if not unique_nonempty_strings(allowed):
        return False

    candidates = body.get("candidates")

    if not isinstance(candidates, list) or len(candidates) == 0:
        return False

    names = set()

    for candidate in candidates:
        if not isinstance(candidate, dict):
            return False

        name = candidate.get("name")

        if not is_nonempty_string(name):
            return False

        if name in names:
            return False

        names.add(name)

        files = candidate.get("files")

        if not isinstance(files, dict) or len(files) == 0:
            return False

        if "loadable" not in candidate:
            return False

        if not isinstance(candidate["loadable"], bool):
            return False

        cd = candidate.get("calibrationDigest")
        td = candidate.get("tokenizerDigest")

        if not is_nonempty_string(cd):
            return False

        if not is_nonempty_string(td):
            return False

        if "unsupportedReason" in candidate:
            reason = candidate["unsupportedReason"]

            if reason is not None and not is_nonempty_string(reason):
                return False

    return True


def build_freeze_response(body):
    freeze_id = body["freezeId"]
    request_calibration = body["calibrationDigest"]
    request_tokenizer = body["tokenizerDigest"]
    allowed_reasons = set(body["allowedUnsupportedReasons"])

    output_candidates = []

    for candidate in body["candidates"]:
        name = candidate["name"]
        files = candidate["files"]

        valid_files, inventory, total_bytes, package_digest = (
            validate_file_inventory(files)
        )

        reasons = []

        if not valid_files:
            reasons.append("INVALID_INPUT")

            output_candidates.append(
                {
                    "name": name,
                    "status": "invalid",
                    "inventory": [],
                    "totalBytes": None,
                    "packageDigest": None,
                    "reasonCodes": sort_codes(reasons),
                }
            )
            continue

        unsupported_reason = candidate.get("unsupportedReason")

        if unsupported_reason is not None:
            if unsupported_reason not in allowed_reasons:
                reasons.append("UNALLOWED_UNSUPPORTED_REASON")
            else:
                output_candidates.append(
                    {
                        "name": name,
                        "status": "unsupported",
                        "inventory": inventory,
                        "totalBytes": total_bytes,
                        "packageDigest": package_digest,
                        "reasonCodes": [],
                    }
                )
                continue

        if not candidate["loadable"]:
            reasons.append("NOT_LOADABLE")

        if candidate["calibrationDigest"] != request_calibration:
            reasons.append("CALIBRATION_MISMATCH")

        if candidate["tokenizerDigest"] != request_tokenizer:
            reasons.append("TOKENIZER_MISMATCH")

        if reasons:
            status = "invalid"
        else:
            status = "frozen"

        output_candidates.append(
            {
                "name": name,
                "status": status,
                "inventory": inventory,
                "totalBytes": total_bytes,
                "packageDigest": package_digest,
                "reasonCodes": sort_codes(reasons),
            }
        )

    output_candidates.sort(key=lambda x: utf8_key(x["name"]))

    return {
        "freezeId": freeze_id,
        "candidates": output_candidates,
    }


def candidate_set(candidates):
    return {c.get("name") for c in candidates if isinstance(c, dict)}


def same_json(a, b):
    return compact_json_bytes(a) == compact_json_bytes(b)


def recompute_manifest(candidate):
    files = candidate.get("files")

    valid, inventory, total, digest = validate_file_inventory(files)

    if not valid:
        return False, [], None, None

    return True, inventory, total, digest


def validate_policy(policy):
    if not isinstance(policy, dict):
        return False

    max_bytes = policy.get("maxBytes")
    aggregate_floor = policy.get("aggregateFloor")
    required_slices = policy.get("requiredSlices")
    max_latency = policy.get("maxLatencyMs")
    order = policy.get("candidateOrder")

    if not is_safe_integer(max_bytes):
        return False

    if not is_finite_number(aggregate_floor):
        return False

    if not 0 <= float(aggregate_floor) <= 1:
        return False

    if not isinstance(required_slices, dict):
        return False

    seen = set()

    for name, floor in required_slices.items():
        if not is_nonempty_string(name):
            return False

        if name in seen:
            return False

        seen.add(name)

        if not is_finite_number(floor):
            return False

        if not 0 <= float(floor) <= 1:
            return False

    if not is_finite_number(max_latency):
        return False

    if float(max_latency) < 0:
        return False

    if not unique_nonempty_strings(order):
        return False

    return True


def validate_select_input(body):
    if not isinstance(body, dict):
        return False

    if body.get("phase") != "select":
        return False

    if not is_nonempty_string(body.get("freezeId")):
        return False

    if not isinstance(body.get("candidates"), list):
        return False

    if not isinstance(body.get("rows"), list):
        return False

    if not isinstance(body.get("policy"), dict):
        return False

    if not isinstance(body.get("latencies"), dict):
        return False

    return True


def prediction_is_binary(value):
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value in (0, 1)
    )


def calculate_metrics(candidate_name, rows, required_slices):
    """
    Returns:
      valid_predictions,
      aggregate,
      slice_accuracies
    """

    if not isinstance(rows, list):
        return False, None, {}

    if len(rows) == 0:
        # No rows means accuracy cannot be computed.
        return False, None, {}

    correct = 0
    slice_correct = {}
    slice_total = {}

    for row in rows:
        if not isinstance(row, dict):
            return False, None, {}

        if "label" not in row:
            return False, None, {}

        label = row["label"]

        if not prediction_is_binary(label):
            return False, None, {}

        predictions = row.get("predictions")

        if not isinstance(predictions, dict):
            return False, None, {}

        if candidate_name not in predictions:
            return False, None, {}

        prediction = predictions[candidate_name]

        if not prediction_is_binary(prediction):
            return False, None, {}

        if prediction == label:
            correct += 1

        slice_name = row.get("slice")

        if not is_nonempty_string(slice_name):
            return False, None, {}

        slice_total[slice_name] = slice_total.get(slice_name, 0) + 1

        if prediction == label:
            slice_correct[slice_name] = (
                slice_correct.get(slice_name, 0) + 1
            )

    aggregate = round12(correct / len(rows))

    slices = {}

    for slice_name in required_slices:
        if slice_name in slice_total:
            slices[slice_name] = round12(
                slice_correct.get(slice_name, 0)
                / slice_total[slice_name]
            )

    return True, aggregate, slices


def select_candidate(body, stored_response):
    request_candidates = body["candidates"]
    rows = body["rows"]
    policy = body["policy"]
    latencies = body["latencies"]

    stored_candidates = stored_response["candidates"]

    reasons_global = []

    stored_names = {
        c["name"]
        for c in stored_candidates
    }

    supplied_names = {
        c.get("name")
        for c in request_candidates
        if isinstance(c, dict)
    }

    if supplied_names != stored_names:
        reasons_global.append("INVALID_LINEAGE")

    if not validate_policy(policy):
        reasons_global.append("INVALID_POLICY")

    order = policy.get("candidateOrder", [])

    if supplied_names != set(order):
        reasons_global.append("INVALID_POLICY")

    if not isinstance(latencies, dict):
        reasons_global.append("INVALID_POLICY")

    results = []

    # Map stored freeze result by name.
    stored_by_name = {
        c["name"]: c
        for c in stored_candidates
    }

    # Map supplied candidate data by name.
    supplied_by_name = {}

    for c in request_candidates:
        if isinstance(c, dict) and isinstance(c.get("name"), str):
            supplied_by_name[c["name"]] = c

    required_slices = (
        policy.get("requiredSlices", {})
        if isinstance(policy, dict)
        else {}
    )

    max_bytes = (
        policy.get("maxBytes")
        if isinstance(policy, dict)
        else None
    )

    aggregate_floor = (
        policy.get("aggregateFloor")
        if isinstance(policy, dict)
        else None
    )

    max_latency = (
        policy.get("maxLatencyMs")
        if isinstance(policy, dict)
        else None
    )

    order_index = {
        name: i
        for i, name in enumerate(order)
    }

    names_for_results = list(supplied_names)

    names_for_results.sort(
        key=lambda n: (
            order_index.get(n, len(order)),
            utf8_key(n),
        )
    )

    for name in names_for_results:
        reasons = []

        stored = stored_by_name.get(name)
        supplied = supplied_by_name.get(name)

        aggregate = None
        slices = {}

        total_bytes = None
        latency_ms = None

        if stored is None:
            reasons.append("NOT_FROZEN")
            results.append(
                {
                    "name": name,
                    "aggregate": None,
                    "slices": {},
                    "totalBytes": None,
                    "latencyMs": None,
                    "admitted": False,
                    "reasonCodes": sort_codes(reasons),
                }
            )
            continue

        if stored["status"] != "frozen":
            reasons.append("NOT_FROZEN")

        # Validate submitted candidate against recorded freeze result.
        if supplied is None:
            reasons.append("INVALID_LINEAGE")
        else:
            valid_manifest, inventory, recomputed_total, recomputed_digest = (
                recompute_manifest(supplied)
            )

            if not valid_manifest:
                reasons.append("INVALID_MANIFEST")
            else:
                recorded_inventory = stored.get("inventory")

                if not same_json(inventory, recorded_inventory):
                    reasons.append("INVALID_MANIFEST")

                if recomputed_total != stored.get("totalBytes"):
                    reasons.append("INVALID_MANIFEST")

                if recomputed_digest != stored.get("packageDigest"):
                    reasons.append("INVALID_MANIFEST")

                # The supplied candidate's metadata must agree with
                # the frozen response.
                if supplied.get("name") != stored["name"]:
                    reasons.append("INVALID_LINEAGE")

                total_bytes = recomputed_total

        # Latency is only valid if supplied and finite/non-negative.
        if name not in latencies:
            reasons.append("LATENCY_LIMIT")
        else:
            latency_value = latencies[name]

            if not is_finite_number(latency_value):
                reasons.append("LATENCY_LIMIT")
            elif float(latency_value) < 0:
                reasons.append("LATENCY_LIMIT")
            else:
                latency_ms = float(latency_value)

        # Predictions
        valid_predictions, aggregate, slices = calculate_metrics(
            name,
            rows,
            required_slices,
        )

        if not valid_predictions:
            aggregate = None
            slices = {}
            reasons.append("INVALID_PREDICTIONS")

        else:
            if aggregate < float(aggregate_floor):
                reasons.append("AGGREGATE_FLOOR")

            for slice_name, floor in required_slices.items():

                if slice_name not in slices:
                    reasons.append(f"MISSING_SLICE:{slice_name}")
                elif slices[slice_name] < float(floor):
                    reasons.append(f"SLICE_FLOOR:{slice_name}")

        # Size
        if total_bytes is not None:
            if total_bytes > max_bytes:
                reasons.append("SIZE_LIMIT")

        # Latency
        if latency_ms is not None:
            if latency_ms > float(max_latency):
                reasons.append("LATENCY_LIMIT")

        admitted = len(reasons) == 0

        results.append(
            {
                "name": name,
                "aggregate": aggregate,
                "slices": slices,
                "totalBytes": total_bytes,
                "latencyMs": latency_ms,
                "admitted": admitted,
                "reasonCodes": sort_codes(reasons),
            }
        )

    # Choose winner:
    # 1. admitted only
    # 2. smaller bytes
    # 3. lower latency
    # 4. candidate order
    admitted_results = [
        r for r in results
        if r["admitted"]
    ]

    winner = None

    if admitted_results:
        winner = min(
            admitted_results,
            key=lambda r: (
                r["totalBytes"],
                r["latencyMs"],
                order_index.get(
                    r["name"],
                    len(order)
                ),
                utf8_key(r["name"]),
            ),
        )

    package_manifest = None

    if winner is not None:
        stored_winner = stored_by_name[winner["name"]]

        package_manifest = canonical_copy(stored_winner)

    return {
        "freezeId": body["freezeId"],
        "selected": (
            winner["name"]
            if winner is not None
            else None
        ),
        "results": results,
        "packageManifest": package_manifest,
    }


@app.post("/quantize")
async def quantize(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            status_code=400,
            content={"error": "INVALID_INPUT"},
        )

    # -------------------------
    # FREEZE
    # -------------------------
    if isinstance(body, dict) and body.get("phase") == "freeze":

        if not validate_freeze_request(body):
            return JSONResponse(
                status_code=400,
                content={"error": "INVALID_INPUT"},
            )

        freeze_id = body["freezeId"]

        canonical_input = canonicalize_freeze_input(body)

        with LOCK:
            existing = FREEZES.get(freeze_id)

            if existing is not None:

                if same_json(existing["input"], canonical_input):
                    # Identical replay: return exactly the same object.
                    return JSONResponse(
                        status_code=200,
                        content=canonical_copy(
                            existing["response"]
                        ),
                    )

                return JSONResponse(
                    status_code=409,
                    content={
                        "error": "FREEZE_ID_CONFLICT"
                    },
                )

            response = build_freeze_response(body)

            FREEZES[freeze_id] = {
                "input": canonical_input,
                "response": canonical_copy(response),
            }

        return JSONResponse(
            status_code=200,
            content=response,
        )

    # -------------------------
    # SELECT
    # -------------------------
    if isinstance(body, dict) and body.get("phase") == "select":

        if not validate_select_input(body):
            return JSONResponse(
                status_code=400,
                content={"error": "INVALID_INPUT"},
            )

        freeze_id = body["freezeId"]

        with LOCK:
            stored = FREEZES.get(freeze_id)

        if stored is None:
            # The specification requires selection codes for
            # selection-time lineage failures rather than a 404.
            # Build a response using the submitted candidates.
            policy = body["policy"]

            results = []

            order = (
                policy.get("candidateOrder", [])
                if isinstance(policy, dict)
                else []
            )

            order_index = {
                n: i for i, n in enumerate(order)
            }

            supplied = body["candidates"]

            names = [
                c.get("name")
                for c in supplied
                if isinstance(c, dict)
                and isinstance(c.get("name"), str)
            ]

            names = list(set(names))

            names.sort(
                key=lambda n: (
                    order_index.get(n, len(order)),
                    utf8_key(n),
                )
            )

            for name in names:
                results.append(
                    {
                        "name": name,
                        "aggregate": None,
                        "slices": {},
                        "totalBytes": None,
                        "latencyMs": None,
                        "admitted": False,
                        "reasonCodes": ["NOT_FROZEN"],
                    }
                )

            return JSONResponse(
                status_code=200,
                content={
                    "freezeId": freeze_id,
                    "selected": None,
                    "results": results,
                    "packageManifest": None,
                },
            )

        response = select_candidate(
            body,
            stored["response"],
        )

        return JSONResponse(
            status_code=200,
            content=response,
        )

    # Unknown/missing phase.
    return JSONResponse(
        status_code=400,
        content={"error": "INVALID_INPUT"},
    )


@app.get("/")
def root():
    return {
        "service": "quantize",
        "endpoint": "/quantize",
        "method": "POST",
    }
