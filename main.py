import hashlib
import json
import math
import threading
from typing import Any, Dict, List, Optional, Union

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

app = FastAPI(title="Quantize Candidate Admission API")


# ============================================================
# Pydantic request model
# ============================================================

class QuantizeRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    phase: Optional[str] = None
    freezeId: Optional[str] = None

    calibrationDigest: Optional[str] = None
    tokenizerDigest: Optional[str] = None

    allowedUnsupportedReasons: Optional[List[str]] = None

    candidates: Optional[List[Dict[str, Any]]] = None

    policy: Optional[Dict[str, Any]] = None

    latencies: Optional[Dict[str, Any]] = None

    rows: Optional[List[Dict[str, Any]]] = None


# ============================================================
# State
# ============================================================

FREEZES: Dict[str, Dict[str, Any]] = {}
LOCK = threading.Lock()


# ============================================================
# Helpers
# ============================================================

def compact_json(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":")
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def utf8_sort_key(value: str):
    return value.encode("utf-8")


def sort_codes(codes):
    return sorted(set(codes), key=utf8_sort_key)


def is_nonempty_string(value):
    return isinstance(value, str) and len(value) > 0


def is_safe_integer(value):
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 0 <= value <= 9007199254740991
    )


def is_finite_number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def round12(value):
    return float(f"{value:.12f}")


def deep_copy(value):
    return json.loads(json.dumps(value, ensure_ascii=False))


# ============================================================
# File integrity
# ============================================================

def build_inventory(files):
    if not isinstance(files, dict):
        return False, [], None, None

    if len(files) == 0:
        return False, [], None, None

    inventory = []

    for filename, text in files.items():

        if not isinstance(filename, str) or filename == "":
            return False, [], None, None

        if not isinstance(text, str):
            return False, [], None, None

        try:
            data = text.encode("utf-8")
        except UnicodeEncodeError:
            return False, [], None, None

        inventory.append({
            "name": filename,
            "bytes": len(data),
            "sha256": sha256_bytes(data)
        })

    inventory.sort(
        key=lambda item: utf8_sort_key(item["name"])
    )

    total_bytes = sum(
        item["bytes"]
        for item in inventory
    )

    package_digest = sha256_bytes(
        compact_json(inventory)
    )

    return (
        True,
        inventory,
        total_bytes,
        package_digest
    )


# ============================================================
# Freeze validation
# ============================================================

def validate_freeze(body):

    if body.get("phase") != "freeze":
        return False

    freeze_id = body.get("freezeId")

    if (
        not isinstance(freeze_id, str)
        or len(freeze_id) == 0
        or len(freeze_id) > 128
    ):
        return False

    if not is_nonempty_string(
        body.get("calibrationDigest")
    ):
        return False

    if not is_nonempty_string(
        body.get("tokenizerDigest")
    ):
        return False

    allowed = body.get(
        "allowedUnsupportedReasons"
    )

    if not isinstance(allowed, list):
        return False

    if not all(
        is_nonempty_string(x)
        for x in allowed
    ):
        return False

    if len(allowed) != len(set(allowed)):
        return False

    candidates = body.get("candidates")

    if not isinstance(candidates, list):
        return False

    if len(candidates) == 0:
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

        if not isinstance(
            candidate.get("files"),
            dict
        ):
            return False

        if len(candidate["files"]) == 0:
            return False

        if not isinstance(
            candidate.get("loadable"),
            bool
        ):
            return False

        if not is_nonempty_string(
            candidate.get("calibrationDigest")
        ):
            return False

        if not is_nonempty_string(
            candidate.get("tokenizerDigest")
        ):
            return False

        if "unsupportedReason" in candidate:

            reason = candidate["unsupportedReason"]

            if (
                reason is not None
                and not is_nonempty_string(reason)
            ):
                return False

    return True


# ============================================================
# Freeze
# ============================================================

def perform_freeze(body):

    request_calibration = body[
        "calibrationDigest"
    ]

    request_tokenizer = body[
        "tokenizerDigest"
    ]

    allowed = set(
        body["allowedUnsupportedReasons"]
    )

    results = []

    for candidate in body["candidates"]:

        name = candidate["name"]

        valid_files, inventory, total_bytes, package_digest = (
            build_inventory(
                candidate["files"]
            )
        )

        # Invalid file data
        if not valid_files:

            results.append({
                "name": name,
                "status": "invalid",
                "inventory": [],
                "totalBytes": None,
                "packageDigest": None,
                "reasonCodes": [
                    "INVALID_INPUT"
                ]
            })

            continue

        reason_codes = []

        unsupported_reason = candidate.get(
            "unsupportedReason"
        )

        # Unsupported candidate
        if unsupported_reason is not None:

            if unsupported_reason not in allowed:

                reason_codes.append(
                    "UNALLOWED_UNSUPPORTED_REASON"
                )

                results.append({
                    "name": name,
                    "status": "invalid",
                    "inventory": inventory,
                    "totalBytes": total_bytes,
                    "packageDigest": package_digest,
                    "reasonCodes": sort_codes(
                        reason_codes
                    )
                })

            else:

                results.append({
                    "name": name,
                    "status": "unsupported",
                    "inventory": inventory,
                    "totalBytes": total_bytes,
                    "packageDigest": package_digest,
                    "reasonCodes": []
                })

            continue

        # Normal candidate
        if not candidate["loadable"]:
            reason_codes.append(
                "NOT_LOADABLE"
            )

        if (
            candidate["calibrationDigest"]
            != request_calibration
        ):
            reason_codes.append(
                "CALIBRATION_MISMATCH"
            )

        if (
            candidate["tokenizerDigest"]
            != request_tokenizer
        ):
            reason_codes.append(
                "TOKENIZER_MISMATCH"
            )

        status = (
            "frozen"
            if len(reason_codes) == 0
            else "invalid"
        )

        results.append({
            "name": name,
            "status": status,
            "inventory": inventory,
            "totalBytes": total_bytes,
            "packageDigest": package_digest,
            "reasonCodes": sort_codes(
                reason_codes
            )
        })

    # UTF-8 name ordering
    results.sort(
        key=lambda item: utf8_sort_key(
            item["name"]
        )
    )

    return {
        "freezeId": body["freezeId"],
        "candidates": results
    }


# ============================================================
# Policy validation
# ============================================================

def validate_policy(policy):

    if not isinstance(policy, dict):
        return False

    if not is_safe_integer(
        policy.get("maxBytes")
    ):
        return False

    floor = policy.get(
        "aggregateFloor"
    )

    if not is_finite_number(floor):
        return False

    if not 0 <= float(floor) <= 1:
        return False

    required = policy.get(
        "requiredSlices"
    )

    if not isinstance(required, dict):
        return False

    for name, value in required.items():

        if not is_nonempty_string(name):
            return False

        if not is_finite_number(value):
            return False

        if not 0 <= float(value) <= 1:
            return False

    latency = policy.get(
        "maxLatencyMs"
    )

    if not is_finite_number(latency):
        return False

    if float(latency) < 0:
        return False

    order = policy.get(
        "candidateOrder"
    )

    if not isinstance(order, list):
        return False

    if not all(
        is_nonempty_string(x)
        for x in order
    ):
        return False

    if len(order) != len(set(order)):
        return False

    return True


# ============================================================
# Prediction validation
# ============================================================

def is_binary(value):
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value in (0, 1)
    )


def calculate_accuracy(
    candidate_name,
    rows,
    required_slices
):

    if not isinstance(rows, list):
        return False, None, {}

    if len(rows) == 0:
        return False, None, {}

    correct = 0

    slice_total = {}
    slice_correct = {}

    for row in rows:

        if not isinstance(row, dict):
            return False, None, {}

        label = row.get("label")

        if not is_binary(label):
            return False, None, {}

        slice_name = row.get(
            "slice"
        )

        if not is_nonempty_string(
            slice_name
        ):
            return False, None, {}

        predictions = row.get(
            "predictions"
        )

        if not isinstance(
            predictions,
            dict
        ):
            return False, None, {}

        if candidate_name not in predictions:
            return False, None, {}

        prediction = predictions[
            candidate_name
        ]

        if not is_binary(prediction):
            return False, None, {}

        if prediction == label:
            correct += 1

        slice_total[slice_name] = (
            slice_total.get(slice_name, 0) + 1
        )

        if prediction == label:
            slice_correct[slice_name] = (
                slice_correct.get(slice_name, 0) + 1
            )

    aggregate = round12(
        correct / len(rows)
    )

    slices = {}

    for slice_name in required_slices:

        if slice_name in slice_total:

            slices[slice_name] = round12(
                slice_correct.get(
                    slice_name,
                    0
                )
                / slice_total[slice_name]
            )

    return True, aggregate, slices


# ============================================================
# Select
# ============================================================

def perform_select(
    body,
    frozen_response
):

    stored_candidates = (
        frozen_response["candidates"]
    )

    supplied_candidates = (
        body["candidates"]
    )

    supplied_by_name = {}

    for candidate in supplied_candidates:

        if isinstance(candidate, dict):
            name = candidate.get("name")

            if isinstance(name, str):
                supplied_by_name[name] = candidate

    stored_by_name = {
        candidate["name"]: candidate
        for candidate in stored_candidates
    }

    supplied_names = set(
        supplied_by_name.keys()
    )

    stored_names = set(
        stored_by_name.keys()
    )

    policy = body["policy"]

    policy_valid = validate_policy(
        policy
    )

    required_slices = (
        policy.get(
            "requiredSlices",
            {}
        )
        if isinstance(policy, dict)
        else {}
    )

    candidate_order = (
        policy.get(
            "candidateOrder",
            []
        )
        if isinstance(policy, dict)
        else []
    )

    reasons_global = []

    if supplied_names != stored_names:
        reasons_global.append(
            "INVALID_LINEAGE"
        )

    if not policy_valid:
        reasons_global.append(
            "INVALID_POLICY"
        )

    if supplied_names != set(
        candidate_order
    ):
        reasons_global.append(
            "INVALID_POLICY"
        )

    order_index = {
        name: index
        for index, name
        in enumerate(candidate_order)
    }

    names = list(
        supplied_names
    )

    names.sort(
        key=lambda name: (
            order_index.get(
                name,
                len(candidate_order)
            ),
            utf8_sort_key(name)
        )
    )

    results = []

    for name in names:

        candidate_reasons = list(
            reasons_global
        )

        stored = stored_by_name.get(
            name
        )

        supplied = supplied_by_name.get(
            name
        )

        aggregate = None
        slices = {}

        total_bytes = None
        latency_ms = None

        # Must have been frozen
        if (
            stored is None
            or stored.get("status") != "frozen"
        ):
            candidate_reasons.append(
                "NOT_FROZEN"
            )

        # Validate manifest
        if supplied is None:

            candidate_reasons.append(
                "INVALID_LINEAGE"
            )

        else:

            valid, inventory, total, digest = (
                build_inventory(
                    supplied.get("files")
                )
            )

            if not valid:

                candidate_reasons.append(
                    "INVALID_MANIFEST"
                )

            else:

                if inventory != stored.get(
                    "inventory"
                ):
                    candidate_reasons.append(
                        "INVALID_MANIFEST"
                    )

                if total != stored.get(
                    "totalBytes"
                ):
                    candidate_reasons.append(
                        "INVALID_MANIFEST"
                    )

                if digest != stored.get(
                    "packageDigest"
                ):
                    candidate_reasons.append(
                        "INVALID_MANIFEST"
                    )

                if not candidate_reasons or True:
                    total_bytes = total

        # Latency
        latencies = body.get(
            "latencies"
        )

        if (
            not isinstance(
                latencies,
                dict
            )
            or name not in latencies
            or not is_finite_number(
                latencies[name]
            )
            or float(latencies[name]) < 0
        ):

            candidate_reasons.append(
                "LATENCY_LIMIT"
            )

        else:

            latency_ms = float(
                latencies[name]
            )

        # Predictions
        valid_predictions, aggregate, slices = (
            calculate_accuracy(
                name,
                body["rows"],
                required_slices
            )
        )

        if not valid_predictions:

            aggregate = None
            slices = {}

            candidate_reasons.append(
                "INVALID_PREDICTIONS"
            )

        else:

            if aggregate < float(
                policy["aggregateFloor"]
            ):
                candidate_reasons.append(
                    "AGGREGATE_FLOOR"
                )

            for slice_name, floor in (
                required_slices.items()
            ):

                if slice_name not in slices:

                    candidate_reasons.append(
                        f"MISSING_SLICE:{slice_name}"
                    )

                elif slices[slice_name] < float(
                    floor
                ):

                    candidate_reasons.append(
                        f"SLICE_FLOOR:{slice_name}"
                    )

        # Size
        if total_bytes is not None:

            if total_bytes > policy[
                "maxBytes"
            ]:

                candidate_reasons.append(
                    "SIZE_LIMIT"
                )

        # Latency
        if latency_ms is not None:

            if latency_ms > float(
                policy["maxLatencyMs"]
            ):

                candidate_reasons.append(
                    "LATENCY_LIMIT"
                )

        candidate_reasons = sort_codes(
            candidate_reasons
        )

        admitted = (
            len(candidate_reasons) == 0
        )

        results.append({
            "name": name,
            "aggregate": aggregate,
            "slices": slices,
            "totalBytes": total_bytes,
            "latencyMs": latency_ms,
            "admitted": admitted,
            "reasonCodes": candidate_reasons
        })

    # Winner
    admitted = [
        result
        for result in results
        if result["admitted"]
    ]

    winner = None

    if admitted:

        winner = min(
            admitted,
            key=lambda result: (
                result["totalBytes"],
                result["latencyMs"],
                order_index.get(
                    result["name"],
                    len(candidate_order)
                ),
                utf8_sort_key(
                    result["name"]
                )
            )
        )

    package_manifest = None

    if winner is not None:

        package_manifest = deep_copy(
            stored_by_name[
                winner["name"]
            ]
        )

    return {
        "freezeId": body["freezeId"],
        "selected": (
            winner["name"]
            if winner
            else None
        ),
        "results": results,
        "packageManifest": package_manifest
    }


# ============================================================
# POST /quantize
# ============================================================

@app.post(
    "/quantize",
    response_class=JSONResponse
)
async def quantize(body: QuantizeRequest):

    data = body.model_dump(
        exclude_none=False
    )

    phase = data.get(
        "phase"
    )

    # ========================================================
    # FREEZE
    # ========================================================

    if phase == "freeze":

        if not validate_freeze(data):

            return JSONResponse(
                status_code=400,
                content={
                    "error": "INVALID_INPUT"
                }
            )

        freeze_id = data[
            "freezeId"
        ]

        with LOCK:

            existing = FREEZES.get(
                freeze_id
            )

            if existing is not None:

                if existing["input"] == data:

                    return JSONResponse(
                        status_code=200,
                        content=deep_copy(
                            existing[
                                "response"
                            ]
                        )
                    )

                return JSONResponse(
                    status_code=409,
                    content={
                        "error":
                        "FREEZE_ID_CONFLICT"
                    }
                )

            response = perform_freeze(
                data
            )

            FREEZES[freeze_id] = {
                "input": deep_copy(data),
                "response": deep_copy(
                    response
                )
            }

        return JSONResponse(
            status_code=200,
            content=response
        )

    # ========================================================
    # SELECT
    # ========================================================

    if phase == "select":

        # Required top-level structure
        if (
            not isinstance(
                data.get("freezeId"),
                str
            )
            or not isinstance(
                data.get("candidates"),
                list
            )
            or not isinstance(
                data.get("rows"),
                list
            )
            or not isinstance(
                data.get("policy"),
                dict
            )
        ):

            return JSONResponse(
                status_code=400,
                content={
                    "error": "INVALID_INPUT"
                }
            )

        freeze_id = data[
            "freezeId"
        ]

        with LOCK:
            stored = FREEZES.get(
                freeze_id
            )

        if stored is None:

            # Selection response with NOT_FROZEN
            results = []

            for candidate in data[
                "candidates"
            ]:

                if not isinstance(
                    candidate,
                    dict
                ):
                    continue

                name = candidate.get(
                    "name"
                )

                if not isinstance(
                    name,
                    str
                ):
                    continue

                results.append({
                    "name": name,
                    "aggregate": None,
                    "slices": {},
                    "totalBytes": None,
                    "latencyMs": None,
                    "admitted": False,
                    "reasonCodes": [
                        "NOT_FROZEN"
                    ]
                })

            results.sort(
                key=lambda r:
                utf8_sort_key(
                    r["name"]
                )
            )

            return JSONResponse(
                status_code=200,
                content={
                    "freezeId": freeze_id,
                    "selected": None,
                    "results": results,
                    "packageManifest": None
                }
            )

        response = perform_select(
            data,
            stored["response"]
        )

        return JSONResponse(
            status_code=200,
            content=response
        )

    # Missing/unknown phase
    return JSONResponse(
        status_code=400,
        content={
            "error": "INVALID_INPUT"
        }
    )


@app.get("/")
def root():
    return {
        "service": "quantize",
        "endpoint": "POST /quantize"
    }
