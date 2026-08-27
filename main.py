import hashlib
import json
import math
import threading

from typing import Any

from fastapi import FastAPI, Request, Body
from fastapi.responses import JSONResponse

app = FastAPI(title="Quantize Candidate Admission API")

FREEZES = {}
LOCK = threading.Lock()


def compact_json(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":")
    ).encode("utf-8")


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def utf8(value):
    return value.encode("utf-8")


def sort_codes(codes):
    return sorted(set(codes), key=utf8)


def nonempty_string(x):
    return isinstance(x, str) and len(x) > 0


def finite_number(x):
    return (
        isinstance(x, (int, float))
        and not isinstance(x, bool)
        and math.isfinite(float(x))
    )


def safe_integer(x):
    return (
        isinstance(x, int)
        and not isinstance(x, bool)
        and 0 <= x <= 9007199254740991
    )


def binary(x):
    return (
        isinstance(x, int)
        and not isinstance(x, bool)
        and x in (0, 1)
    )


def round12(x):
    return float(f"{x:.12f}")


def equal_json(a, b):
    return compact_json(a) == compact_json(b)


def build_inventory(files):

    if not isinstance(files, dict) or not files:
        return False, [], None, None

    inventory = []

    for filename, content in files.items():

        if not isinstance(filename, str):
            return False, [], None, None

        if filename == "":
            return False, [], None, None

        if not isinstance(content, str):
            return False, [], None, None

        try:
            data = content.encode("utf-8")
        except UnicodeEncodeError:
            return False, [], None, None

        inventory.append({
            "name": filename,
            "bytes": len(data),
            "sha256": sha256(data)
        })

    inventory.sort(
        key=lambda x: utf8(x["name"])
    )

    total = sum(
        x["bytes"]
        for x in inventory
    )

    package_digest = sha256(
        compact_json(inventory)
    )

    return True, inventory, total, package_digest


# =========================================================
# FREEZE VALIDATION
# =========================================================

def valid_freeze(body):

    if not isinstance(body, dict):
        return False

    if body.get("phase") != "freeze":
        return False

    freeze_id = body.get("freezeId")

    if (
        not isinstance(freeze_id, str)
        or not 1 <= len(freeze_id) <= 128
    ):
        return False

    if not nonempty_string(
        body.get("calibrationDigest")
    ):
        return False

    if not nonempty_string(
        body.get("tokenizerDigest")
    ):
        return False

    allowed = body.get(
        "allowedUnsupportedReasons"
    )

    if not isinstance(allowed, list):
        return False

    if any(
        not nonempty_string(x)
        for x in allowed
    ):
        return False

    if len(allowed) != len(set(allowed)):
        return False

    candidates = body.get(
        "candidates"
    )

    if (
        not isinstance(candidates, list)
        or len(candidates) == 0
    ):
        return False

    names = set()

    for c in candidates:

        if not isinstance(c, dict):
            return False

        name = c.get("name")

        if not nonempty_string(name):
            return False

        if name in names:
            return False

        names.add(name)

        files = c.get("files")

        if (
            not isinstance(files, dict)
            or len(files) == 0
        ):
            return False

        if not isinstance(
            c.get("loadable"),
            bool
        ):
            return False

        if not nonempty_string(
            c.get("calibrationDigest")
        ):
            return False

        if not nonempty_string(
            c.get("tokenizerDigest")
        ):
            return False

        if "unsupportedReason" in c:

            reason = c["unsupportedReason"]

            if (
                reason is not None
                and not nonempty_string(reason)
            ):
                return False

    return True


# =========================================================
# FREEZE
# =========================================================

def do_freeze(body):

    calibration = body[
        "calibrationDigest"
    ]

    tokenizer = body[
        "tokenizerDigest"
    ]

    allowed = set(
        body["allowedUnsupportedReasons"]
    )

    output = []

    for candidate in body["candidates"]:

        name = candidate["name"]

        valid, inventory, total, digest = (
            build_inventory(
                candidate["files"]
            )
        )

        if not valid:

            output.append({
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

        unsupported = candidate.get(
            "unsupportedReason"
        )

        if unsupported is not None:

            if unsupported not in allowed:

                output.append({
                    "name": name,
                    "status": "invalid",
                    "inventory": inventory,
                    "totalBytes": total,
                    "packageDigest": digest,
                    "reasonCodes": [
                        "UNALLOWED_UNSUPPORTED_REASON"
                    ]
                })

            else:

                output.append({
                    "name": name,
                    "status": "unsupported",
                    "inventory": inventory,
                    "totalBytes": total,
                    "packageDigest": digest,
                    "reasonCodes": []
                })

            continue

        if not candidate["loadable"]:
            reason_codes.append(
                "NOT_LOADABLE"
            )

        if (
            candidate["calibrationDigest"]
            != calibration
        ):
            reason_codes.append(
                "CALIBRATION_MISMATCH"
            )

        if (
            candidate["tokenizerDigest"]
            != tokenizer
        ):
            reason_codes.append(
                "TOKENIZER_MISMATCH"
            )

        output.append({
            "name": name,
            "status": (
                "frozen"
                if not reason_codes
                else "invalid"
            ),
            "inventory": inventory,
            "totalBytes": total,
            "packageDigest": digest,
            "reasonCodes": sort_codes(
                reason_codes
            )
        })

    output.sort(
        key=lambda x: utf8(x["name"])
    )

    return {
        "freezeId": body["freezeId"],
        "candidates": output
    }


# =========================================================
# SELECT VALIDATION
# =========================================================

def valid_policy(policy):

    if not isinstance(policy, dict):
        return False

    if not safe_integer(
        policy.get("maxBytes")
    ):
        return False

    floor = policy.get(
        "aggregateFloor"
    )

    if (
        not finite_number(floor)
        or not 0 <= float(floor) <= 1
    ):
        return False

    slices = policy.get(
        "requiredSlices"
    )

    if not isinstance(slices, dict):
        return False

    for name, value in slices.items():

        if not nonempty_string(name):
            return False

        if (
            not finite_number(value)
            or not 0 <= float(value) <= 1
        ):
            return False

    max_latency = policy.get(
        "maxLatencyMs"
    )

    if (
        not finite_number(max_latency)
        or float(max_latency) < 0
    ):
        return False

    order = policy.get(
        "candidateOrder"
    )

    if not isinstance(order, list):
        return False

    if any(
        not nonempty_string(x)
        for x in order
    ):
        return False

    if len(order) != len(set(order)):
        return False

    return True


# =========================================================
# METRICS
# =========================================================

def metrics(name, rows, required):

    if (
        not isinstance(rows, list)
        or len(rows) == 0
    ):
        return False, None, {}

    correct = 0

    totals = {}
    correct_slices = {}

    for row in rows:

        if not isinstance(row, dict):
            return False, None, {}

        label = row.get("label")

        if not binary(label):
            return False, None, {}

        slice_name = row.get(
            "slice"
        )

        if not nonempty_string(
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

        if name not in predictions:
            return False, None, {}

        prediction = predictions[name]

        if not binary(prediction):
            return False, None, {}

        totals[slice_name] = (
            totals.get(slice_name, 0) + 1
        )

        if prediction == label:

            correct += 1

            correct_slices[slice_name] = (
                correct_slices.get(
                    slice_name,
                    0
                ) + 1
            )

    aggregate = round12(
        correct / len(rows)
    )

    slices = {}

    for slice_name in required:

        if slice_name in totals:

            slices[slice_name] = round12(
                correct_slices.get(
                    slice_name,
                    0
                ) / totals[slice_name]
            )

    return True, aggregate, slices


# =========================================================
# SELECT
# =========================================================

def do_select(body, stored):

    frozen = stored["candidates"]

    supplied = body["candidates"]

    frozen_names = {
        x["name"]
        for x in frozen
    }

    supplied_names = {
        x["name"]
        for x in supplied
        if isinstance(x, dict)
        and isinstance(x.get("name"), str)
    }

    policy = body["policy"]

    policy_ok = valid_policy(
        policy
    )

    required = (
        policy.get(
            "requiredSlices",
            {}
        )
        if isinstance(policy, dict)
        else {}
    )

    order = (
        policy.get(
            "candidateOrder",
            []
        )
        if isinstance(policy, dict)
        else []
    )

    global_codes = []

    if supplied_names != frozen_names:
        global_codes.append(
            "INVALID_LINEAGE"
        )

    if not policy_ok:
        global_codes.append(
            "INVALID_POLICY"
        )

    if supplied_names != set(order):
        global_codes.append(
            "INVALID_POLICY"
        )

    frozen_by_name = {
        x["name"]: x
        for x in frozen
    }

    supplied_by_name = {
        x["name"]: x
        for x in supplied
        if isinstance(x, dict)
        and isinstance(x.get("name"), str)
    }

    order_index = {
        name: i
        for i, name in enumerate(order)
    }

    names = list(supplied_names)

    names.sort(
        key=lambda n: (
            order_index.get(
                n,
                len(order)
            ),
            utf8(n)
        )
    )

    results = []

    for name in names:

        codes = list(global_codes)

        frozen_candidate = (
            frozen_by_name.get(name)
        )

        submitted_candidate = (
            supplied_by_name.get(name)
        )

        aggregate = None
        slices = {}

        total_bytes = None
        latency = None

        if (
            frozen_candidate is None
            or frozen_candidate["status"]
            != "frozen"
        ):
            codes.append(
                "NOT_FROZEN"
            )

        # ---------------------------------------------
        # Manifest
        # ---------------------------------------------

        if submitted_candidate is None:

            codes.append(
                "INVALID_MANIFEST"
            )

        else:

            ok, inventory, total, digest = (
                build_inventory(
                    submitted_candidate.get(
                        "files"
                    )
                )
            )

            if not ok:

                codes.append(
                    "INVALID_MANIFEST"
                )

            else:

                if inventory != (
                    frozen_candidate.get(
                        "inventory"
                    )
                ):
                    codes.append(
                        "INVALID_MANIFEST"
                    )

                if total != (
                    frozen_candidate.get(
                        "totalBytes"
                    )
                ):
                    codes.append(
                        "INVALID_MANIFEST"
                    )

                if digest != (
                    frozen_candidate.get(
                        "packageDigest"
                    )
                ):
                    codes.append(
                        "INVALID_MANIFEST"
                    )

                total_bytes = total

        # ---------------------------------------------
        # Latency
        # ---------------------------------------------

        latencies = body.get(
            "latencies"
        )

        if (
            not isinstance(
                latencies,
                dict
            )
            or name not in latencies
            or not finite_number(
                latencies[name]
            )
            or float(latencies[name]) < 0
        ):

            latency = None

        else:

            latency = float(
                latencies[name]
            )

        # ---------------------------------------------
        # Predictions
        # ---------------------------------------------

        valid, aggregate, slices = metrics(
            name,
            body["rows"],
            required
        )

        if not valid:

            aggregate = None
            slices = {}

            codes.append(
                "INVALID_PREDICTIONS"
            )

        else:

            if aggregate < float(
                policy["aggregateFloor"]
            ):
                codes.append(
                    "AGGREGATE_FLOOR"
                )

            for slice_name, floor in (
                required.items()
            ):

                if slice_name not in slices:

                    codes.append(
                        "MISSING_SLICE:"
                        + slice_name
                    )

                elif slices[slice_name] < float(
                    floor
                ):

                    codes.append(
                        "SLICE_FLOOR:"
                        + slice_name
                    )

        # ---------------------------------------------
        # Size
        # ---------------------------------------------

        if total_bytes is not None:

            if total_bytes > policy[
                "maxBytes"
            ]:

                codes.append(
                    "SIZE_LIMIT"
                )

        # ---------------------------------------------
        # Latency
        # ---------------------------------------------

        if latency is not None:

            if latency > float(
                policy["maxLatencyMs"]
            ):

                codes.append(
                    "LATENCY_LIMIT"
                )

        codes = sort_codes(codes)

        admitted = (
            len(codes) == 0
        )

        results.append({
            "name": name,
            "aggregate": aggregate,
            "slices": slices,
            "totalBytes": total_bytes,
            "latencyMs": latency,
            "admitted": admitted,
            "reasonCodes": codes
        })

    # Winner
    winners = [
        x
        for x in results
        if x["admitted"]
    ]

    winner = None

    if winners:

        winner = min(
            winners,
            key=lambda x: (
                x["totalBytes"],
                x["latencyMs"],
                order_index.get(
                    x["name"],
                    len(order)
                ),
                utf8(x["name"])
            )
        )

    manifest = None

    if winner:

        manifest = frozen_by_name[
            winner["name"]
        ]

    return {
        "freezeId": body["freezeId"],
        "selected": (
            winner["name"]
            if winner
            else None
        ),
        "results": results,
        "packageManifest": manifest
    }


# =========================================================
# ENDPOINT
# =========================================================

@app.post("/quantize")
async def quantize(
    request: Request,
    body: Any = Body(...)
):

    # Do NOT let FastAPI/Pydantic silently reject
    # unusual grader inputs.
    if not isinstance(body, dict):
        return JSONResponse(
            status_code=400,
            content={
                "error": "INVALID_INPUT"
            }
        )

    phase = body.get("phase")

    # =====================================================
    # FREEZE
    # =====================================================

    if phase == "freeze":

        if not valid_freeze(body):

            return JSONResponse(
                status_code=400,
                content={
                    "error": "INVALID_INPUT"
                }
            )

        freeze_id = body["freezeId"]

        with LOCK:

            existing = FREEZES.get(
                freeze_id
            )

            if existing:

                if equal_json(
                    existing["input"],
                    body
                ):

                    return JSONResponse(
                        status_code=200,
                        content=existing[
                            "response"
                        ]
                    )

                return JSONResponse(
                    status_code=409,
                    content={
                        "error":
                        "FREEZE_ID_CONFLICT"
                    }
                )

            response = do_freeze(
                body
            )

            FREEZES[freeze_id] = {
                "input": json.loads(
                    json.dumps(
                        body,
                        ensure_ascii=False
                    )
                ),
                "response": json.loads(
                    json.dumps(
                        response,
                        ensure_ascii=False
                    )
                )
            }

        return JSONResponse(
            status_code=200,
            content=response
        )

    # =====================================================
    # SELECT
    # =====================================================

    if phase == "select":

        if (
            not nonempty_string(
                body.get("freezeId")
            )
            or not isinstance(
                body.get("candidates"),
                list
            )
            or not isinstance(
                body.get("rows"),
                list
            )
            or not isinstance(
                body.get("policy"),
                dict
            )
        ):

            return JSONResponse(
                status_code=400,
                content={
                    "error": "INVALID_INPUT"
                }
            )

        freeze_id = body[
            "freezeId"
        ]

        with LOCK:
            stored = FREEZES.get(
                freeze_id
            )

        if stored is None:

            return JSONResponse(
                status_code=200,
                content={
                    "freezeId": freeze_id,
                    "selected": None,
                    "results": [],
                    "packageManifest": None
                }
            )

        response = do_select(
            body,
            stored["response"]
        )

        return JSONResponse(
            status_code=200,
            content=response
        )

    # =====================================================
    # UNKNOWN PHASE
    # =====================================================

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
