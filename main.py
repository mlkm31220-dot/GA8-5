import hashlib
import json
import math
import os
import sqlite3
import threading

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI(
    title="Quantize Candidate Admission API",
    version="1.0.0"
)

DB_PATH = "quantize_state.db"
LOCK = threading.Lock()


# ============================================================
# DATABASE
# ============================================================

def init_db():
    with sqlite3.connect(DB_PATH) as db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS freezes (
                freeze_id TEXT PRIMARY KEY,
                input_json TEXT NOT NULL,
                response_json TEXT NOT NULL
            )
        """)
        db.commit()


init_db()


# ============================================================
# HELPERS
# ============================================================

def is_string(x):
    return isinstance(x, str)


def nonempty_string(x):
    return isinstance(x, str) and len(x) > 0


def utf8(x):
    return x.encode("utf-8")


def compact_json(x):
    return json.dumps(
        x,
        ensure_ascii=False,
        separators=(",", ":")
    ).encode("utf-8")


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_json(data):
    return sha256_bytes(compact_json(data))


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


def codes_sorted(codes):
    return sorted(set(codes), key=lambda x: utf8(x))


def same_json(a, b):
    return compact_json(a) == compact_json(b)


# ============================================================
# INVENTORY
# ============================================================

def build_inventory(files):

    if not isinstance(files, dict) or not files:
        return False, [], None, None

    inventory = []

    try:
        for filename, text in files.items():

            if not isinstance(filename, str):
                return False, [], None, None

            if filename == "":
                return False, [], None, None

            if not isinstance(text, str):
                return False, [], None, None

            raw = text.encode("utf-8")

            inventory.append({
                "name": filename,
                "bytes": len(raw),
                "sha256": sha256_bytes(raw)
            })

    except (UnicodeEncodeError, TypeError):
        return False, [], None, None

    inventory.sort(
        key=lambda x: utf8(x["name"])
    )

    total = sum(
        x["bytes"]
        for x in inventory
    )

    digest = sha256_json(inventory)

    return True, inventory, total, digest


# ============================================================
# FREEZE VALIDATION
# ============================================================

def valid_freeze(body):

    if not isinstance(body, dict):
        return False

    if body.get("phase") != "freeze":
        return False

    freeze_id = body.get("freezeId")

    if (
        not isinstance(freeze_id, str)
        or not freeze_id
        or len(freeze_id) > 128
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

    candidates = body.get("candidates")

    if (
        not isinstance(candidates, list)
        or len(candidates) == 0
    ):
        return False

    names = []

    for candidate in candidates:

        if not isinstance(candidate, dict):
            return False

        name = candidate.get("name")

        if not nonempty_string(name):
            return False

        names.append(name)

        files = candidate.get("files")

        if not isinstance(files, dict):
            return False

        if len(files) == 0:
            return False

        if not isinstance(
            candidate.get("loadable"),
            bool
        ):
            return False

        if not nonempty_string(
            candidate.get("calibrationDigest")
        ):
            return False

        if not nonempty_string(
            candidate.get("tokenizerDigest")
        ):
            return False

        if "unsupportedReason" in candidate:

            reason = candidate["unsupportedReason"]

            if (
                reason is not None
                and not nonempty_string(reason)
            ):
                return False

    if len(names) != len(set(names)):
        return False

    return True


# ============================================================
# FREEZE
# ============================================================

def do_freeze(body):

    request_cal = body["calibrationDigest"]
    request_tok = body["tokenizerDigest"]

    allowed = set(
        body["allowedUnsupportedReasons"]
    )

    output = []

    for candidate in body["candidates"]:

        name = candidate["name"]

        ok, inventory, total, digest = build_inventory(
            candidate["files"]
        )

        if not ok:

            output.append({
                "name": name,
                "status": "invalid",
                "inventory": [],
                "totalBytes": None,
                "packageDigest": None,
                "reasonCodes": ["INVALID_INPUT"]
            })

            continue

        reason = candidate.get(
            "unsupportedReason"
        )

        # Unsupported candidate
        if reason is not None:

            if reason in allowed:

                output.append({
                    "name": name,
                    "status": "unsupported",
                    "inventory": inventory,
                    "totalBytes": total,
                    "packageDigest": digest,
                    "reasonCodes": []
                })

            else:

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

            continue

        reasons = []

        if not candidate["loadable"]:
            reasons.append("NOT_LOADABLE")

        if (
            candidate["calibrationDigest"]
            != request_cal
        ):
            reasons.append(
                "CALIBRATION_MISMATCH"
            )

        if (
            candidate["tokenizerDigest"]
            != request_tok
        ):
            reasons.append(
                "TOKENIZER_MISMATCH"
            )

        reasons = codes_sorted(reasons)

        output.append({
            "name": name,
            "status": (
                "frozen"
                if not reasons
                else "invalid"
            ),
            "inventory": inventory,
            "totalBytes": total,
            "packageDigest": digest,
            "reasonCodes": reasons
        })

    output.sort(
        key=lambda x: utf8(x["name"])
    )

    return {
        "freezeId": body["freezeId"],
        "candidates": output
    }


# ============================================================
# POLICY
# ============================================================

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

    if len(slices) != len(set(slices)):
        return False

    for name, value in slices.items():

        if not nonempty_string(name):
            return False

        if (
            not finite_number(value)
            or not 0 <= float(value) <= 1
        ):
            return False

    latency = policy.get(
        "maxLatencyMs"
    )

    if (
        not finite_number(latency)
        or float(latency) < 0
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


# ============================================================
# METRICS
# ============================================================

def metrics(name, rows, required):

    if not isinstance(rows, list) or not rows:
        return False, None, {}

    correct = 0
    totals = {}
    correct_by_slice = {}

    for row in rows:

        if not isinstance(row, dict):
            return False, None, {}

        label = row.get("label")

        if not binary(label):
            return False, None, {}

        slice_name = row.get("slice")

        if not nonempty_string(slice_name):
            return False, None, {}

        predictions = row.get("predictions")

        if not isinstance(predictions, dict):
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

            correct_by_slice[slice_name] = (
                correct_by_slice.get(slice_name, 0) + 1
            )

    aggregate = round(
        correct / len(rows),
        12
    )

    slices = {}

    for slice_name in required:

        if slice_name in totals:

            slices[slice_name] = round(
                correct_by_slice.get(
                    slice_name,
                    0
                ) / totals[slice_name],
                12
            )

    return True, aggregate, slices


# ============================================================
# SELECT
# ============================================================

def do_select(body, frozen_response):

    frozen = frozen_response["candidates"]
    supplied = body["candidates"]
    policy = body["policy"]

    frozen_map = {
        x["name"]: x
        for x in frozen
    }

    supplied_map = {}

    for x in supplied:

        if isinstance(x, dict):
            name = x.get("name")

            if isinstance(name, str):
                supplied_map[name] = x

    names = set(supplied_map)
    order = policy.get(
        "candidateOrder",
        []
    )

    global_codes = []

    if not same_json(
        supplied,
        frozen
    ):
        global_codes.append(
            "INVALID_LINEAGE"
        )

    policy_ok = valid_policy(policy)

    if not policy_ok:
        global_codes.append(
            "INVALID_POLICY"
        )

    if policy_ok and names != set(order):
        global_codes.append(
            "INVALID_POLICY"
        )

    required = (
        policy.get(
            "requiredSlices",
            {}
        )
        if policy_ok
        else {}
    )

    order_index = {
        x: i
        for i, x in enumerate(order)
    }

    result_names = sorted(
        names,
        key=lambda x: (
            order_index.get(
                x,
                len(order)
            ),
            utf8(x)
        )
    )

    results = []

    for name in result_names:

        codes = list(global_codes)

        frozen_candidate = frozen_map.get(name)

        aggregate = None
        slices = {}
        total_bytes = None
        latency = None

        # ----------------------------------------------------
        # Frozen
        # ----------------------------------------------------

        if (
            frozen_candidate is None
            or frozen_candidate.get("status")
            != "frozen"
        ):
            codes.append(
                "NOT_FROZEN"
            )

        # ----------------------------------------------------
        # Manifest
        # ----------------------------------------------------

        if frozen_candidate is None:

            codes.append(
                "INVALID_MANIFEST"
            )

        else:

            inventory = frozen_candidate.get(
                "inventory"
            )

            recorded_total = frozen_candidate.get(
                "totalBytes"
            )

            recorded_digest = frozen_candidate.get(
                "packageDigest"
            )

            manifest_ok = True

            if not isinstance(inventory, list):
                manifest_ok = False

            if not safe_integer(recorded_total):
                manifest_ok = False

            if (
                not isinstance(recorded_digest, str)
                or len(recorded_digest) != 64
            ):
                manifest_ok = False

            if manifest_ok:

                previous = None
                calculated_total = 0

                for item in inventory:

                    if not isinstance(item, dict):
                        manifest_ok = False
                        break

                    item_name = item.get("name")
                    item_bytes = item.get("bytes")
                    item_hash = item.get("sha256")

                    if not nonempty_string(item_name):
                        manifest_ok = False
                        break

                    if not safe_integer(item_bytes):
                        manifest_ok = False
                        break

                    if (
                        not isinstance(item_hash, str)
                        or len(item_hash) != 64
                    ):
                        manifest_ok = False
                        break

                    if (
                        previous is not None
                        and utf8(item_name)
                        <= utf8(previous)
                    ):
                        manifest_ok = False
                        break

                    previous = item_name
                    calculated_total += item_bytes

                if calculated_total != recorded_total:
                    manifest_ok = False

                if sha256_json(inventory) != recorded_digest:
                    manifest_ok = False

            if not manifest_ok:
                codes.append(
                    "INVALID_MANIFEST"
                )
            else:
                total_bytes = recorded_total

        # ----------------------------------------------------
        # Latency
        # ----------------------------------------------------

        latencies = body.get("latencies")

        if isinstance(latencies, dict):

            value = latencies.get(name)

            if (
                finite_number(value)
                and float(value) >= 0
            ):
                latency = float(value)

        # ----------------------------------------------------
        # Predictions
        # ----------------------------------------------------

        prediction_ok, aggregate, slices = metrics(
            name,
            body["rows"],
            required
        )

        if not prediction_ok:

            aggregate = None
            slices = {}

            codes.append(
                "INVALID_PREDICTIONS"
            )

        elif policy_ok:

            if aggregate < float(
                policy["aggregateFloor"]
            ):
                codes.append(
                    "AGGREGATE_FLOOR"
                )

            for slice_name, floor in required.items():

                if slice_name not in slices:

                    codes.append(
                        "MISSING_SLICE:" + slice_name
                    )

                elif slices[slice_name] < float(floor):

                    codes.append(
                        "SLICE_FLOOR:" + slice_name
                    )

        # ----------------------------------------------------
        # Size
        # ----------------------------------------------------

        if (
            policy_ok
            and total_bytes is not None
            and total_bytes > policy["maxBytes"]
        ):
            codes.append(
                "SIZE_LIMIT"
            )

        # ----------------------------------------------------
        # Latency
        # ----------------------------------------------------

        if (
            policy_ok
            and latency is not None
            and latency > float(
                policy["maxLatencyMs"]
            )
        ):
            codes.append(
                "LATENCY_LIMIT"
            )

        codes = codes_sorted(codes)

        results.append({
            "name": name,
            "aggregate": aggregate,
            "slices": slices,
            "totalBytes": total_bytes,
            "latencyMs": latency,
            "admitted": len(codes) == 0,
            "reasonCodes": codes
        })

    admitted = [
        x for x in results
        if x["admitted"]
    ]

    winner = None

    if admitted:

        winner = min(
            admitted,
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
        manifest = frozen_map[
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


# ============================================================
# POST /quantize
# ============================================================

@app.post("/quantize")
async def quantize(request: Request):

    try:
        raw = await request.body()

        print("========== QUANTIZE REQUEST ==========")
        print(raw.decode("utf-8"))
        print("=======================================")

        body = json.loads(
            raw.decode("utf-8")
        )

    except Exception as e:

        print("JSON ERROR:", repr(e))

        return JSONResponse(
            status_code=400,
            content={
                "error": "INVALID_INPUT"
            }
        )

    if not isinstance(body, dict):

        return JSONResponse(
            status_code=400,
            content={
                "error": "INVALID_INPUT"
            }
        )

    phase = body.get("phase")

    print("PHASE:", repr(phase))

    # ========================================================
    # FREEZE
    # ========================================================

    if phase == "freeze":

        if not valid_freeze(body):

            print(
                "FREEZE VALIDATION FAILED"
            )

            return JSONResponse(
                status_code=400,
                content={
                    "error": "INVALID_INPUT"
                }
            )

        freeze_id = body["freezeId"]

        with LOCK:

            old = get_freeze(freeze_id)

            if old is not None:

                if same_json(
                    old["input"],
                    body
                ):
                    return JSONResponse(
                        status_code=200,
                        content=old["response"]
                    )

                return JSONResponse(
                    status_code=409,
                    content={
                        "error":
                        "FREEZE_ID_CONFLICT"
                    }
                )

            response = do_freeze(body)

            save_freeze(
                freeze_id,
                body,
                response
            )

        return JSONResponse(
            status_code=200,
            content=response
        )

    # ========================================================
    # SELECT
    # ========================================================

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

        freeze_id = body["freezeId"]

        with LOCK:
            stored = get_freeze(
                freeze_id
            )

        if stored is None:

            names = []

            for candidate in body["candidates"]:

                if isinstance(candidate, dict):

                    name = candidate.get("name")

                    if isinstance(name, str):
                        names.append(name)

            names = sorted(
                set(names),
                key=utf8
            )

            results = []

            for name in names:

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

            return JSONResponse(
                status_code=200,
                content={
                    "freezeId": freeze_id,
                    "selected": None,
                    "results": results,
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

    # ========================================================
    # INVALID PHASE
    # ========================================================

    return JSONResponse(
        status_code=400,
        content={
            "error": "INVALID_INPUT"
        }
    )


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():
    return {
        "service": "quantize",
        "endpoint": "POST /quantize",
        "docs": "/docs",
        "openapi": "/openapi.json"
    }


@app.get("/health")
def health():
    return {
        "status": "ok"
    }
