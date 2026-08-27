import hashlib
import json
import math
import sqlite3
import threading
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI(title="Quantize Candidate Admission API", version="1.0")

DB_PATH = "quantize_state.db"
DB_LOCK = threading.Lock()
MAX_SAFE_INTEGER = 9007199254740991


# ============================================================
# DATABASE
# ============================================================

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS freezes (
                freeze_id TEXT PRIMARY KEY,
                input_json TEXT NOT NULL,
                response_json TEXT NOT NULL
            )
        """)
        conn.commit()


def get_freeze(freeze_id):
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT input_json, response_json FROM freezes WHERE freeze_id = ?",
            (freeze_id,)
        ).fetchone()

    if row is None:
        return None

    return {
        "input": json.loads(row[0]),
        "response": json.loads(row[1])
    }


def save_freeze(freeze_id, input_data, response_data):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO freezes
            (freeze_id, input_json, response_json)
            VALUES (?, ?, ?)
            """,
            (
                freeze_id,
                json.dumps(
                    input_data,
                    ensure_ascii=False,
                    separators=(",", ":")
                ),
                json.dumps(
                    response_data,
                    ensure_ascii=False,
                    separators=(",", ":")
                )
            )
        )
        conn.commit()


init_db()


# ============================================================
# HELPERS
# ============================================================

def utf8(value):
    return value.encode("utf-8")


def compact_json(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":")
    ).encode("utf-8")


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_json(value):
    return sha256_bytes(compact_json(value))


def nonempty_string(value):
    return isinstance(value, str) and len(value) > 0


def finite_number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def safe_integer(value):
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 0 <= value <= MAX_SAFE_INTEGER
    )


def binary(value):
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value in (0, 1)
    )


def round12(value):
    return float(f"{value:.12f}")


def code_sort(c):
    return utf8(c)


def codes_unique(codes):
    return sorted(set(codes), key=code_sort)


def json_equal(a, b):
    return compact_json(a) == compact_json(b)


# ============================================================
# FILE INVENTORY
# ============================================================

def make_inventory(files):
    if not isinstance(files, dict) or not files:
        return False, [], None, None

    inventory = []
    filenames = set()

    try:
        for filename, content in files.items():

            if not isinstance(filename, str) or filename == "":
                return False, [], None, None

            if filename in filenames:
                return False, [], None, None

            filenames.add(filename)

            if not isinstance(content, str):
                return False, [], None, None

            data = content.encode("utf-8")

            inventory.append({
                "name": filename,
                "bytes": len(data),
                "sha256": sha256_bytes(data)
            })

    except (UnicodeEncodeError, TypeError):
        return False, [], None, None

    inventory.sort(key=lambda x: utf8(x["name"]))

    total = sum(x["bytes"] for x in inventory)

    digest = sha256_json(inventory)

    return True, inventory, total, digest


# ============================================================
# FREEZE VALIDATION
# ============================================================

def validate_freeze(body):

    if body.get("phase") != "freeze":
        return False

    freeze_id = body.get("freezeId")

    if (
        not isinstance(freeze_id, str)
        or not freeze_id
        or len(freeze_id) > 128
    ):
        return False

    if not nonempty_string(body.get("calibrationDigest")):
        return False

    if not nonempty_string(body.get("tokenizerDigest")):
        return False

    allowed = body.get("allowedUnsupportedReasons")

    if not isinstance(allowed, list):
        return False

    seen_allowed = set()

    for reason in allowed:
        if not nonempty_string(reason):
            return False
        if reason in seen_allowed:
            return False
        seen_allowed.add(reason)

    candidates = body.get("candidates")

    if not isinstance(candidates, list) or len(candidates) == 0:
        return False

    names = set()

    for candidate in candidates:

        if not isinstance(candidate, dict):
            return False

        name = candidate.get("name")

        if not nonempty_string(name):
            return False

        if name in names:
            return False

        names.add(name)

        files = candidate.get("files")

        if not isinstance(files, dict) or len(files) == 0:
            return False

        # Validate filenames/content here.
        files_ok, _, _, _ = make_inventory(files)

        if not files_ok:
            return False

        if not isinstance(candidate.get("loadable"), bool):
            return False

        if not nonempty_string(candidate.get("calibrationDigest")):
            return False

        if not nonempty_string(candidate.get("tokenizerDigest")):
            return False

        if "unsupportedReason" in candidate:
            reason = candidate.get("unsupportedReason")

            if reason is not None and not nonempty_string(reason):
                return False

    return True


# ============================================================
# FREEZE
# ============================================================

def do_freeze(body):

    request_cal = body["calibrationDigest"]
    request_tok = body["tokenizerDigest"]

    allowed = set(body["allowedUnsupportedReasons"])

    output = []

    for candidate in body["candidates"]:

        name = candidate["name"]

        files_ok, inventory, total, digest = make_inventory(
            candidate["files"]
        )

        if not files_ok:
            output.append({
                "name": name,
                "status": "invalid",
                "inventory": [],
                "totalBytes": None,
                "packageDigest": None,
                "reasonCodes": ["INVALID_INPUT"]
            })
            continue

        reason = candidate.get("unsupportedReason")

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

        if candidate["calibrationDigest"] != request_cal:
            reasons.append("CALIBRATION_MISMATCH")

        if candidate["tokenizerDigest"] != request_tok:
            reasons.append("TOKENIZER_MISMATCH")

        reasons = codes_unique(reasons)

        output.append({
            "name": name,
            "status": "frozen" if not reasons else "invalid",
            "inventory": inventory,
            "totalBytes": total,
            "packageDigest": digest,
            "reasonCodes": reasons
        })

    output.sort(key=lambda x: utf8(x["name"]))

    return {
        "freezeId": body["freezeId"],
        "candidates": output
    }


# ============================================================
# POLICY
# ============================================================

def validate_policy(policy):

    if not isinstance(policy, dict):
        return False

    if not safe_integer(policy.get("maxBytes")):
        return False

    floor = policy.get("aggregateFloor")

    if (
        not finite_number(floor)
        or not 0 <= float(floor) <= 1
    ):
        return False

    required = policy.get("requiredSlices")

    if not isinstance(required, dict):
        return False

    slice_names = set()

    for name, value in required.items():

        if not nonempty_string(name):
            return False

        if name in slice_names:
            return False

        slice_names.add(name)

        if (
            not finite_number(value)
            or not 0 <= float(value) <= 1
        ):
            return False

    latency = policy.get("maxLatencyMs")

    if (
        not finite_number(latency)
        or float(latency) < 0
    ):
        return False

    order = policy.get("candidateOrder")

    if not isinstance(order, list):
        return False

    seen = set()

    for name in order:
        if not nonempty_string(name):
            return False
        if name in seen:
            return False
        seen.add(name)

    return True


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(name, rows, required_slices):

    if not isinstance(rows, list) or len(rows) == 0:
        return False, None, {}

    correct = 0

    slice_total = {}
    slice_correct = {}

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

        slice_total[slice_name] = (
            slice_total.get(slice_name, 0) + 1
        )

        if prediction == label:
            correct += 1
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


# ============================================================
# SELECT
# ============================================================

def do_select(body, stored):

    frozen = stored["candidates"]
    supplied = body["candidates"]
    policy = body["policy"]

    frozen_by_name = {
        c["name"]: c
        for c in frozen
    }

    supplied_by_name = {}

    for candidate in supplied:
        if isinstance(candidate, dict):
            name = candidate.get("name")
            if isinstance(name, str):
                supplied_by_name[name] = candidate

    frozen_names = set(frozen_by_name)
    supplied_names = set(supplied_by_name)

    global_codes = []

    # Exact frozen candidate array required.
    if not json_equal(supplied, frozen):
        global_codes.append("INVALID_LINEAGE")

    policy_ok = validate_policy(policy)

    if not policy_ok:
        global_codes.append("INVALID_POLICY")

    if policy_ok:

        order = policy["candidateOrder"]

        if supplied_names != set(order):
            global_codes.append("INVALID_POLICY")

    else:
        order = []

    required_slices = (
        policy.get("requiredSlices", {})
        if isinstance(policy, dict)
        else {}
    )

    order_index = {
        name: i
        for i, name in enumerate(order)
    }

    result_names = list(supplied_names)

    result_names.sort(
        key=lambda name: (
            order_index.get(name, len(order)),
            utf8(name)
        )
    )

    results = []

    for name in result_names:

        codes = list(global_codes)

        frozen_candidate = frozen_by_name.get(name)
        submitted_candidate = supplied_by_name.get(name)

        aggregate = None
        slices = {}
        total_bytes = None
        latency_ms = None

        # ----------------------------------------------------
        # Frozen status
        # ----------------------------------------------------

        if (
            frozen_candidate is None
            or frozen_candidate.get("status") != "frozen"
        ):
            codes.append("NOT_FROZEN")

        # ----------------------------------------------------
        # Manifest
        # ----------------------------------------------------

        if submitted_candidate is None:

            codes.append("INVALID_MANIFEST")

        else:

            files = submitted_candidate.get("files")

            ok, inventory, total, digest = make_inventory(files)

            if not ok:

                codes.append("INVALID_MANIFEST")

            else:

                total_bytes = total

                if frozen_candidate is None:
                    codes.append("INVALID_MANIFEST")

                else:

                    if inventory != frozen_candidate.get("inventory"):
                        codes.append("INVALID_MANIFEST")

                    if total != frozen_candidate.get("totalBytes"):
                        codes.append("INVALID_MANIFEST")

                    if digest != frozen_candidate.get("packageDigest"):
                        codes.append("INVALID_MANIFEST")

        # ----------------------------------------------------
        # Latency
        # ----------------------------------------------------

        latencies = body.get("latencies")

        if (
            isinstance(latencies, dict)
            and name in latencies
            and finite_number(latencies[name])
            and float(latencies[name]) >= 0
        ):
            latency_ms = float(latencies[name])

        # ----------------------------------------------------
        # Predictions
        # ----------------------------------------------------

        predictions_ok, aggregate, slices = calculate_metrics(
            name,
            body.get("rows"),
            required_slices
        )

        if not predictions_ok:

            aggregate = None
            slices = {}
            codes.append("INVALID_PREDICTIONS")

        elif policy_ok:

            if aggregate < float(policy["aggregateFloor"]):
                codes.append("AGGREGATE_FLOOR")

            for slice_name, floor in required_slices.items():

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

        if policy_ok and total_bytes is not None:

            if total_bytes > policy["maxBytes"]:
                codes.append("SIZE_LIMIT")

        # ----------------------------------------------------
        # Latency
        # ----------------------------------------------------

        if policy_ok and latency_ms is not None:

            if latency_ms > float(policy["maxLatencyMs"]):
                codes.append("LATENCY_LIMIT")

        codes = codes_unique(codes)

        admitted = len(codes) == 0

        results.append({
            "name": name,
            "aggregate": aggregate,
            "slices": slices,
            "totalBytes": total_bytes,
            "latencyMs": latency_ms,
            "admitted": admitted,
            "reasonCodes": codes
        })

    # --------------------------------------------------------
    # Winner
    # --------------------------------------------------------

    admitted = [
        r for r in results
        if r["admitted"]
    ]

    winner = None

    if admitted:

        winner = min(
            admitted,
            key=lambda r: (
                r["totalBytes"],
                r["latencyMs"],
                order_index.get(
                    r["name"],
                    len(order)
                ),
                utf8(r["name"])
            )
        )

    manifest = None

    if winner is not None:
        manifest = frozen_by_name[winner["name"]]

    return {
        "freezeId": body["freezeId"],
        "selected": (
            winner["name"]
            if winner is not None
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
        body = await request.json()
    except Exception:
        return JSONResponse(
            status_code=400,
            content={"error": "INVALID_INPUT"}
        )

    if not isinstance(body, dict):
        return JSONResponse(
            status_code=400,
            content={"error": "INVALID_INPUT"}
        )

    phase = body.get("phase")

    # ========================================================
    # FREEZE
    # ========================================================

    if phase == "freeze":

        if not validate_freeze(body):

            return JSONResponse(
                status_code=400,
                content={"error": "INVALID_INPUT"}
            )

        freeze_id = body["freezeId"]

        with DB_LOCK:

            existing = get_freeze(freeze_id)

            if existing is not None:

                if json_equal(existing["input"], body):
                    return JSONResponse(
                        status_code=200,
                        content=existing["response"]
                    )

                return JSONResponse(
                    status_code=409,
                    content={
                        "error": "FREEZE_ID_CONFLICT"
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

        # Required by specification:
        # freezeId + candidates array + rows array + policy object
        if (
            not nonempty_string(body.get("freezeId"))
            or not isinstance(body.get("candidates"), list)
            or not isinstance(body.get("rows"), list)
            or not isinstance(body.get("policy"), dict)
        ):
            return JSONResponse(
                status_code=400,
                content={"error": "INVALID_INPUT"}
            )

        freeze_id = body["freezeId"]

        with DB_LOCK:
            stored = get_freeze(freeze_id)

        # Unknown freeze is a valid select request.
        if stored is None:

            names = []

            for candidate in body["candidates"]:
                if isinstance(candidate, dict):
                    name = candidate.get("name")
                    if isinstance(name, str) and name:
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
                    "reasonCodes": ["NOT_FROZEN"]
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

        return JSONResponse(
            status_code=200,
            content=do_select(
                body,
                stored["response"]
            )
        )

    # ========================================================
    # INVALID PHASE
    # ========================================================

    return JSONResponse(
        status_code=400,
        content={"error": "INVALID_INPUT"}
    )


# ============================================================
# ROOT / HEALTH
# ============================================================

@app.get("/")
def root():
    return {
        "service": "quantize",
        "endpoint": "POST /quantize"
    }


@app.get("/health")
def health():
    return {"status": "ok"}
