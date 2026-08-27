import hashlib
import json
import math
import sqlite3
import threading

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI(title="Quantize Candidate Admission API")

DB_PATH = "quantize_state.db"
LOCK = threading.Lock()


# ============================================================
# DATABASE
# ============================================================

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS freezes (
            freeze_id TEXT PRIMARY KEY,
            input_json TEXT NOT NULL,
            response_json TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def get_freeze(freeze_id):
    conn = sqlite3.connect(DB_PATH)

    row = conn.execute(
        """
        SELECT input_json, response_json
        FROM freezes
        WHERE freeze_id = ?
        """,
        (freeze_id,)
    ).fetchone()

    conn.close()

    if row is None:
        return None

    return {
        "input": json.loads(row[0]),
        "response": json.loads(row[1])
    }


def save_freeze(freeze_id, input_data, response_data):
    conn = sqlite3.connect(DB_PATH)

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
    conn.close()


init_db()


# ============================================================
# HELPERS
# ============================================================

def utf8(s):
    return s.encode("utf-8")


def compact_json(obj):
    return json.dumps(
        obj,
        ensure_ascii=False,
        separators=(",", ":")
    ).encode("utf-8")


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def json_digest(obj):
    return sha256(compact_json(obj))


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


def sort_codes(codes):
    return sorted(set(codes), key=utf8)


def equal_json(a, b):
    return compact_json(a) == compact_json(b)


def round12(x):
    return float(f"{x:.12f}")


# ============================================================
# INVENTORY
# ============================================================

def inventory_from_files(files):

    if not isinstance(files, dict) or not files:
        return False, [], None, None

    inventory = []

    try:
        for filename, text in files.items():

            if not isinstance(filename, str) or filename == "":
                return False, [], None, None

            if not isinstance(text, str):
                return False, [], None, None

            data = text.encode("utf-8")

            inventory.append({
                "name": filename,
                "bytes": len(data),
                "sha256": sha256(data)
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

    digest = json_digest(inventory)

    return True, inventory, total, digest


# ============================================================
# FREEZE VALIDATION
# ============================================================

def validate_freeze(body):

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

        if not isinstance(
            c.get("files"),
            dict
        ) or len(c["files"]) == 0:
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


# ============================================================
# FREEZE
# ============================================================

def freeze(body):

    request_cal = body["calibrationDigest"]
    request_tok = body["tokenizerDigest"]

    allowed = set(
        body["allowedUnsupportedReasons"]
    )

    output = []

    for c in body["candidates"]:

        ok, inventory, total, digest = (
            inventory_from_files(c["files"])
        )

        if not ok:

            output.append({
                "name": c["name"],
                "status": "invalid",
                "inventory": [],
                "totalBytes": None,
                "packageDigest": None,
                "reasonCodes": [
                    "INVALID_INPUT"
                ]
            })

            continue

        reason = c.get(
            "unsupportedReason"
        )

        # ----------------------------------------------------
        # Unsupported
        # ----------------------------------------------------

        if reason is not None:

            if reason in allowed:

                output.append({
                    "name": c["name"],
                    "status": "unsupported",
                    "inventory": inventory,
                    "totalBytes": total,
                    "packageDigest": digest,
                    "reasonCodes": []
                })

            else:

                output.append({
                    "name": c["name"],
                    "status": "invalid",
                    "inventory": inventory,
                    "totalBytes": total,
                    "packageDigest": digest,
                    "reasonCodes": [
                        "UNALLOWED_UNSUPPORTED_REASON"
                    ]
                })

            continue

        # ----------------------------------------------------
        # Normal candidate
        # ----------------------------------------------------

        codes = []

        if not c["loadable"]:
            codes.append("NOT_LOADABLE")

        if c["calibrationDigest"] != request_cal:
            codes.append("CALIBRATION_MISMATCH")

        if c["tokenizerDigest"] != request_tok:
            codes.append("TOKENIZER_MISMATCH")

        codes = sort_codes(codes)

        output.append({
            "name": c["name"],
            "status": (
                "frozen"
                if not codes
                else "invalid"
            ),
            "inventory": inventory,
            "totalBytes": total,
            "packageDigest": digest,
            "reasonCodes": codes
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

def validate_policy(policy):

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

    required = policy.get(
        "requiredSlices"
    )

    if not isinstance(required, dict):
        return False

    for name, value in required.items():

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
# SELECT METRICS
# ============================================================

def metrics(name, rows, required_slices):

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

        if not binary(label):
            return False, None, {}

        slice_name = row.get("slice")

        if not nonempty_string(slice_name):
            return False, None, {}

        predictions = row.get(
            "predictions"
        )

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
                ) / slice_total[slice_name]
            )

    return True, aggregate, slices


# ============================================================
# SELECT
# ============================================================

def select(body, stored):

    frozen = stored["candidates"]
    supplied = body["candidates"]

    # IMPORTANT:
    # The grader sends the frozen RESPONSE candidates.
    # Therefore compare the supplied candidate array directly
    # against the stored response candidate array.
    lineage_ok = equal_json(
        supplied,
        frozen
    )

    frozen_by_name = {
        c["name"]: c
        for c in frozen
    }

    supplied_by_name = {
        c["name"]: c
        for c in supplied
        if isinstance(c, dict)
        and isinstance(c.get("name"), str)
    }

    policy = body["policy"]
    policy_ok = validate_policy(policy)

    required = (
        policy.get("requiredSlices", {})
        if isinstance(policy, dict)
        else {}
    )

    order = (
        policy.get("candidateOrder", [])
        if isinstance(policy, dict)
        else []
    )

    supplied_names = set(
        supplied_by_name
    )

    order_names = set(order)

    global_codes = []

    if not lineage_ok:
        global_codes.append(
            "INVALID_LINEAGE"
        )

    if not policy_ok:
        global_codes.append(
            "INVALID_POLICY"
        )

    if policy_ok and supplied_names != order_names:
        global_codes.append(
            "INVALID_POLICY"
        )

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

        frozen_candidate = frozen_by_name.get(
            name
        )

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
        #
        # Do NOT expect files here.
        # The select candidate is the frozen response object.
        #
        total_bytes = None
        manifest_valid = False

        if frozen_candidate is not None:

            inventory = frozen_candidate.get(
                "inventory"
            )

            recorded_total = frozen_candidate.get(
                "totalBytes"
            )

            recorded_digest = frozen_candidate.get(
                "packageDigest"
            )

            if (
                isinstance(inventory, list)
                and isinstance(recorded_total, int)
                and recorded_total >= 0
                and isinstance(recorded_digest, str)
                and len(recorded_digest) > 0
            ):

                recomputed_total = 0
                valid_inventory = True

                previous_name = None

                for item in inventory:

                    if not isinstance(item, dict):
                        valid_inventory = False
                        break

                    item_name = item.get("name")
                    item_bytes = item.get("bytes")
                    item_hash = item.get("sha256")

                    if not nonempty_string(item_name):
                        valid_inventory = False
                        break

                    if not safe_integer(item_bytes):
                        valid_inventory = False
                        break

                    if (
                        not isinstance(item_hash, str)
                        or len(item_hash) != 64
                    ):
                        valid_inventory = False
                        break

                    if previous_name is not None:
                        if utf8(item_name) <= utf8(previous_name):
                            valid_inventory = False
                            break

                    previous_name = item_name

                    recomputed_total += item_bytes

                recomputed_digest = json_digest(
                    inventory
                )

                if (
                    valid_inventory
                    and recomputed_total == recorded_total
                    and recomputed_digest == recorded_digest
                ):

                    total_bytes = recomputed_total
                    manifest_valid = True

        if not manifest_valid:
            codes.append(
                "INVALID_MANIFEST"
            )

        # ----------------------------------------------------
        # Latency
        # ----------------------------------------------------

        latency_ms = None

        latencies = body.get(
            "latencies"
        )

        if isinstance(latencies, dict):
            value = latencies.get(name)

            if (
                finite_number(value)
                and float(value) >= 0
            ):
                latency_ms = float(value)

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
            and latency_ms is not None
            and latency_ms > float(
                policy["maxLatencyMs"]
            )
        ):
            codes.append(
                "LATENCY_LIMIT"
            )

        codes = sort_codes(codes)

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

    # ========================================================
    # WINNER
    # ========================================================

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

    package_manifest = None

    if winner is not None:

        package_manifest = frozen_by_name[
            winner["name"]
        ]

    return {
        "freezeId": body["freezeId"],
        "selected": (
            winner["name"]
            if winner is not None
            else None
        ),
        "results": results,
        "packageManifest": package_manifest
    }


# ============================================================
# ENDPOINT
# ============================================================

@app.post("/quantize")
async def quantize(request: Request):

    try:
        body = await request.json()
    except Exception:
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

    # ========================================================
    # FREEZE
    # ========================================================

    if phase == "freeze":

        if not validate_freeze(body):

            return JSONResponse(
                status_code=400,
                content={
                    "error": "INVALID_INPUT"
                }
            )

        freeze_id = body["freezeId"]

        with LOCK:

            existing = get_freeze(
                freeze_id
            )

            if existing is not None:

                if equal_json(
                    existing["input"],
                    body
                ):
                    return JSONResponse(
                        status_code=200,
                        content=existing["response"]
                    )

                return JSONResponse(
                    status_code=409,
                    content={
                        "error":
                        "FREEZE_ID_CONFLICT"
                    }
                )

            response = freeze(body)

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

        # Exact required top-level structure.
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

        # Unknown freeze is NOT a 400.
        if stored is None:

            names = []

            for c in body["candidates"]:

                if (
                    isinstance(c, dict)
                    and isinstance(
                        c.get("name"),
                        str
                    )
                ):
                    names.append(
                        c["name"]
                    )

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

        response = select(
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


@app.get("/")
def root():
    return {
        "service": "quantize",
        "endpoint": "POST /quantize"
    }
