import hashlib
import json
import math
import sqlite3
import threading
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict


app = FastAPI(
    title="Quantize Candidate Admission API",
    version="1.0"
)

DB_PATH = "quantize_state.db"
LOCK = threading.Lock()


# ============================================================
# OPENAPI REQUEST MODEL
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


def get_freeze(freeze_id):
    with sqlite3.connect(DB_PATH) as db:
        row = db.execute(
            """
            SELECT input_json, response_json
            FROM freezes
            WHERE freeze_id=?
            """,
            (freeze_id,)
        ).fetchone()

    if row is None:
        return None

    return {
        "input": json.loads(row[0]),
        "response": json.loads(row[1])
    }


def save_freeze(freeze_id, data, response):
    with sqlite3.connect(DB_PATH) as db:
        db.execute(
            """
            INSERT INTO freezes
            (freeze_id,input_json,response_json)
            VALUES (?,?,?)
            """,
            (
                freeze_id,
                json.dumps(
                    data,
                    ensure_ascii=False,
                    separators=(",", ":")
                ),
                json.dumps(
                    response,
                    ensure_ascii=False,
                    separators=(",", ":")
                )
            )
        )
        db.commit()


# ============================================================
# HELPERS
# ============================================================

def utf8(x):
    return x.encode("utf-8")


def compact_json(x):
    return json.dumps(
        x,
        ensure_ascii=False,
        separators=(",", ":")
    ).encode("utf-8")


def sha256_bytes(x):
    return hashlib.sha256(x).hexdigest()


def sha256_json(x):
    return sha256_bytes(compact_json(x))


def nonempty(x):
    return isinstance(x, str) and len(x) > 0


def finite(x):
    return (
        isinstance(x, (int, float))
        and not isinstance(x, bool)
        and math.isfinite(float(x))
    )


def safe_int(x):
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


def same_json(a, b):
    return compact_json(a) == compact_json(b)


# ============================================================
# INVENTORY
# ============================================================

def inventory(files):

    if not isinstance(files, dict) or not files:
        return False, [], None, None

    result = []

    try:
        for filename, content in files.items():

            if not isinstance(filename, str):
                return False, [], None, None

            if filename == "":
                return False, [], None, None

            if not isinstance(content, str):
                return False, [], None, None

            raw = content.encode("utf-8")

            result.append({
                "name": filename,
                "bytes": len(raw),
                "sha256": sha256_bytes(raw)
            })

    except Exception:
        return False, [], None, None

    result.sort(
        key=lambda x: utf8(x["name"])
    )

    total = sum(
        x["bytes"]
        for x in result
    )

    digest = sha256_json(result)

    return True, result, total, digest


# ============================================================
# FREEZE VALIDATION
# ============================================================

def validate_freeze(body):

    if body.get("phase") != "freeze":
        return False

    freeze_id = body.get("freezeId")

    if (
        not nonempty(freeze_id)
        or len(freeze_id) > 128
    ):
        return False

    if not nonempty(
        body.get("calibrationDigest")
    ):
        return False

    if not nonempty(
        body.get("tokenizerDigest")
    ):
        return False

    allowed = body.get(
        "allowedUnsupportedReasons"
    )

    if not isinstance(allowed, list):
        return False

    if any(
        not nonempty(x)
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

        if not nonempty(name):
            return False

        if name in names:
            return False

        names.add(name)

        if not isinstance(
            c.get("files"),
            dict
        ):
            return False

        if not c["files"]:
            return False

        if not isinstance(
            c.get("loadable"),
            bool
        ):
            return False

        if not nonempty(
            c.get("calibrationDigest")
        ):
            return False

        if not nonempty(
            c.get("tokenizerDigest")
        ):
            return False

        if "unsupportedReason" in c:

            reason = c["unsupportedReason"]

            if (
                reason is not None
                and not nonempty(reason)
            ):
                return False

    return True


# ============================================================
# FREEZE
# ============================================================

def freeze(body):

    allowed = set(
        body["allowedUnsupportedReasons"]
    )

    output = []

    for c in body["candidates"]:

        name = c["name"]

        ok, inv, total, digest = inventory(
            c["files"]
        )

        if not ok:

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

        reason = c.get(
            "unsupportedReason"
        )

        if reason is not None:

            if reason in allowed:

                output.append({
                    "name": name,
                    "status": "unsupported",
                    "inventory": inv,
                    "totalBytes": total,
                    "packageDigest": digest,
                    "reasonCodes": []
                })

            else:

                output.append({
                    "name": name,
                    "status": "invalid",
                    "inventory": inv,
                    "totalBytes": total,
                    "packageDigest": digest,
                    "reasonCodes": [
                        "UNALLOWED_UNSUPPORTED_REASON"
                    ]
                })

            continue

        codes = []

        if not c["loadable"]:
            codes.append("NOT_LOADABLE")

        if (
            c["calibrationDigest"]
            != body["calibrationDigest"]
        ):
            codes.append(
                "CALIBRATION_MISMATCH"
            )

        if (
            c["tokenizerDigest"]
            != body["tokenizerDigest"]
        ):
            codes.append(
                "TOKENIZER_MISMATCH"
            )

        codes = sort_codes(codes)

        output.append({
            "name": name,
            "status": (
                "frozen"
                if not codes
                else "invalid"
            ),
            "inventory": inv,
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

    if not safe_int(
        policy.get("maxBytes")
    ):
        return False

    floor = policy.get(
        "aggregateFloor"
    )

    if (
        not finite(floor)
        or not 0 <= float(floor) <= 1
    ):
        return False

    required = policy.get(
        "requiredSlices"
    )

    if not isinstance(required, dict):
        return False

    for name, value in required.items():

        if not nonempty(name):
            return False

        if (
            not finite(value)
            or not 0 <= float(value) <= 1
        ):
            return False

    max_latency = policy.get(
        "maxLatencyMs"
    )

    if (
        not finite(max_latency)
        or float(max_latency) < 0
    ):
        return False

    order = policy.get(
        "candidateOrder"
    )

    if not isinstance(order, list):
        return False

    if any(
        not nonempty(x)
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
    correct_slices = {}

    for row in rows:

        if not isinstance(row, dict):
            return False, None, {}

        label = row.get("label")

        if not binary(label):
            return False, None, {}

        slice_name = row.get("slice")

        if not nonempty(slice_name):
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

            correct_slices[slice_name] = (
                correct_slices.get(
                    slice_name,
                    0
                ) + 1
            )

    aggregate = round(
        correct / len(rows),
        12
    )

    slices = {}

    for name in required:

        if name in totals:

            slices[name] = round(
                correct_slices.get(
                    name,
                    0
                ) / totals[name],
                12
            )

    return True, aggregate, slices


# ============================================================
# SELECT
# ============================================================

def select(body, stored):

    frozen = stored["candidates"]

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

    global_codes = []

    if not same_json(
        supplied,
        frozen
    ):
        global_codes.append(
            "INVALID_LINEAGE"
        )

    policy_ok = validate_policy(policy)

    if not policy_ok:
        global_codes.append(
            "INVALID_POLICY"
        )

    order = (
        policy.get("candidateOrder", [])
        if policy_ok
        else []
    )

    if policy_ok and names != set(order):
        global_codes.append(
            "INVALID_POLICY"
        )

    required = (
        policy.get("requiredSlices", {})
        if policy_ok
        else {}
    )

    order_index = {
        name: i
        for i, name in enumerate(order)
    }

    names = sorted(
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

    for name in names:

        codes = list(global_codes)

        fc = frozen_map.get(name)

        aggregate = None
        slices = {}
        total = None
        latency = None

        if (
            fc is None
            or fc.get("status") != "frozen"
        ):
            codes.append(
                "NOT_FROZEN"
            )

        # Manifest validation
        if fc is None:

            codes.append(
                "INVALID_MANIFEST"
            )

        else:

            inv = fc.get("inventory")
            recorded_total = fc.get("totalBytes")
            recorded_digest = fc.get(
                "packageDigest"
            )

            manifest_ok = True

            if not isinstance(inv, list):
                manifest_ok = False

            if not safe_int(recorded_total):
                manifest_ok = False

            if (
                not isinstance(
                    recorded_digest,
                    str
                )
                or len(recorded_digest) != 64
            ):
                manifest_ok = False

            if manifest_ok:

                previous = None
                calculated_total = 0
                names_seen = set()

                for item in inv:

                    if not isinstance(item, dict):
                        manifest_ok = False
                        break

                    item_name = item.get("name")
                    item_bytes = item.get("bytes")
                    item_hash = item.get("sha256")

                    if not nonempty(item_name):
                        manifest_ok = False
                        break

                    if item_name in names_seen:
                        manifest_ok = False
                        break

                    names_seen.add(item_name)

                    if not safe_int(item_bytes):
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

                if sha256_json(inv) != recorded_digest:
                    manifest_ok = False

            if not manifest_ok:
                codes.append(
                    "INVALID_MANIFEST"
                )
            else:
                total = recorded_total

        # Latency
        latencies = body.get("latencies")

        if isinstance(latencies, dict):

            value = latencies.get(name)

            if (
                finite(value)
                and float(value) >= 0
            ):
                latency = float(value)

        # Predictions
        ok, aggregate, slices = metrics(
            name,
            body["rows"],
            required
        )

        if not ok:

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

        if (
            policy_ok
            and total is not None
            and total > policy["maxBytes"]
        ):
            codes.append(
                "SIZE_LIMIT"
            )

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

        codes = sort_codes(codes)

        results.append({
            "name": name,
            "aggregate": aggregate,
            "slices": slices,
            "totalBytes": total,
            "latencyMs": latency,
            "admitted": not codes,
            "reasonCodes": codes
        })

    winners = [
        x for x in results
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

@app.post(
    "/quantize",
    response_class=JSONResponse
)
async def quantize(
    request: QuantizeRequest
):

    # Pydantic gives Swagger a Request body,
    # but we convert back to plain JSON so that
    # the required protocol remains under our control.
    body = request.model_dump(
        exclude_none=True
    )

    print(
        "========== QUANTIZE =========="
    )
    print(
        json.dumps(
            body,
            ensure_ascii=False
        )
    )
    print(
        "=============================="
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

            print(
                "FREEZE INVALID INPUT"
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

        if (
            not nonempty(
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
            stored = get_freeze(freeze_id)

        if stored is None:

            names = []

            for c in body["candidates"]:

                if isinstance(c, dict):

                    name = c.get("name")

                    if isinstance(name, str):
                        names.append(name)

            names = sorted(
                set(names),
                key=utf8
            )

            results = [
                {
                    "name": name,
                    "aggregate": None,
                    "slices": {},
                    "totalBytes": None,
                    "latencyMs": None,
                    "admitted": False,
                    "reasonCodes": [
                        "NOT_FROZEN"
                    ]
                }
                for name in names
            ]

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
            content=select(
                body,
                stored["response"]
            )
        )

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
