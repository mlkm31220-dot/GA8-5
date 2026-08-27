import hashlib
import json
import math
import sqlite3
import threading

from typing import Any

from fastapi import FastAPI, Body, Request
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


def nonempty_string(value):
    return (
        isinstance(value, str)
        and len(value) > 0
    )


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
        and 0 <= value <= 9007199254740991
    )


def binary(value):
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value in (0, 1)
    )


def round12(value):
    return float(f"{value:.12f}")


def equal_json(a, b):
    return compact_json(a) == compact_json(b)


# ============================================================
# INVENTORY
# ============================================================

def build_inventory(files):

    if (
        not isinstance(files, dict)
        or len(files) == 0
    ):
        return False, [], None, None

    inventory = []

    filenames = set()

    for filename, content in files.items():

        if (
            not isinstance(filename, str)
            or filename == ""
        ):
            return False, [], None, None

        if filename in filenames:
            return False, [], None, None

        filenames.add(filename)

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
        key=lambda item: utf8(item["name"])
    )

    total_bytes = sum(
        item["bytes"]
        for item in inventory
    )

    package_digest = sha256(
        compact_json(inventory)
    )

    return (
        True,
        inventory,
        total_bytes,
        package_digest
    )


# ============================================================
# FREEZE INPUT VALIDATION
# ============================================================

def valid_freeze(body):

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

        if (
            not isinstance(files, dict)
            or len(files) == 0
        ):
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

            reason = candidate.get(
                "unsupportedReason"
            )

            if (
                reason is not None
                and not nonempty_string(reason)
            ):
                return False

    return True


# ============================================================
# FREEZE
# ============================================================

def do_freeze(body):

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

        unsupported_reason = candidate.get(
            "unsupportedReason"
        )

        # ----------------------------------------------------
        # Unsupported candidate
        # ----------------------------------------------------

        if unsupported_reason is not None:

            if unsupported_reason in allowed:

                results.append({
                    "name": name,
                    "status": "unsupported",
                    "inventory": inventory,
                    "totalBytes": total_bytes,
                    "packageDigest": package_digest,
                    "reasonCodes": []
                })

            else:

                results.append({
                    "name": name,
                    "status": "invalid",
                    "inventory": inventory,
                    "totalBytes": total_bytes,
                    "packageDigest": package_digest,
                    "reasonCodes": [
                        "UNALLOWED_UNSUPPORTED_REASON"
                    ]
                })

            continue

        # ----------------------------------------------------
        # Normal candidate
        # ----------------------------------------------------

        reason_codes = []

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

        reason_codes = sort_codes(
            reason_codes
        )

        results.append({
            "name": name,
            "status": (
                "frozen"
                if len(reason_codes) == 0
                else "invalid"
            ),
            "inventory": inventory,
            "totalBytes": total_bytes,
            "packageDigest": package_digest,
            "reasonCodes": reason_codes
        })

    results.sort(
        key=lambda item: utf8(
            item["name"]
        )
    )

    return {
        "freezeId": body["freezeId"],
        "candidates": results
    }


# ============================================================
# POLICY
# ============================================================

def valid_policy(policy):

    if not isinstance(policy, dict):
        return False

    max_bytes = policy.get(
        "maxBytes"
    )

    if not safe_integer(max_bytes):
        return False

    aggregate_floor = policy.get(
        "aggregateFloor"
    )

    if (
        not finite_number(aggregate_floor)
        or not 0 <= float(aggregate_floor) <= 1
    ):
        return False

    required_slices = policy.get(
        "requiredSlices"
    )

    if not isinstance(
        required_slices,
        dict
    ):
        return False

    slice_names = set()

    for name, floor in required_slices.items():

        if not nonempty_string(name):
            return False

        if name in slice_names:
            return False

        slice_names.add(name)

        if (
            not finite_number(floor)
            or not 0 <= float(floor) <= 1
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

    candidate_order = policy.get(
        "candidateOrder"
    )

    if not isinstance(
        candidate_order,
        list
    ):
        return False

    if any(
        not nonempty_string(name)
        for name in candidate_order
    ):
        return False

    if len(candidate_order) != len(
        set(candidate_order)
    ):
        return False

    return True


# ============================================================
# PREDICTIONS
# ============================================================

def calculate_metrics(
    candidate_name,
    rows,
    required_slices
):

    if (
        not isinstance(rows, list)
        or len(rows) == 0
    ):
        return False, None, {}

    total_correct = 0

    slice_total = {}
    slice_correct = {}

    for row in rows:

        if not isinstance(row, dict):
            return False, None, {}

        label = row.get(
            "label"
        )

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

        if candidate_name not in predictions:
            return False, None, {}

        prediction = predictions[
            candidate_name
        ]

        if not binary(prediction):
            return False, None, {}

        slice_total[slice_name] = (
            slice_total.get(
                slice_name,
                0
            ) + 1
        )

        if prediction == label:

            total_correct += 1

            slice_correct[slice_name] = (
                slice_correct.get(
                    slice_name,
                    0
                ) + 1
            )

    aggregate = round12(
        total_correct / len(rows)
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
# SELECT
# ============================================================

def do_select(body, frozen_response):

    frozen_candidates = (
        frozen_response["candidates"]
    )

    supplied_candidates = body[
        "candidates"
    ]

    frozen_by_name = {
        candidate["name"]: candidate
        for candidate in frozen_candidates
    }

    supplied_by_name = {}

    for candidate in supplied_candidates:

        if not isinstance(
            candidate,
            dict
        ):
            continue

        name = candidate.get(
            "name"
        )

        if isinstance(name, str):
            supplied_by_name[name] = candidate

    frozen_names = set(
        frozen_by_name
    )

    supplied_names = set(
        supplied_by_name
    )

    policy = body[
        "policy"
    ]

    policy_valid = valid_policy(
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

    global_codes = []

    if supplied_names != frozen_names:
        global_codes.append(
            "INVALID_LINEAGE"
        )

    if not policy_valid:
        global_codes.append(
            "INVALID_POLICY"
        )

    if (
        policy_valid
        and supplied_names != set(
            candidate_order
        )
    ):
        global_codes.append(
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
            utf8(name)
        )
    )

    results = []

    for name in names:

        codes = list(
            global_codes
        )

        frozen_candidate = (
            frozen_by_name.get(name)
        )

        supplied_candidate = (
            supplied_by_name.get(name)
        )

        aggregate = None
        slices = {}

        total_bytes = None
        latency_ms = None

        # ----------------------------------------------------
        # Frozen status
        # ----------------------------------------------------

        if (
            frozen_candidate is None
            or frozen_candidate.get(
                "status"
            ) != "frozen"
        ):
            codes.append(
                "NOT_FROZEN"
            )

        # ----------------------------------------------------
        # Manifest
        # ----------------------------------------------------

        if supplied_candidate is None:

            codes.append(
                "INVALID_MANIFEST"
            )

        else:

            files = supplied_candidate.get(
                "files"
            )

            ok, inventory, total, digest = (
                build_inventory(files)
            )

            if not ok:

                codes.append(
                    "INVALID_MANIFEST"
                )

            else:

                if (
                    frozen_candidate is None
                    or inventory != frozen_candidate.get(
                        "inventory"
                    )
                    or total != frozen_candidate.get(
                        "totalBytes"
                    )
                    or digest != frozen_candidate.get(
                        "packageDigest"
                    )
                ):
                    codes.append(
                        "INVALID_MANIFEST"
                    )

                total_bytes = total

        # ----------------------------------------------------
        # Latency
        # ----------------------------------------------------

        latencies = body.get(
            "latencies"
        )

        if (
            isinstance(latencies, dict)
            and name in latencies
            and finite_number(
                latencies[name]
            )
            and float(latencies[name]) >= 0
        ):

            latency_ms = float(
                latencies[name]
            )

        # ----------------------------------------------------
        # Predictions
        # ----------------------------------------------------

        valid_predictions, aggregate, slices = (
            calculate_metrics(
                name,
                body["rows"],
                required_slices
            )
        )

        if not valid_predictions:

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
                required_slices.items()
            ):

                if slice_name not in slices:

                    codes.append(
                        f"MISSING_SLICE:{slice_name}"
                    )

                elif slices[slice_name] < float(
                    floor
                ):

                    codes.append(
                        f"SLICE_FLOOR:{slice_name}"
                    )

        # ----------------------------------------------------
        # Size
        # ----------------------------------------------------

        if total_bytes is not None:

            if total_bytes > policy[
                "maxBytes"
            ]:

                codes.append(
                    "SIZE_LIMIT"
                )

        # ----------------------------------------------------
        # Latency
        # ----------------------------------------------------

        if latency_ms is not None:

            if latency_ms > float(
                policy["maxLatencyMs"]
            ):

                codes.append(
                    "LATENCY_LIMIT"
                )

        codes = sort_codes(
            codes
        )

        admitted = (
            len(codes) == 0
        )

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
    # Select winner
    # --------------------------------------------------------

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
                utf8(result["name"])
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
# POST /quantize
# ============================================================

@app.post("/quantize")
async def quantize(
    request: Request,
    body: Any = Body(...)
):

    if not isinstance(body, dict):

        return JSONResponse(
            status_code=400,
            content={
                "error": "INVALID_INPUT"
            }
        )

    phase = body.get(
        "phase"
    )

    # ========================================================
    # FREEZE
    # ========================================================

    if phase == "freeze":

        if not valid_freeze(body):

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

        freeze_id = body[
            "freezeId"
        ]

        with LOCK:

            stored = get_freeze(
                freeze_id
            )

        # ----------------------------------------------------
        # Unknown freeze
        # ----------------------------------------------------

        if stored is None:

            results = []

            for candidate in body[
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
                key=lambda x:
                utf8(x["name"])
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

        # ----------------------------------------------------
        # Perform selection
        # ----------------------------------------------------

        response = do_select(
            body,
            stored["response"]
        )

        return JSONResponse(
            status_code=200,
            content=response
        )

    # ========================================================
    # UNKNOWN / MISSING PHASE
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
