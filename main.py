import hashlib
import json
import math
import sqlite3
import threading

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI(title="Quantize Candidate Admission API")

DB_PATH = "quantize_state.db"
DB_LOCK = threading.Lock()


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


def safe_nonnegative_integer(value):
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 0 <= value <= 9007199254740991
    )


def binary_prediction(value):
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value in (0, 1)
    )


def sorted_unique_codes(codes):
    return sorted(set(codes), key=utf8)


def json_equal(a, b):
    return compact_json(a) == compact_json(b)


def round12(value):
    return float(f"{value:.12f}")


# ============================================================
# INVENTORY
# ============================================================

def make_inventory(files):

    if not isinstance(files, dict):
        return False, [], None, None

    if len(files) == 0:
        return False, [], None, None

    inventory = []

    try:
        for filename, content in files.items():

            if not isinstance(filename, str):
                return False, [], None, None

            if filename == "":
                return False, [], None, None

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

    inventory.sort(
        key=lambda item: utf8(item["name"])
    )

    total_bytes = sum(
        item["bytes"]
        for item in inventory
    )

    package_digest = sha256_json(
        inventory
    )

    return (
        True,
        inventory,
        total_bytes,
        package_digest
    )


# ============================================================
# FREEZE VALIDATION
# ============================================================

def validate_freeze_input(body):

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

    candidates = body.get("candidates")

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
# FREEZE OPERATION
# ============================================================

def freeze_operation(body):

    request_calibration = body[
        "calibrationDigest"
    ]

    request_tokenizer = body[
        "tokenizerDigest"
    ]

    allowed_reasons = set(
        body["allowedUnsupportedReasons"]
    )

    result = []

    for candidate in body["candidates"]:

        name = candidate["name"]

        files_ok, inventory, total_bytes, package_digest = (
            make_inventory(
                candidate["files"]
            )
        )

        if not files_ok:

            result.append({
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

            if unsupported_reason in allowed_reasons:

                result.append({
                    "name": name,
                    "status": "unsupported",
                    "inventory": inventory,
                    "totalBytes": total_bytes,
                    "packageDigest": package_digest,
                    "reasonCodes": []
                })

            else:

                result.append({
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

        codes = []

        if not candidate["loadable"]:
            codes.append("NOT_LOADABLE")

        if candidate["calibrationDigest"] != request_calibration:
            codes.append("CALIBRATION_MISMATCH")

        if candidate["tokenizerDigest"] != request_tokenizer:
            codes.append("TOKENIZER_MISMATCH")

        codes = sorted_unique_codes(codes)

        result.append({
            "name": name,
            "status": (
                "frozen"
                if len(codes) == 0
                else "invalid"
            ),
            "inventory": inventory,
            "totalBytes": total_bytes,
            "packageDigest": package_digest,
            "reasonCodes": codes
        })

    result.sort(
        key=lambda item: utf8(item["name"])
    )

    return {
        "freezeId": body["freezeId"],
        "candidates": result
    }


# ============================================================
# POLICY
# ============================================================

def validate_policy(policy):

    if not isinstance(policy, dict):
        return False

    max_bytes = policy.get("maxBytes")

    if not safe_nonnegative_integer(max_bytes):
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

    if not isinstance(required_slices, dict):
        return False

    for slice_name, floor in required_slices.items():

        if not nonempty_string(slice_name):
            return False

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

    if not isinstance(candidate_order, list):
        return False

    if any(
        not nonempty_string(name)
        for name in candidate_order
    ):
        return False

    if len(candidate_order) != len(set(candidate_order)):
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

        if not binary_prediction(label):
            return False, None, {}

        slice_name = row.get("slice")

        if not nonempty_string(slice_name):
            return False, None, {}

        predictions = row.get("predictions")

        if not isinstance(predictions, dict):
            return False, None, {}

        if candidate_name not in predictions:
            return False, None, {}

        prediction = predictions[candidate_name]

        if not binary_prediction(prediction):
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
# MANIFEST VALIDATION
# ============================================================

def validate_recorded_manifest(candidate):

    inventory = candidate.get("inventory")
    total = candidate.get("totalBytes")
    digest = candidate.get("packageDigest")

    if not isinstance(inventory, list):
        return False, None

    if not safe_nonnegative_integer(total):
        return False, None

    if not isinstance(digest, str) or len(digest) != 64:
        return False, None

    previous_name = None
    calculated_total = 0

    for item in inventory:

        if not isinstance(item, dict):
            return False, None

        name = item.get("name")
        byte_count = item.get("bytes")
        item_hash = item.get("sha256")

        if not nonempty_string(name):
            return False, None

        if not safe_nonnegative_integer(byte_count):
            return False, None

        if (
            not isinstance(item_hash, str)
            or len(item_hash) != 64
        ):
            return False, None

        # Inventory must be sorted by UTF-8 filename.
        if previous_name is not None:
            if utf8(name) <= utf8(previous_name):
                return False, None

        previous_name = name

        calculated_total += byte_count

    calculated_digest = sha256_json(
        inventory
    )

    if calculated_total != total:
        return False, None

    if calculated_digest != digest:
        return False, None

    return True, total


# ============================================================
# SELECT
# ============================================================

def select_operation(body, stored_response):

    frozen_candidates = stored_response["candidates"]

    supplied_candidates = body["candidates"]

    frozen_by_name = {
        item["name"]: item
        for item in frozen_candidates
    }

    supplied_by_name = {}

    for item in supplied_candidates:

        if isinstance(item, dict):

            name = item.get("name")

            if isinstance(name, str):
                supplied_by_name[name] = item

    frozen_names = set(frozen_by_name)
    supplied_names = set(supplied_by_name)

    policy = body["policy"]
    policy_ok = validate_policy(policy)

    codes_global = []

    # The grader sends the exact frozen candidate response.
    if not json_equal(
        supplied_candidates,
        frozen_candidates
    ):
        codes_global.append(
            "INVALID_LINEAGE"
        )

    if not policy_ok:
        codes_global.append(
            "INVALID_POLICY"
        )

    required_slices = (
        policy.get("requiredSlices", {})
        if isinstance(policy, dict)
        else {}
    )

    candidate_order = (
        policy.get("candidateOrder", [])
        if isinstance(policy, dict)
        else []
    )

    # Candidate names and candidateOrder
    # must be the same unique set.
    if policy_ok:

        if supplied_names != set(candidate_order):
            codes_global.append(
                "INVALID_POLICY"
            )

    order_index = {
        name: i
        for i, name in enumerate(candidate_order)
    }

    # Results are ordered by candidateOrder.
    names = list(supplied_names)

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

        codes = list(codes_global)

        frozen = frozen_by_name.get(name)

        aggregate = None
        slices = {}

        total_bytes = None
        latency_ms = None

        # ----------------------------------------------------
        # Frozen candidate
        # ----------------------------------------------------

        if (
            frozen is None
            or frozen.get("status") != "frozen"
        ):
            codes.append(
                "NOT_FROZEN"
            )

        # ----------------------------------------------------
        # Validate recorded manifest
        # ----------------------------------------------------

        if frozen is None:

            codes.append(
                "INVALID_MANIFEST"
            )

        else:

            manifest_ok, total_bytes = (
                validate_recorded_manifest(
                    frozen
                )
            )

            if not manifest_ok:
                codes.append(
                    "INVALID_MANIFEST"
                )

        # ----------------------------------------------------
        # Latency
        # ----------------------------------------------------

        latencies = body.get("latencies")

        if (
            isinstance(latencies, dict)
            and name in latencies
        ):

            value = latencies[name]

            if (
                finite_number(value)
                and float(value) >= 0
            ):
                latency_ms = float(value)

        # ----------------------------------------------------
        # Predictions
        # ----------------------------------------------------

        prediction_ok, aggregate, slices = (
            calculate_metrics(
                name,
                body["rows"],
                required_slices
            )
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

        codes = sorted_unique_codes(codes)

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

    admitted_candidates = [
        result
        for result in results
        if result["admitted"]
    ]

    winner = None

    if admitted_candidates:

        winner = min(
            admitted_candidates,
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
async def quantize(request: Request):

    # --------------------------------------------------------
    # IMPORTANT DEBUG LOGGING
    # --------------------------------------------------------

    try:
        raw = await request.body()

        print("========== QUANTIZE REQUEST ==========")
        print(raw.decode("utf-8"))
        print("=======================================")

        body = json.loads(
            raw.decode("utf-8")
        )

    except Exception as exc:

        print(
            "JSON PARSE ERROR:",
            repr(exc)
        )

        return JSONResponse(
            status_code=400,
            content={
                "error": "INVALID_INPUT"
            }
        )

    print(
        "QUANTIZE PHASE:",
        body.get("phase") if isinstance(body, dict) else None
    )

    # JSON root must be object.
    if not isinstance(body, dict):

        print("REJECTED: body is not object")

        return JSONResponse(
            status_code=400,
            content={
                "error": "INVALID_INPUT"
            }
        )

    phase = body.get("phase")

    print("PROCESSING PHASE:", repr(phase))

    # ========================================================
    # FREEZE
    # ========================================================

    if phase == "freeze":

        print("FREEZE REQUEST")

        if not validate_freeze_input(body):

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

        with DB_LOCK:

            existing = get_freeze(
                freeze_id
            )

            if existing is not None:

                if json_equal(
                    existing["input"],
                    body
                ):

                    print(
                        "IDENTICAL FREEZE REPLAY"
                    )

                    return JSONResponse(
                        status_code=200,
                        content=existing["response"]
                    )

                print(
                    "FREEZE ID CONFLICT"
                )

                return JSONResponse(
                    status_code=409,
                    content={
                        "error":
                        "FREEZE_ID_CONFLICT"
                    }
                )

            response = freeze_operation(
                body
            )

            save_freeze(
                freeze_id,
                body,
                response
            )

        print(
            "FREEZE SUCCESS:",
            freeze_id
        )

        return JSONResponse(
            status_code=200,
            content=response
        )

    # ========================================================
    # SELECT
    # ========================================================

    if phase == "select":

        print("SELECT REQUEST")

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

            print(
                "SELECT TOP LEVEL VALIDATION FAILED"
            )

            return JSONResponse(
                status_code=400,
                content={
                    "error": "INVALID_INPUT"
                }
            )

        freeze_id = body["freezeId"]

        with DB_LOCK:
            stored = get_freeze(
                freeze_id
            )

        # Unknown freeze ID is a valid select response.
        if stored is None:

            print(
                "SELECT: FREEZE NOT FOUND:",
                freeze_id
            )

            names = []

            for candidate in body["candidates"]:

                if (
                    isinstance(candidate, dict)
                    and isinstance(
                        candidate.get("name"),
                        str
                    )
                ):
                    names.append(
                        candidate["name"]
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

        response = select_operation(
            body,
            stored["response"]
        )

        print(
            "SELECT SUCCESS:",
            response["selected"]
        )

        return JSONResponse(
            status_code=200,
            content=response
        )

    # ========================================================
    # UNKNOWN / MISSING PHASE
    # ========================================================

    print(
        "INVALID PHASE:",
        repr(phase)
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
        "endpoint": "POST /quantize"
    }
