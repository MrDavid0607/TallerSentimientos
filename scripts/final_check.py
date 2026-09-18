import json
import os
import sys
import time
import urllib.error
import urllib.request


API = os.getenv(
    "API_URL",
    "http://ec2-100-26-91-142.compute-1.amazonaws.com:8000"
).rstrip("/")

MLFLOW = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://ec2-100-26-91-142.compute-1.amazonaws.com:5000"
).rstrip("/")


failures = []


def check(name, condition, detail=""):
    print(
        "PASS" if condition else "FAIL",
        "-",
        name,
        detail
    )

    if not condition:
        failures.append(name)


def request(
    path,
    method="GET",
    body=None
):
    headers = {
        "Accept": "application/json"
    }

    data = None

    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = \
            "application/json"

    req = urllib.request.Request(
        API + path,
        data=data,
        headers=headers,
        method=method
    )

    start = time.time()

    try:
        with urllib.request.urlopen(
            req,
            timeout=120
        ) as response:

            status = response.status

            payload = json.loads(
                response.read().decode()
            )

    except urllib.error.HTTPError as e:

        status = e.code

        try:
            payload = json.loads(
                e.read().decode()
            )
        except Exception:
            payload = {}

    return (
        status,
        payload,
        time.time() - start
    )


# MLflow público

try:
    with urllib.request.urlopen(
        MLFLOW,
        timeout=10
    ) as response:

        check(
            "MLflow public reachable",
            response.status == 200
        )

except Exception as e:
    check(
        "MLflow public reachable",
        False,
        str(e)
    )


# HEALTH

status, health, _ = request(
    "/health"
)

check(
    "/health HTTP 200",
    status == 200
)

check(
    "/health status ok",
    health.get("status") == "ok"
)


# PROTOCOL

status, protocol, _ = request(
    "/audit/protocol"
)

check(
    "/audit/protocol HTTP 200",
    status == 200
)

check(
    "sample_size == 200000",
    protocol.get("sample_size")
    == 200000
)

check(
    "cv_folds == 3",
    protocol.get("cv_folds")
    == 3
)


# MODEL

status, model, _ = request(
    "/audit/model"
)

check(
    "/audit/model HTTP 200",
    status == 200
)

check(
    "model_name sentiment140",
    model.get("model_name")
    == "sentiment140"
)

check(
    "alias champion",
    model.get("alias")
    == "champion"
)

check(
    "training_size",
    model.get("training_size")
    == 1360000
)


# TRACEABILITY

check(
    "health == audit model run",
    health.get("model_run_id")
    == model.get("run_id")
)


# PREDICT

status, prediction, elapsed = request(
    "/api/v1/predict",
    method="POST",
    body={
        "text": [
            "i loved it",
            "worst day ever",
            "not bad at all"
        ]
    }
)

check(
    "/predict HTTP 200",
    status == 200
)

check(
    "3 predictions returned",
    len(
        prediction.get(
            "predictions",
            []
        )
    )
    == 3
)

check(
    "valid prediction labels",
    set(
        prediction.get(
            "predictions",
            []
        )
    )
    <= {
        "positive",
        "negative"
    }
)

check(
    "predict under 10 seconds",
    elapsed <= 10,
    f"{elapsed:.3f}s"
)

check(
    "predict == final run",
    prediction.get("model_run_id")
    == model.get("run_id")
)


# INVALID REQUESTS

invalid = [
    {},
    {"text": None},
    {"text": ""},
    {"text": "   "},
    {"text": []},
    {"text": ["ok", ""]},
    {"text": ["ok", 7]},
    {"text": ["x"] * 33},
    {"text": "x" * 1001},
]

for i, body in enumerate(
    invalid,
    start=1
):
    status, _, _ = request(
        "/api/v1/predict",
        method="POST",
        body=body
    )

    check(
        f"invalid request {i}",
        400 <= status < 500,
        f"HTTP {status}"
    )


# RUNS

status, payload, _ = request(
    "/audit/runs"
)

runs = payload.get(
    "runs",
    []
)

run_ids = [
    run["run_id"]
    for run in runs
]

check(
    "/audit/runs HTTP 200",
    status == 200
)

check(
    "runs ordered",
    run_ids == sorted(run_ids)
)


# CONTRIBUTIONS

status, contributions, _ = request(
    "/audit/contributions"
)

check(
    "/audit/contributions HTTP 200",
    status == 200
)

members = contributions.get(
    "members",
    []
)

check(
    "3 members",
    len(members) == 3
)

for member in members:

    check(
        member["member_id"]
        + " >=3 configurations",

        member.get(
            "valid_configurations",
            0
        )
        >= 3
    )

    check(
        member["member_id"]
        + " >=2 stages",

        len(
            member.get(
                "stages",
                []
            )
        )
        >= 2
    )


check(
    "unattributed_run_ids empty",
    contributions.get(
        "unattributed_run_ids"
    )
    == []
)

check(
    "invalid_run_ids empty",
    contributions.get(
        "invalid_run_ids"
    )
    == []
)


print()
print("=" * 60)

if failures:

    print(
        "FINAL CHECK: FAIL"
    )

    print(
        "Problems:",
        len(failures)
    )

    for failure in failures:
        print("-", failure)

    sys.exit(1)

else:

    print(
        "FINAL CHECK: PASS"
    )

    sys.exit(0)
