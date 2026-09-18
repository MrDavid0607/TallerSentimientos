import csv
import json
import math
import os
from functools import lru_cache
from typing import Any, Union

import mlflow
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from pydantic import BaseModel, StrictStr, field_validator


# ============================================================
# CONFIGURACIÓN
# ============================================================

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://ec2-100-26-91-142.compute-1.amazonaws.com:5000",
)

EXPERIMENT_NAME = "nlp-lab2-sentiment140"

MODEL_NAME = "sentiment140"
MODEL_ALIAS = "champion"

PRESENTED_RUN_TYPES = {
    "protocol",
    "experiment",
    "final",
}

REQUIRED_EXPERIMENT_METRICS = {
    "macro_f1_fold_0",
    "macro_f1_fold_1",
    "macro_f1_fold_2",
    "macro_f1_mean",
    "macro_f1_std",
}

ABLATION_DECISIONS = {
    "preprocessing.stopwords",
    "preprocessing.lemmatize",
    "preprocessing.elongation",
    "preprocessing.emoji",
    "representation",
    "classifier",
}

PREPROCESSING_KEYS = {
    "lowercase",
    "url",
    "mention",
    "whitespace",
    "stopwords",
    "negators",
    "lemmatize",
    "elongation",
    "elongation_spec",
    "emoji",
    "emoji_spec",
    "resources",
    "additional",
}

REPRESENTATION_KEYS = {
    "type",
    "ngram_range",
    "library",
    "library_version",
    "spacy_model",
    "spacy_model_version",
    "parameters",
}

CLASSIFIER_KEYS = {
    "type",
    "library",
    "library_version",
    "parameters",
}

EXPERIMENT_STAGE_MAP = {
    "T0": "reference",
    "B0": "baseline",

    "P_STOPWORDS": "preprocessing",
    "P_STOPWORDS_NEGATION": "preprocessing",
    "P_LEMMA": "preprocessing",
    "P_ELONGATION": "preprocessing",
    "P_EMOJI": "preprocessing",

    "R_BOW": "representation",
    "R_TFIDF_UNI": "representation",
    "R_TFIDF_UNI_BI": "representation",
    "R_SPACY": "representation",

    "C_LOGREG": "classifier",
    "C_LINEAR_SVM": "classifier",
    "C_SGD": "classifier",

    "ABLATION": "ablation",
}


mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_registry_uri(MLFLOW_TRACKING_URI)


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Sentiment140 API",
    version="1.0.0",
)


# ============================================================
# MODELOS DE ENTRADA
# ============================================================

class PredictRequest(BaseModel):
    text: Union[StrictStr, list[StrictStr]]

    @field_validator("text")
    @classmethod
    def validate_text(cls, value):
        texts = [value] if isinstance(value, str) else value

        if len(texts) == 0:
            raise ValueError("text list cannot be empty")

        if len(texts) > 32:
            raise ValueError("maximum 32 texts")

        for text in texts:
            if not text.strip():
                raise ValueError(
                    "text cannot be empty or whitespace"
                )

            if len(text) > 1000:
                raise ValueError(
                    "maximum 1000 characters per text"
                )

        return value


# ============================================================
# HELPERS MLFLOW
# ============================================================

def get_client() -> MlflowClient:
    return MlflowClient()


def get_experiment(client: MlflowClient):
    return client.get_experiment_by_name(
        EXPERIMENT_NAME
    )


def is_mlflow_not_found(exc: Exception) -> bool:
    error_code = getattr(
        exc,
        "error_code",
        None,
    )

    if error_code == "RESOURCE_DOES_NOT_EXIST":
        return True

    message = str(exc).lower()

    return (
        "alias" in message
        and (
            "does not exist" in message
            or "not found" in message
        )
    )


def mlflow_unavailable(exc: Exception):
    raise HTTPException(
        status_code=503,
        detail="mlflow_unavailable",
    ) from exc


# ============================================================
# CHAMPION MODEL
# ============================================================

@lru_cache(maxsize=8)
def load_registered_model(
    version: str,
):
    """
    La caché se hace por versión concreta.
    De esta forma el alias champion se resuelve en MLflow,
    pero no descargamos el modelo en cada petición.
    """

    return mlflow.pyfunc.load_model(
        f"models:/{MODEL_NAME}/{version}"
    )


def resolve_champion():
    client = get_client()

    version = client.get_model_version_by_alias(
        MODEL_NAME,
        MODEL_ALIAS,
    )

    model = load_registered_model(
        str(version.version)
    )

    return model, version


# ============================================================
# ARTEFACTOS
# ============================================================

def list_artifact_files(
    client: MlflowClient,
    run_id: str,
    path: str = "",
) -> list[str]:
    """
    Lista recursivamente todos los archivos del run.
    Devuelve rutas relativas y ordenadas.
    """

    result = []

    for item in client.list_artifacts(
        run_id,
        path,
    ):
        if item.is_dir:
            result.extend(
                list_artifact_files(
                    client,
                    run_id,
                    item.path,
                )
            )
        else:
            result.append(item.path)

    return sorted(result)


def load_json_artifact(
    client: MlflowClient,
    run_id: str,
    artifact_path: str,
):
    local_path = client.download_artifacts(
        run_id,
        artifact_path,
    )

    try:
        with open(
            local_path,
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)

    except (
        json.JSONDecodeError,
        UnicodeDecodeError,
        OSError,
    ):
        return None


def get_configuration(
    client: MlflowClient,
    run,
    artifacts: list[str] | None = None,
):
    """
    protocol -> null

    experimental/final:
    run/configuration.json -> objeto
    falta o JSON inválido -> null
    """

    if (
        run.data.tags.get("lab_run_type")
        == "protocol"
    ):
        return None

    if artifacts is None:
        artifacts = list_artifact_files(
            client,
            run.info.run_id,
        )

    if (
        "run/configuration.json"
        not in artifacts
    ):
        return None

    return load_json_artifact(
        client,
        run.info.run_id,
        "run/configuration.json",
    )


# ============================================================
# PROTOCOLO
# ============================================================

def find_protocol_runs(
    client: MlflowClient,
):
    experiment = get_experiment(client)

    if experiment is None:
        return []

    runs = client.search_runs(
        experiment_ids=[
            experiment.experiment_id
        ],
        filter_string=(
            "tags.lab_run_type = 'protocol'"
        ),
    )

    return list(runs)


def get_unique_protocol_run(
    client: MlflowClient,
):
    runs = find_protocol_runs(client)

    if len(runs) != 1:
        raise HTTPException(
            status_code=409,
            detail="protocol_not_unique",
        )

    return runs[0]


def load_protocol_members(
    client: MlflowClient,
    protocol_run,
):
    path = client.download_artifacts(
        protocol_run.info.run_id,
        "protocol/members.csv",
    )

    members = []

    with open(
        path,
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            members.append({
                "member_id":
                    row["member_id"],
                "notebook_arn":
                    row["notebook_arn"],
            })

    return sorted(
        members,
        key=lambda x: x["member_id"],
    )


# ============================================================
# VALIDACIÓN DE CONFIGURACIÓN
# ============================================================

def valid_configuration_structure(
    config: Any,
    experiment_id: str,
) -> bool:

    if not isinstance(config, dict):
        return False

    if set(config.keys()) != {
        "preprocessing",
        "representation",
        "classifier",
    }:
        return False

    # T0 es especial
    if experiment_id == "T0":
        if config["preprocessing"] is not None:
            return False

        if config["representation"] is not None:
            return False

        clf = config["classifier"]

        if not isinstance(clf, dict):
            return False

        if set(clf.keys()) != CLASSIFIER_KEYS:
            return False

        if (
            clf.get("type")
            != "most_frequent"
        ):
            return False

        if not isinstance(
            clf.get("parameters"),
            dict,
        ):
            return False

        return True

    pre = config["preprocessing"]
    rep = config["representation"]
    clf = config["classifier"]

    if not all(
        isinstance(x, dict)
        for x in (pre, rep, clf)
    ):
        return False

    if set(pre.keys()) != PREPROCESSING_KEYS:
        return False

    if set(rep.keys()) != REPRESENTATION_KEYS:
        return False

    if set(clf.keys()) != CLASSIFIER_KEYS:
        return False

    # preprocessing
    if not isinstance(
        pre.get("lowercase"),
        bool,
    ):
        return False

    if pre.get("url") not in {
        "keep",
        "drop",
    } and not str(
        pre.get("url")
    ).startswith("token:"):
        return False

    if pre.get("mention") not in {
        "keep",
        "drop",
    } and not str(
        pre.get("mention")
    ).startswith("token:"):
        return False

    if pre.get("whitespace") not in {
        "keep",
        "normalize",
    }:
        return False

    if pre.get("stopwords") not in {
        "keep",
        "remove",
        "remove_preserve_negation",
    }:
        return False

    if not isinstance(
        pre.get("negators"),
        list,
    ):
        return False

    if (
        pre.get("stopwords")
        == "remove_preserve_negation"
    ):
        if len(pre["negators"]) == 0:
            return False
    else:
        if pre["negators"] != []:
            return False

    if not isinstance(
        pre.get("lemmatize"),
        bool,
    ):
        return False

    if pre.get("elongation") not in {
        "keep",
        "normalize",
    }:
        return False

    if (
        pre.get("elongation")
        == "normalize"
    ):
        if pre.get("elongation_spec") is None:
            return False
    else:
        if pre.get("elongation_spec") is not None:
            return False

    if pre.get("emoji") not in {
        "keep",
        "text",
    }:
        return False

    if pre.get("emoji") == "text":
        if pre.get("emoji_spec") is None:
            return False
    else:
        if pre.get("emoji_spec") is not None:
            return False

    if not isinstance(
        pre.get("resources"),
        dict,
    ):
        return False

    if not isinstance(
        pre.get("additional"),
        dict,
    ):
        return False

    # representation
    if rep.get("type") not in {
        "bow",
        "tfidf",
        "spacy_embedding",
    }:
        return False

    if not isinstance(
        rep.get("parameters"),
        dict,
    ):
        return False

    if rep["type"] in {
        "bow",
        "tfidf",
    }:
        ngram = rep.get("ngram_range")

        if (
            not isinstance(ngram, list)
            or len(ngram) != 2
            or not all(
                isinstance(v, int)
                for v in ngram
            )
        ):
            return False

        if rep.get("spacy_model") is not None:
            return False

        if (
            rep.get("spacy_model_version")
            is not None
        ):
            return False

    if rep["type"] == "spacy_embedding":
        if rep.get("ngram_range") != []:
            return False

        if not rep.get("spacy_model"):
            return False

        if not rep.get(
            "spacy_model_version"
        ):
            return False

        method = rep.get(
            "parameters",
            {},
        ).get(
            "document_vector_method"
        )

        if (
            not isinstance(method, str)
            or not method.strip()
        ):
            return False

    # classifier
    if clf.get("type") not in {
        "logistic_regression",
        "linear_svm",
        "sgd",
        "most_frequent",
    }:
        return False

    if not isinstance(
        clf.get("parameters"),
        dict,
    ):
        return False

    return True


# ============================================================
# VALIDACIÓN DE MÉTRICAS
# ============================================================

def metrics_are_consistent(metrics):
    if not REQUIRED_EXPERIMENT_METRICS.issubset(
        metrics.keys()
    ):
        return False

    scores = [
        float(metrics["macro_f1_fold_0"]),
        float(metrics["macro_f1_fold_1"]),
        float(metrics["macro_f1_fold_2"]),
    ]

    mean = sum(scores) / 3

    std = math.sqrt(
        sum(
            (score - mean) ** 2
            for score in scores
        )
        / 3
    )

    if abs(
        mean
        - float(metrics["macro_f1_mean"])
    ) > 1e-6:
        return False

    if abs(
        std
        - float(metrics["macro_f1_std"])
    ) > 1e-6:
        return False

    return True


# ============================================================
# VALIDACIÓN DE PROCEDENCIA
# ============================================================

def provenance_matches(
    client,
    run,
    artifacts,
) -> bool:

    provenance_path = (
        "provenance/"
        "sagemaker-resource-metadata.json"
    )

    if provenance_path not in artifacts:
        return False

    provenance = load_json_artifact(
        client,
        run.info.run_id,
        provenance_path,
    )

    if not isinstance(
        provenance,
        dict,
    ):
        return False

    return (
        provenance.get("ResourceArn")
        == run.data.tags.get(
            "notebook_arn"
        )
    )


# ============================================================
# VALIDEZ DE RUN EXPERIMENTAL
# ============================================================

def validate_experimental_run(
    client,
    run,
    protocol_run_id,
    member_pairs,
):

    tags = run.data.tags
    metrics = run.data.metrics

    if run.info.status != "FINISHED":
        return False, None

    if (
        tags.get("lab_run_type")
        != "experiment"
    ):
        return False, None

    required_tags = {
        "lab_protocol_run_id",
        "lab_experiment_id",
        "lab_stage",
        "lab_member_id",
        "lab_configuration_id",
        "notebook_arn",
    }

    if not all(
        tags.get(tag)
        for tag in required_tags
    ):
        return False, None

    if (
        tags["lab_protocol_run_id"]
        != protocol_run_id
    ):
        return False, None

    pair = (
        tags["lab_member_id"],
        tags["notebook_arn"],
    )

    if pair not in member_pairs:
        return False, None

    experiment_id = tags[
        "lab_experiment_id"
    ]

    stage = tags["lab_stage"]

    if experiment_id in EXPERIMENT_STAGE_MAP:
        if (
            stage
            != EXPERIMENT_STAGE_MAP[
                experiment_id
            ]
        ):
            return False, None

    elif experiment_id == "EXTRA":
        if stage not in {
            "preprocessing",
            "representation",
            "classifier",
            "ablation",
        }:
            return False, None

    else:
        return False, None

    if not metrics_are_consistent(
        metrics
    ):
        return False, None

    artifacts = list_artifact_files(
        client,
        run.info.run_id,
    )

    if (
        "run/configuration.json"
        not in artifacts
    ):
        return False, None

    if not provenance_matches(
        client,
        run,
        artifacts,
    ):
        return False, None

    config = get_configuration(
        client,
        run,
        artifacts,
    )

    if not valid_configuration_structure(
        config,
        experiment_id,
    ):
        return False, None

    # Reglas adicionales para ablación
    if stage == "ablation":

        parent_id = tags.get(
            "lab_ablation_parent_run_id"
        )

        decision = run.data.params.get(
            "ablation_reverted_decision"
        )

        if not parent_id:
            return False, None

        if decision not in ABLATION_DECISIONS:
            return False, None

        if "macro_f1_delta" not in metrics:
            return False, None

        try:
            parent = client.get_run(
                parent_id
            )

            parent_mean = float(
                parent.data.metrics[
                    "macro_f1_mean"
                ]
            )

            expected_delta = (
                parent_mean
                - float(
                    metrics[
                        "macro_f1_mean"
                    ]
                )
            )

            if abs(
                expected_delta
                - float(
                    metrics[
                        "macro_f1_delta"
                    ]
                )
            ) > 1e-6:
                return False, None

        except Exception:
            return False, None

    return True, config


# ============================================================
# VALIDEZ DE RUN FINAL
# ============================================================

def validate_final_run(
    client,
    run,
    protocol_run_id,
    member_pairs,
):

    tags = run.data.tags
    params = run.data.params
    metrics = run.data.metrics

    if run.info.status != "FINISHED":
        return False, None

    if tags.get("lab_run_type") != "final":
        return False, None

    required_tags = {
        "lab_protocol_run_id",
        "lab_selected_experiment_run_id",
        "lab_configuration_id",
        "lab_member_id",
        "notebook_arn",
    }

    if not all(
        tags.get(tag)
        for tag in required_tags
    ):
        return False, None

    if (
        tags["lab_protocol_run_id"]
        != protocol_run_id
    ):
        return False, None

    pair = (
        tags["lab_member_id"],
        tags["notebook_arn"],
    )

    if pair not in member_pairs:
        return False, None

    try:
        if (
            int(params.get("training_size"))
            != 1360000
        ):
            return False, None
    except Exception:
        return False, None

    if "test_macro_f1" not in metrics:
        return False, None

    artifacts = list_artifact_files(
        client,
        run.info.run_id,
    )

    required_artifacts = {
        "run/configuration.json",
        "provenance/sagemaker-resource-metadata.json",
        "reports/error_analysis.csv",
        "reports/error_analysis.md",
    }

    if not required_artifacts.issubset(
        set(artifacts)
    ):
        return False, None

    if not provenance_matches(
        client,
        run,
        artifacts,
    ):
        return False, None

    config = get_configuration(
        client,
        run,
        artifacts,
    )

    if not valid_configuration_structure(
        config,
        "FINAL",
    ):
        return False, None

    # Debe apuntar a un run experimental
    selected_id = tags[
        "lab_selected_experiment_run_id"
    ]

    try:
        selected = client.get_run(
            selected_id
        )
    except Exception:
        return False, None

    valid_selected, selected_config = (
        validate_experimental_run(
            client,
            selected,
            protocol_run_id,
            member_pairs,
        )
    )

    if not valid_selected:
        return False, None

    if (
        selected.data.tags.get(
            "lab_configuration_id"
        )
        != tags.get(
            "lab_configuration_id"
        )
    ):
        return False, None

    if selected_config != config:
        return False, None

    return True, config


# ============================================================
# CONFIGURATION-ID CONSISTENCY
# ============================================================

def find_configuration_id_conflicts(
    valid_runs,
):
    """
    La misma configuración JSON debe compartir ID.
    Un mismo ID tampoco debería describir
    configuraciones diferentes.
    """

    canonical_to_ids = {}
    id_to_canonical = {}

    run_canonicals = {}

    for run, config in valid_runs:
        if config is None:
            continue

        canonical = json.dumps(
            config,
            sort_keys=True,
            separators=(",", ":"),
        )

        configuration_id = (
            run.data.tags.get(
                "lab_configuration_id"
            )
        )

        run_canonicals[
            run.info.run_id
        ] = (
            canonical,
            configuration_id,
        )

        canonical_to_ids.setdefault(
            canonical,
            set(),
        ).add(configuration_id)

        id_to_canonical.setdefault(
            configuration_id,
            set(),
        ).add(canonical)

    invalid = set()

    for run_id, (
        canonical,
        configuration_id,
    ) in run_canonicals.items():

        if len(
            canonical_to_ids[
                canonical
            ]
        ) > 1:
            invalid.add(run_id)

        if len(
            id_to_canonical[
                configuration_id
            ]
        ) > 1:
            invalid.add(run_id)

    return invalid


# ============================================================
# ENDPOINT ROOT
# ============================================================

@app.get("/")
def root():
    return {
        "service": "sentiment140",
        "status": "running",
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():
    try:
        model, version = resolve_champion()

        # Verificación de inferencia real
        prediction = model.predict(
            ["health check"]
        )

        if len(prediction) != 1:
            raise RuntimeError(
                "invalid health prediction"
            )

        return {
            "status": "ok",
            "model_run_id":
                version.run_id,
        }

    except Exception:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unavailable",
                "model_run_id": None,
            },
        )


# ============================================================
# PREDICT
# ============================================================

@app.post("/api/v1/predict")
def predict(request: PredictRequest):
    texts = (
        [request.text]
        if isinstance(
            request.text,
            str,
        )
        else request.text
    )

    try:
        model, version = resolve_champion()

        predictions = model.predict(
            texts
        )

        predictions = [
            str(value)
            for value in predictions
        ]

        if (
            len(predictions)
            != len(texts)
        ):
            raise RuntimeError(
                "prediction count mismatch"
            )

        if not all(
            prediction in {
                "negative",
                "positive",
            }
            for prediction in predictions
        ):
            raise RuntimeError(
                "invalid prediction label"
            )

        return {
            "model_run_id":
                version.run_id,
            "predictions":
                predictions,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="model_unavailable",
        ) from exc


# ============================================================
# AUDIT PROTOCOL
# ============================================================

@app.get("/audit/protocol")
def audit_protocol():
    try:
        client = get_client()

        protocol = get_unique_protocol_run(
            client
        )

        params = protocol.data.params

        return {
            "protocol_run_id":
                protocol.info.run_id,

            "dataset_id":
                params["dataset_id"],

            "dataset_revision":
                params[
                    "dataset_revision"
                ],

            "sampling_strategy":
                params[
                    "sampling_strategy"
                ],

            "sample_size":
                int(
                    params["sample_size"]
                ),

            "random_seed":
                int(
                    params["random_seed"]
                ),

            "cv_strategy":
                params["cv_strategy"],

            "cv_folds":
                int(
                    params["cv_folds"]
                ),

            "cv_shuffle":
                str(
                    params["cv_shuffle"]
                ).lower()
                == "true",

            "partitions_artifact":
                "protocol/partitions.csv",

            "members_artifact":
                "protocol/members.csv",
        }

    except HTTPException:
        raise

    except Exception as exc:
        mlflow_unavailable(exc)


# ============================================================
# AUDIT RUNS
# ============================================================

@app.get("/audit/runs")
def audit_runs():
    try:
        client = get_client()

        experiment = get_experiment(
            client
        )

        if experiment is None:
            return {
                "runs": []
            }

        all_runs = client.search_runs(
            experiment_ids=[
                experiment.experiment_id
            ]
        )

        presented = [
            run
            for run in all_runs
            if run.data.tags.get(
                "lab_run_type"
            )
            in PRESENTED_RUN_TYPES
        ]

        result = []

        for run in presented:

            artifacts = (
                list_artifact_files(
                    client,
                    run.info.run_id,
                )
            )

            configuration = (
                get_configuration(
                    client,
                    run,
                    artifacts,
                )
            )

            result.append({
                "run_id":
                    run.info.run_id,

                "status":
                    run.info.status,

                "run_type":
                    run.data.tags.get(
                        "lab_run_type"
                    ),

                "params": {
                    str(key):
                        str(value)
                    for key, value
                    in run.data.params.items()
                },

                "metrics": {
                    str(key):
                        float(value)
                    for key, value
                    in run.data.metrics.items()
                },

                "tags": {
                    str(key):
                        str(value)
                    for key, value
                    in run.data.tags.items()
                },

                "artifacts":
                    artifacts,

                "configuration":
                    configuration,
            })

        result.sort(
            key=lambda x: x["run_id"]
        )

        return {
            "runs": result
        }

    except Exception as exc:
        mlflow_unavailable(exc)


# ============================================================
# AUDIT CONTRIBUTIONS
# ============================================================

@app.get("/audit/contributions")
def audit_contributions():
    try:
        client = get_client()

        protocol = get_unique_protocol_run(
            client
        )

        protocol_run_id = (
            protocol.info.run_id
        )

        members = load_protocol_members(
            client,
            protocol,
        )

        member_pairs = {
            (
                member["member_id"],
                member["notebook_arn"],
            )
            for member in members
        }

        experiment = get_experiment(
            client
        )

        if experiment is None:
            raise RuntimeError(
                "experiment missing"
            )

        all_runs = client.search_runs(
            experiment_ids=[
                experiment.experiment_id
            ]
        )

        relevant = [
            run
            for run in all_runs
            if run.data.tags.get(
                "lab_run_type"
            )
            in {
                "experiment",
                "final",
            }
        ]

        prelim_valid = []
        invalid_ids = set()
        unattributed_ids = set()

        # ----------------------------------------
        # Primera validación
        # ----------------------------------------

        for run in relevant:

            tags = run.data.tags

            member_id = tags.get(
                "lab_member_id"
            )

            notebook_arn = tags.get(
                "notebook_arn"
            )

            pair = (
                member_id,
                notebook_arn,
            )

            attributed = (
                pair in member_pairs
            )

            if not attributed:
                invalid_ids.add(
                    run.info.run_id
                )

                unattributed_ids.add(
                    run.info.run_id
                )

                continue

            run_type = tags.get(
                "lab_run_type"
            )

            if run_type == "experiment":
                valid, config = (
                    validate_experimental_run(
                        client,
                        run,
                        protocol_run_id,
                        member_pairs,
                    )
                )

            else:
                valid, config = (
                    validate_final_run(
                        client,
                        run,
                        protocol_run_id,
                        member_pairs,
                    )
                )

            if valid:
                prelim_valid.append(
                    (run, config)
                )
            else:
                invalid_ids.add(
                    run.info.run_id
                )

        # ----------------------------------------
        # Equivalencia configuration_id
        # ----------------------------------------

        config_conflicts = (
            find_configuration_id_conflicts(
                prelim_valid
            )
        )

        invalid_ids.update(
            config_conflicts
        )

        valid_runs = [
            (run, config)
            for run, config
            in prelim_valid
            if run.info.run_id
            not in config_conflicts
        ]

        # ----------------------------------------
        # Construir contribuciones
        # ----------------------------------------

        member_results = []

        for member in members:

            member_id = member[
                "member_id"
            ]

            notebook_arn = member[
                "notebook_arn"
            ]

            valid_experiments = [
                run
                for run, _
                in valid_runs
                if (
                    run.data.tags.get(
                        "lab_run_type"
                    )
                    == "experiment"
                    and
                    run.data.tags.get(
                        "lab_member_id"
                    )
                    == member_id
                    and
                    run.data.tags.get(
                        "notebook_arn"
                    )
                    == notebook_arn
                )
            ]

            run_ids = sorted(
                run.info.run_id
                for run
                in valid_experiments
            )

            counted = [
                run
                for run
                in valid_experiments
                if run.data.tags.get(
                    "lab_experiment_id"
                )
                not in {
                    "T0",
                    "B0",
                }
            ]

            counted_run_ids = sorted(
                run.info.run_id
                for run
                in counted
            )

            configuration_ids = sorted({
                run.data.tags[
                    "lab_configuration_id"
                ]
                for run
                in counted
            })

            stages = sorted({
                run.data.tags[
                    "lab_stage"
                ]
                for run
                in counted
            })

            member_results.append({
                "member_id":
                    member_id,

                "notebook_arn":
                    notebook_arn,

                "run_ids":
                    run_ids,

                "counted_run_ids":
                    counted_run_ids,

                "configuration_ids":
                    configuration_ids,

                "valid_configurations":
                    len(
                        configuration_ids
                    ),

                "stages":
                    stages,
            })

        member_results.sort(
            key=lambda x:
                x["member_id"]
        )

        return {
            "members":
                member_results,

            "invalid_run_ids":
                sorted(
                    invalid_ids
                ),

            "unattributed_run_ids":
                sorted(
                    unattributed_ids
                ),
        }

    except HTTPException:
        raise

    except Exception as exc:
        mlflow_unavailable(exc)


# ============================================================
# AUDIT MODEL
# ============================================================

@app.get("/audit/model")
def audit_model():
    try:
        client = get_client()

        # ----------------------------------------
        # Resolver champion
        # ----------------------------------------

        try:
            model_version = (
                client
                .get_model_version_by_alias(
                    MODEL_NAME,
                    MODEL_ALIAS,
                )
            )

        except MlflowException as exc:

            if is_mlflow_not_found(exc):
                raise HTTPException(
                    status_code=404,
                    detail="champion_not_found",
                ) from exc

            mlflow_unavailable(exc)

        # ----------------------------------------
        # Recuperar protocolo y miembros
        # ----------------------------------------

        protocol = get_unique_protocol_run(
            client
        )

        members = load_protocol_members(
            client,
            protocol,
        )

        member_pairs = {
            (
                member["member_id"],
                member["notebook_arn"],
            )
            for member in members
        }

        # ----------------------------------------
        # Recuperar run final
        # ----------------------------------------

        run = client.get_run(
            model_version.run_id
        )

        valid, configuration = (
            validate_final_run(
                client,
                run,
                protocol.info.run_id,
                member_pairs,
            )
        )

        if not valid:
            raise HTTPException(
                status_code=409,
                detail="champion_invalid",
            )

        tags = run.data.tags
        params = run.data.params
        metrics = run.data.metrics

        return {
            "model_name":
                MODEL_NAME,

            "alias":
                MODEL_ALIAS,

            "version":
                int(
                    model_version.version
                ),

            "run_id":
                run.info.run_id,

            "protocol_run_id":
                tags[
                    "lab_protocol_run_id"
                ],

            "selected_experiment_run_id":
                tags[
                    "lab_selected_experiment_run_id"
                ],

            "configuration_id":
                tags[
                    "lab_configuration_id"
                ],

            "configuration":
                configuration,

            "training_size":
                int(
                    params[
                        "training_size"
                    ]
                ),

            "test_macro_f1":
                float(
                    metrics[
                        "test_macro_f1"
                    ]
                ),
        }

    except HTTPException:
        raise

    except Exception as exc:
        mlflow_unavailable(exc)