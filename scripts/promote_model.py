import os
import mlflow


def promote_model():

    dagshub_token = os.getenv("DAGSHUB_TOKEN")

    if not dagshub_token:
        raise EnvironmentError(
            "DAGSHUB_TOKEN environment variable is not set"
        )

    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

    mlflow.set_tracking_uri(
        "https://dagshub.com/Parshaw3558/sentiment-analysis-mlops.mlflow"
    )

    client = mlflow.MlflowClient()

    model_name = "my_model"

    try:

        staging_versions = client.get_latest_versions(
            model_name,
            stages=["Staging"]
        )

        if len(staging_versions) == 0:
            print(
                f"No staging version found for model {model_name}"
            )
            return

        latest_version_staging = staging_versions[0].version

        prod_versions = client.get_latest_versions(
            model_name,
            stages=["Production"]
        )

        for version in prod_versions:

            client.transition_model_version_stage(
                name=model_name,
                version=version.version,
                stage="Archived"
            )

        client.transition_model_version_stage(
            name=model_name,
            version=latest_version_staging,
            stage="Production"
        )

        print(
            f"Model version {latest_version_staging} promoted successfully"
        )

    except Exception as e:
        print(f"Promotion failed: {e}")
        raise


if __name__ == "__main__":
    promote_model()