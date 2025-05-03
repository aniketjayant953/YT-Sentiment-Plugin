import os
import mlflow

def promote_model():
    # Set up AWS MLflow tracking URI
    mlflow.set_tracking_uri("http://ec2-13-61-19-60.eu-north-1.compute.amazonaws.com:5000/")

    client = mlflow.MlflowClient()

    model_name = "my_model"

    # Archive the current production model
    prod_versions = client.get_latest_versions(model_name, stages=["Production"])
    for version in prod_versions:
        client.transition_model_version_stage(
            name=model_name,
            version=version.version,
            stage="Archived"
        )

    # Get the latest version in staging
    latest_version_staging = client.get_latest_versions(model_name, stages=["Staging"])[0].version

    # Promote the new model to production
    client.transition_model_version_stage(
            name=model_name,
            version=latest_version_staging,
            stage="Production"
    )
    print(f"Model version {latest_version_staging} promotes to Production")

if __name__ == "__main__":
    promote_model()