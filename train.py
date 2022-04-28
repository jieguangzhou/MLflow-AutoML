import click
import mlflow
import mlflow.sklearn

from automl.data import load_data


def get_tool(tool_name):
    tool_name = tool_name.lower()
    if tool_name.lower() == "autosklearn":
        from automl.mod_autosklearn import AutoSklearn as Tool
    else:
        raise Exception(f"Does not support {tool_name}")
    return Tool


def create_model_version(model_name, run_id=None, auto_replace=True):
    client = mlflow.tracking.MlflowClient()
    filter_string = "name='{}'".format(model_name)
    versions = client.search_model_versions(filter_string)

    if not versions:
        client.create_registered_model(model_name)

    # TODO: 根据与上一个version对比来判断是否更换Production的模型版本
    for version in versions:
        if version.current_stage == "Production":
            client.transition_model_version_stage(
                model_name, version=version.version, stage="Archived"
            )

    uri = f"runs:/{run_id}/artifact"
    mv = mlflow.register_model(uri, model_name)
    client.transition_model_version_stage(
        model_name, version=mv.version, stage="Production"
    )
    return versions


@click.command()
@click.option("--tool")
@click.option("--data_path")
@click.option("--label_column", default="label")
@click.option("--model_name", default=None)
@click.option("--random_state", default=0)
@click.option("--params", default=None)
def main(
    tool,
    data_path,
    label_column,
    model_name,
    random_state,
    params,
):

    train_x, train_y, test_x, test_y = load_data(
        data_path, label_column, random_state=random_state
    )
    Tool = get_tool(tool)

    classifier = Tool.train_classifier(train_x, train_y, other_params=params)
    # import pickle

    # classifier = pickle.load(open(Tool.model_path, "rb"))

    metrics = Tool.eval(classifier, test_x, test_y)
    print(metrics)
    mlflow.log_metrics(metrics)

    Tool.save_classifier(classifier, Tool.model_path)

    from predictor import PredictorWrapper

    artifacts = {"model_path": Tool.model_path}

    model_info = mlflow.pyfunc.log_model(
        artifact_path="artifact",
        python_model=PredictorWrapper(),
        artifacts=artifacts,
        conda_env=Tool.conda_env,
        code_path=["automl/", "predictor.py"],
    )
    if model_name:
        create_model_version(model_name, run_id=model_info.run_id)


if __name__ == "__main__":
    main()
