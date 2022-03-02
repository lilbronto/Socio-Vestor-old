from Socio_Vestor.utils import compute_rmse
from memoized_property import memoized_property
import mlflow
from  mlflow.tracking import MlflowClient
import joblib
from google.cloud import storage

from Socio_Vestor.params import BUCKET_NAME, STORAGE_LOCATION

from Socio_Vestor.models import SimpleRnn

class Trainer():

    def __init__(self):
        """
            X: pandas DataFrame
            y: pandas Series
        """

        self.experiment_name = "[GER] [MUC] [socio-vestor] socio_vestor_783"

    def set_pipeline(self):
        pass

    def evaluate(self, model, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = model.model.predict(X_test)
        y_pred_reshaped = y_pred.reshape((y_pred.shape[0],))
        rmse = compute_rmse(y_pred_reshaped, y_test)
        print(f"The root mean squared error is: {rmse}")
        return rmse

    # MLFLOW CLIENT
    @memoized_property
    def mlflow_client(self):
        MLFLOW_URI = "https://mlflow.lewagon.co/"
        mlflow.set_tracking_uri(MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)

    # Saving the Model
    def save_model(self, model):
        """ Save the trained model into a model.joblib file """
        joblib.dump(model, 'sociovestor.joblib')

        # Implement here
        client = storage.Client()

        bucket = client.bucket(BUCKET_NAME)

        blob = bucket.blob(STORAGE_LOCATION)

        blob.upload_from_filename('sociovestor.joblib')
        print(f"uploaded model.joblib to gcp cloud storage under \n => {STORAGE_LOCATION}")


if __name__ == "__main__":

    # import your model, get the train and test data and train it
    model = SimpleRnn()
    X_train, X_test, y_train, y_test = model.get_data()
    model.train_rnn(X_train, y_train)

    # initialize a trainer and use it to save the model in the cloud
    trainer = Trainer()
    trainer.save_model(model.model)

    # evaluate the model and print its score
    evaluated_model = trainer.evaluate(model, X_test, y_test)

    # use mlflow to keep track of the hyperparameters
    trainer.mlflow_log_metric("rmse", evaluated_model)
    trainer.mlflow_log_param("model", "SimpleRNN")
    trainer.mlflow_log_param("student_name", trainer.experiment_name)

    # print the website where the hyperparameters can be found
    experiment_id = trainer.mlflow_experiment_id
    print(f"experiment URL: https://mlflow.lewagon.co/#/experiments/{experiment_id}")
