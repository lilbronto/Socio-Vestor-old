from Socio_Vestor.utils import compute_rmse
from memoized_property import memoized_property
import mlflow
from  mlflow.tracking import MlflowClient
import joblib
from google.cloud import storage

from Socio_Vestor.params import BUCKET_NAME, STORAGE_LOCATION

from Socio_Vestor.models import LayerLSTM, SimpleRnn, LSTM

class Trainer():

    def __init__(self):
        """
            X: pandas DataFrame
            y: pandas Series
        """

        self.experiment_name = "[GER] [MUC] [socio-vestor] socio_vestor_783"


    def evaluate(self, model, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = model.model.predict(X_test)
        y_pred_reshaped = y_pred.reshape((y_pred.shape[0],))
        rmse = compute_rmse(y_pred_reshaped, y_test)
        print(f"The root mean squared error is: {rmse}")
        return rmse


    # Saving the Model
    def save_model(self, model):
        """ Save the trained model into a model.joblib file """
        joblib.dump(model, 'sociovestor.joblib')


if __name__ == "__main__":

    # import your model, get the train and test data and train it
    model = LayerLSTM()
    X_train, X_test, y_train, y_test = model.get_data()
    model.train_LSTM(X_train, y_train)

    # initialize a trainer and use it to save the model in the cloud
    trainer = Trainer()
    trainer.save_model(model.model)

    # evaluate the model and print its score
    evaluated_model = trainer.evaluate(model, X_test, y_test)

    # use mlflow to keep track of the hyperparameters
    trainer.mlflow_log_metric("rmse", evaluated_model)
    trainer.mlflow_log_param("model", "LSTM with four Layers")
    trainer.mlflow_log_param("student_name", trainer.experiment_name)

    # print the website where the hyperparameters can be found
    experiment_id = trainer.mlflow_experiment_id
    print(f"experiment URL: https://mlflow.lewagon.co/#/experiments/{experiment_id}")
