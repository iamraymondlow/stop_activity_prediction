from load_data import DataLoader


class MLModel:
    """
    Trains and evaluates the performance of traditional ML models based on the training and test dataset.
    """
    def __init__(self):
        """
        Initialises the model object by loading the training and test datasets.
        """
        loader = DataLoader()
        self.train_data, self.test_data = loader.train_test_split(test_ratio=0.25)

    def train_model(self, algorithm=None):
        """
        Trains a model on the training dataset using a user-defined ML algorithm supported by sklearn.

        Parameters:
            algorithm: str
                Indicates the name of the algorithm used to train the model.

        Returns:
            model: sklearn object
                Contains the model trained on the training dataset.
        """
        model = None
        return model

    def eval_model(self):
        """
        Evaluates the performnace of the trained model based on test dataset.
        """
        return None


if __name__ == '__main__':
    model = MLModel()
