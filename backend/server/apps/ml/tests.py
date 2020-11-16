from django.test import TestCase

from apps.ml.income_classifier.MLP import MLP

# add at the beginning of the file:
import inspect
from apps.ml.registry import MLRegistry


class MLTests(TestCase):
    def test_MLP_algorithm(self):
        input_data = {
            0.05: 0.032,
        }
        my_alg = MLP()
        response = my_alg.compute_prediction(input_data)
        # self.assertEqual('OK', response['status'])
        # add below method to MLTests class:
    def test_registry(self):
        registry = MLRegistry()
        self.assertEqual(len(registry.endpoints), 0)
        endpoint_name = "income_classifier"
        algorithm_object = MLP()
        algorithm_name = "random forest"
        algorithm_status = "production"
        algorithm_version = "0.0.1"
        algorithm_owner = "Piotr"
        algorithm_description = "Random Forest with simple pre- and post-processing"
        algorithm_code = inspect.getsource(MLP)
        # add to registry
        registry.add_algorithm(endpoint_name, algorithm_object, algorithm_name,
                    algorithm_status, algorithm_version, algorithm_owner,
                    algorithm_description, algorithm_code)
        # there should be one endpoint available
        self.assertEqual(len(registry.endpoints), 1)