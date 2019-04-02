from unittest.mock import patch

from autokeras.bayesian import edit_distance
from autokeras.backend.torch.loss_function import classification_loss
from autokeras.nn.metric import Accuracy
from autokeras.search import *
from autokeras.nn.generator import CnnGenerator, MlpGenerator, ResNetGenerator
from tests.common import clean_dir, MockProcess, get_classification_data_loaders, get_classification_data_loaders_mlp, \
    simple_transform, TEST_TEMP_DIR, simple_transform_mlp, mock_train, mock_out_of_memory_train, \
    mock_exception_handling_train

from nas.greedy import GreedySearcher
from nas.grid import GridSearcher
from nas.random import RandomSearcher


@patch('torch.multiprocessing.get_context', side_effect=MockProcess)
@patch('autokeras.bayesian.transform', side_effect=simple_transform)
@patch('autokeras.backend.torch.model_trainer.ModelTrainer.train_model', side_effect=mock_train)
def test_greedy_searcher(_, _1, _2):
    train_data, test_data = get_classification_data_loaders()
    clean_dir(TEST_TEMP_DIR)
    searcher = GreedySearcher(3, (28, 28, 3), verbose=False, path=TEST_TEMP_DIR, metric=Accuracy,
                              loss=classification_loss, generators=[CnnGenerator, CnnGenerator])
    for _ in range(2):
        searcher.search(train_data, test_data)
    clean_dir(TEST_TEMP_DIR)
    assert len(searcher.history) == 2


@patch('torch.multiprocessing.get_context', side_effect=MockProcess)
@patch('autokeras.bayesian.transform', side_effect=simple_transform)
@patch('autokeras.backend.torch.model_trainer.ModelTrainer.train_model', side_effect=mock_train)
@patch('autokeras.search.get_system', return_value=Constant.SYS_GOOGLE_COLAB)
def test_greedy_searcher_sp(_, _1, _2, _3):
    train_data, test_data = get_classification_data_loaders()
    clean_dir(TEST_TEMP_DIR)
    searcher = GreedySearcher(3, (28, 28, 3), verbose=False, path=TEST_TEMP_DIR, metric=Accuracy,
                              loss=classification_loss, generators=[CnnGenerator, CnnGenerator])
    for _ in range(2):
        searcher.search(train_data, test_data)
    clean_dir(TEST_TEMP_DIR)
    assert len(searcher.history) == 2


@patch('torch.multiprocessing.get_context', side_effect=MockProcess)
@patch('autokeras.bayesian.transform', side_effect=simple_transform_mlp)
@patch('autokeras.backend.torch.model_trainer.ModelTrainer.train_model', side_effect=mock_train)
def test_greedy_searcher_mlp(_, _1, _2):
    train_data, test_data = get_classification_data_loaders_mlp()
    clean_dir(TEST_TEMP_DIR)
    generator = GreedySearcher(3, (28,), verbose=False, path=TEST_TEMP_DIR, metric=Accuracy,
                               loss=classification_loss, generators=[MlpGenerator, MlpGenerator])
    for _ in range(2):
        generator.search(train_data, test_data)
    clean_dir(TEST_TEMP_DIR)
    assert len(generator.history) == 2


@patch('torch.multiprocessing.get_context', side_effect=MockProcess)
@patch('autokeras.bayesian.transform', side_effect=simple_transform)
@patch('autokeras.backend.torch.model_trainer.ModelTrainer.train_model', side_effect=mock_train)
def test_random_searcher(_, _1, _2):
    train_data, test_data = get_classification_data_loaders()
    clean_dir(TEST_TEMP_DIR)
    searcher = RandomSearcher(3, (28, 28, 3), verbose=False, path=TEST_TEMP_DIR, metric=Accuracy,
                              loss=classification_loss, generators=[CnnGenerator, CnnGenerator])
    for _ in range(2):
        searcher.search(train_data, test_data)
    clean_dir(TEST_TEMP_DIR)
    assert len(searcher.history) == 2


@patch('torch.multiprocessing.get_context', side_effect=MockProcess)
@patch('autokeras.bayesian.transform', side_effect=simple_transform)
@patch('autokeras.backend.torch.model_trainer.ModelTrainer.train_model', side_effect=mock_train)
@patch('autokeras.search.get_system', return_value=Constant.SYS_GOOGLE_COLAB)
def test_random_searcher_sp(_, _1, _2, _3):
    train_data, test_data = get_classification_data_loaders()
    clean_dir(TEST_TEMP_DIR)
    searcher = RandomSearcher(3, (28, 28, 3), verbose=False, path=TEST_TEMP_DIR, metric=Accuracy,
                              loss=classification_loss, generators=[CnnGenerator, CnnGenerator])
    for _ in range(2):
        searcher.search(train_data, test_data)
    clean_dir(TEST_TEMP_DIR)
    assert len(searcher.history) == 2


@patch('torch.multiprocessing.get_context', side_effect=MockProcess)
@patch('autokeras.bayesian.transform', side_effect=simple_transform)
@patch('autokeras.backend.torch.model_trainer.ModelTrainer.train_model', side_effect=mock_train)
def test_grid_searcher(_, _1, _2):
    train_data, test_data = get_classification_data_loaders()
    clean_dir(TEST_TEMP_DIR)
    searcher = GridSearcher(3, (28, 28, 3), verbose=True, path=TEST_TEMP_DIR, metric=Accuracy,
                            loss=classification_loss, generators=[CnnGenerator, CnnGenerator])
    Constant.N_NEIGHBOURS = 1
    Constant.T_MIN = 0.8
    print(len(searcher.get_search_dimensions()))
    for _ in range(len(searcher.get_search_dimensions())):
        searcher.search(train_data, test_data)
    clean_dir(TEST_TEMP_DIR)
    assert len(searcher.history) == len(searcher.search_dimensions)

