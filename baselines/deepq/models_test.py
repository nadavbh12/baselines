import unittest
from baselines.deepq.models_torch import mlp, cnn_to_mlp


class MlpTest(unittest.TestCase):
    def test_initialization(self):
        try:
            mlp([8, 8, 3], 2, [10])
            mlp([8, 8, 3], 1, [10])
            mlp([8, 8, 3], 10, [10, 20, 100, 2])
        except:
            self.fail('initialization shouldn''t fail')
        with self.assertRaises(ValueError):
            mlp([8, 8, 3], 0, [10])
        with self.assertRaises(ValueError):
            mlp([8, 8, 3], 0, (10))
        with self.assertRaises(ValueError):
            mlp((8, 8, 3), 2, [10])


class CnnToMlpTest(unittest.TestCase):
    def test_initialization(self):
        try:
            cnn_to_mlp(
                input_dim=[3, 20, 20],
                num_actions=2,
                convs=[(8, 2, 1)],
                hiddens=[24])
            cnn_to_mlp(
                input_dim=[3, 20, 20],
                num_actions=100,
                convs=[(8, 2, 1), (8, 2, 1), ],
                hiddens=[24, 15, 34, 3, 90])
        except:
            self.fail('initialization shouldn''t fail')


if __name__ == '__main__':
    unittest.main()
