import numpy
# scipy.specialのインポート。シグモイド関数 expit() 利用のため
import scipy.special


class NeuralNetwork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        """
        ニューラルネットワークの初期化: 入力層、隠れ層および入力層の各層のノード数を設定する
        """
        # 入力層、隠れ層、出力層のノード数の設定
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # リンクの重み行列 wih と who
        # 行列内の重み w_i_j, ノード i から次のそうのノードj へのリンクの重み
        # w11 w21
        # w12 w22 など
        self.wih = numpy.random.normal(
            0.0,
            pow(self.hnodes, -0.5),
            (self.hnodes, self.inodes))
        self.who = numpy.random.normal(
            0.0,
            pow(self.onodes, -0.5),
            (self.onodes, self.hnodes))

        # 学習率の設定
        self.lr = learningrate

        # 活性化関数はシグモイド関数
        self.activation_function = lambda x: scipy.special.expit(x)

        pass

    def train(self, inputs_list, targets_list):
        """
        ニューラルネットワークの学習: 与えられた訓練データから重みを調整する
        """
        # 入力リストを行列に変換
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        # 隠れ層に入ってくる信号の計算
        hidden_inputs = numpy.dot(self.wih, inputs)
        # 隠れ層で結合された信号を活性化関数により出力
        hidden_outputs = self.activation_function(hidden_inputs)

        # 出力層に入ってくる信号の計算
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # 出力層で結合された信号を活性化関数により出力
        final_outputs = self.activation_function(final_inputs)

        # 出力層の誤差 = (目標出力 - 最終出力)
        output_errors = targets - final_outputs
        # 隠れ層の誤差は出力層の誤差をリンクの重みの割合で分配
        hidden_errors = numpy.dot(self.who.T, output_errors)

        # 隠れ層と出力層の間のリンクの重みを更新
        self.who += self.lr * numpy.dot(
                (output_errors * final_outputs * (1.0 - final_outputs)),
                numpy.transpose(hidden_outputs))

        # 入力層と隠れ層の間のリンクの重みを更新
        self.wih += self.lr * numpy.dot(
                (hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
                numpy.transpose(inputs))
        pass

    def query(self, inputs_list):
        """
        ニューラルネットワークへの照会: 与えられた入力に対する出力層からの答えを返す
        """
        # 入力リストを行列に変換
        inputs = numpy.array(inputs_list, ndmin=2).T

        # 隠れ層に入ってくる信号の計算
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        # 出力層に入ってくる信号の計算
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # 出力層で結合された信号を活性化関数により出力
        final_outputs = self.activation_function(final_inputs)

        return final_outputs
