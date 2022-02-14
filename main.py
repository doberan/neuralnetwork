import numpy
import json
# import matplotlib.pyplot
from neuralnetwork import NeuralNetwork

input_nodes = 784
hidden_nodes = 100
output_nodes = 10

# 学習率
learning_rate = 0.3

with open('init.json', mode='rt', encoding='utf-8') as file:
    data = json.load(file)
    input_nodes = data['input_nodes']
    hidden_nodes = data['hidden_nodes']
    output_nodes = data['output_nodes']
    learning_rate = data['learning_rate']
    pass


# ニューラルネットワークのインスタンスの生成
n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# MNIST 訓練データのCSVファイルを読み込んでリストにする
training_data_file = open("mnist_dataset/mnist_train.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

# ニューラルネットワークの学習

# epochs: 訓練データが学習で使われた回数
epochs = 7

for _ in range(epochs):
    # 訓練データの全データに対して実行
    for record in training_data_list:
        # データをコンマ ',' でsplit
        all_values = record.split(',')
        # 入力値のスケーリングとシフト
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        # 目標配列の生成（ラベルの位置が0.99 残りは0.01
        targets = numpy.zeros(output_nodes) + 0.01
        # all_valuesp[0] はこのデータのラベル
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)
        pass

# MNIST テストデータのCSVファイルを読み込んでリストにする
test_data_file = open("mnist_dataset/mnist_test.csv", mode='r')
test_data_list = test_data_file.readlines()
test_data_file.close()

# ニューラルネットワークのテスト

# scorecard は判定のリスト、最初は空
scorecard = []

# テストデータの全てのデータに対して実行
for record in test_data_list:
    # データをコンマ ',' でsplit
    all_values = record.split(',')
    # 正解は配列の1番目
    correct_label = int(all_values[0])
    # print(correct_label, "correct label")
    # 入力値のスケーリングとシフト
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    # ネットワークへの照会
    outputs = n.query(inputs)
    label = numpy.argmax(outputs)
    # print(label, "network's answer")
    # 正解(1), 間違い(0) をリストに追加
    if (label == correct_label):
        # 正解なら1 を追加
        scorecard.append(1)
    else:
        scorecard.append(0)
        pass
    pass

# 評価値（正解の割合）の計算
scorecard_array = numpy.asarray(scorecard)
print("performance = ", scorecard_array.sum() / scorecard_array.size)

# with open('init.json', mode='wt', encoding='utf-8') as file:
#     data = dict()
#     data['input_nodes'] = n.inodes
#     data['hidden_nodes'] = n.hnodes
#     data['output_nodes'] = n.onodes
#     data['learning_rate'] = n.lr

#     json.dump(data, file, ensure_ascii=False, indent=2)
#     pass
