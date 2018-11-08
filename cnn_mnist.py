import numpy as np
import tensorflow as tf

"""下載並載入 MNIST 手寫數字庫 (55000 * 28 * 28) 55000張訓練圖像"""
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('mnist_data', one_hot=True)

"""
one_hot 獨熱編碼 (encoding) 形式
0, 1, 2, 3, 4, 5, 6, 7, 8, 9 的十位數字
0：1000000000
1：0100000000
2：0010000000
3：0001000000
4：0000100000
5：0000010000
6：0000001000
7：0000000100
8：0000000010
9：0000000001
"""

"""None 表示張量 (Tensor) 的第一個維度可以是任何長度"""
input_X = tf.placeholder(tf.float32, [None, 28 * 28]) / 255
output_y = tf.placeholder(tf.int32, [None, 10])         # 輸出：10個數字的標籤
input_X_images = tf.reshape(input_X, [-1, 28, 28, 1])    # 改變形狀之後的輸入

"""從 Test(測試) 數據集裡選取3000個手寫數字的圖片和對應標籤"""
test_X = mnist.test.images[:3000]   # 圖片
test_y = mnist.test.labels[:3000]   # 標籤

"""構建卷積神經網路"""
# tf.nn.conv2d / tf.layers.conv2d(輸入參數較多) 兩個方法雷同
"""第 1 層 卷積"""
conv1 = tf.layers.conv2d(
    inputs=input_X_images,   # 形狀 [28, 28, 1]
    filters=32,             # 32個過濾器，輸出的深度 (depth) 是32
    kernel_size=[5, 5],     # 過濾器在二維的大小是 (5 * 5)
    strides=1,              # 步長是1
    padding='same',          # same 表示輸出大小不變，因此需要在外圍捕零兩圈
    activation=tf.nn.relu   # 激活函數是 Relu
)   # 形狀 [28, 28, 32]

"""第 1 層 池化 (亞採樣)"""
pool1 = tf.layers.max_pooling2d(
    inputs=conv1,       # 形狀 [28, 28, 32]
    pool_size=[2, 2],   # 過濾器在二維的大小是 (2 * 2)
    strides=2,          # 步長是 2
)   # 形狀 [14, 14, 32]

"""第 2 層 卷積"""
conv2 = tf.layers.conv2d(
    inputs=pool1,           # 形狀 [14, 14, 32]
    filters=64,             # 64個過濾器，輸出的深度 (depth) 是64
    kernel_size=[5, 5],     # 過濾器在二維的大小是 (5 * 5)
    strides=1,              # 步長是1
    padding='same',          # same 表示輸出大小不變，因此需要在外圍捕零兩圈
    activation=tf.nn.relu,  # 激活函數是 Relu
)   # 形狀 [14, 14, 64]

"""第 2 層 池化 (亞採樣)"""
pool2 = tf.layers.max_pooling2d(
    inputs=conv2,       # 形狀 [14, 14, 64]
    pool_size=[2, 2],   # 過濾器在二維的大小是 (2 * 2)
    strides=2,          # 步長是 2
)   # 形狀 [7, 7, 64]

"""平坦化 (Flat) """
flat = tf.reshape(pool2, [-1, 7 * 7 * 64])  # 形狀 [7 * 7 * 64, ]

"""1024 個神經元的全連接層"""
dense = tf.layers.dense(inputs=flat, units=1024, activation=tf.nn.relu)

"""Dropout：丟棄 50%, rate=0.5"""
dropout = tf.layers.dropout(inputs=dense, rate=0.5)

"""10 個神經元的全連接層，這裡不用激活函數來做非線性化"""
logits = tf.layers.dense(inputs=dropout, units=10)   # 輸出。形狀[1, 1, 10]

"""計算誤差 (計算 Cross entropy (交叉熵)，再用 Softmax 計算百分概率)"""
loss = tf.losses.softmax_cross_entropy(onehot_labels=output_y, logits=logits)

"""用 Adam 優化器來最小化誤差，學習率 0.001"""
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

"""精度。計算 預測值 和 實際標籤 的匹配程度"""
accuracy = tf.metrics.accuracy(
    labels=tf.argmax(output_y, axis=1),
    predictions=tf.argmax(logits, axis=1)
)[1]    # 返回 (accuracy, update_op)，會創建兩個局部變量


"""創建會話"""
sess = tf.Session()
"""初始化變量"""
init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
sess.run(init)

for i in range(200):
    batch = mnist.train.next_batch(50)  # 從 Train (訓練) 數據集裡取 下一個50個樣本
    train_loss, train_op_ = sess.run([loss, train_op], {input_X: batch[0], output_y: batch[1]})
    if i % 100 == 0:
        test_accuracy = sess.run(accuracy, {input_X: test_X, output_y: test_y})
        print("Step=%d, Train loss=%.4f, [Test accuracy=%.2f]" % (i, train_loss, test_accuracy))

"""測試：打印 20 個預測值 和 真實值 的對應"""
test_output = sess.run(logits, {input_X: test_X[:20]})
inferenced_y = np.argmax(test_output, 1)
print(inferenced_y, 'Inferenced numbers')           # 推測的數字
print(np.argmax(test_y[:20], 1), 'Real numbers')    # 真實的數字
