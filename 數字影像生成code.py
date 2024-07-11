import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt

def make_generator_network(                            #定義一個generator
        num_hidden_layers=1,
        num_hidden_units=100,
        num_output_units=784):
    model = tf.keras.Sequential()
    for i in range(num_hidden_layers):
        model.add(
            tf.keras.layers.Dense(
                units=num_hidden_units,
                use_bias=False)
          )
        model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Dense(
        units=num_output_units, activation="tanh"))
    return model

## define a function for the discriminator:             #定義一個discriminator
def make_discriminator_network(
        num_hidden_layers=1,
        num_hidden_units=100,
        num_output_units=1):
    model = tf.keras.Sequential()
    for i in range(num_hidden_layers):
        model.add(tf.keras.layers.Dense(units=num_hidden_units))
        model.add(tf.keras.layers.LeakyReLU())
        model.add(tf.keras.layers.Dropout(rate=0.5))

    model.add(
        tf.keras.layers.Dense(
            units=num_output_units,
            activation=None)
    )
    return model



image_size =(28, 28)    #圖片大小是28*28
z_size = 20
mode_z = "uniform"      # 'uniform' vs. 'normal'
gen_hidden_layers = 1
gen_hidden_size = 100
disc_hidden_layers = 1
disc_hidden_size = 100

tf.random.set_seed(1)

gen_model = make_generator_network(
    num_hidden_layers=gen_hidden_layers,
    num_hidden_units=gen_hidden_size,
    num_output_units=np.prod(image_size))

gen_model.build(input_shape=(None, z_size))
gen_model.summary()



disc_model = make_discriminator_network(
    num_hidden_layers=disc_hidden_layers,
    num_hidden_units=disc_hidden_size)

disc_model.build(input_shape=(None, np.prod(image_size)))
disc_model.summary()



mnist_bldr = tfds.builder("mnist")                           #Defining the training dataset
mnist_bldr.download_and_prepare()
mnist = mnist_bldr.as_dataset(shuffle_files=False)

def preprocess(ex, mode="uniform"):
    image = ex["image"]
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.reshape(image, [-1])
    image = image*2 - 1.0                                   #我們將圖放大兩倍，並移位-1，以便將像素強度重新縮放到[-1,1]之間
    if mode == "uniform":  #『均勻分佈』
        input_z = tf.random.uniform(                        #我們建立一個隨機向量z
            shape=(z_size,), minval=-1.0, maxval=1.0)       #minval 和 maxval 參數定義了可以抽樣的值的範圍，並且該範圍內的每個值具有相等的被選擇機率
    elif mode == "normal":  #『正態（高斯）分佈』
        input_z = tf.random.normal(shape=(z_size,))
    return input_z, image     #回傳圖片與隨機向量z

mnist_trainset = mnist["train"]                            

print('Before preprocessing:  ')
example = next(iter(mnist_trainset))["image"]
print('dtype: ', example.dtype, ' Min: {} Max: {}'.format(np.min(example), np.max(example)))

mnist_trainset = mnist_trainset.map(preprocess)      
print("After preprocessing: ")
example = next(iter(mnist_trainset))[0]
print("dtype: ", example.dtype, "Min: {} Max: {}".format(np.min(example), np.max(example)))



loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)      #我們建立一個BinaryCrossentropy類別物件來當作損失函數，並使用來計算與「剛準備好的批次數據」
                                                                    #相關的「生成器」和 「鑑別器」的損失
## Loss for the Generator
g_labels_real = tf.ones_like(d_logits_fake)
g_loss = loss_fn(y_true=g_labels_real, y_pred=d_logits_fake)
print("Generator Loss: {:.4f}".format(g_loss))

##Loss for the Discriminator
d_labels_real = tf.ones_like(d_logits_real)                         #0用於「真樣本」； 1用於「偽樣本」==> 為了避免vanishing gradient，我們將「標籤替換」 P206有詳細說明
d_labels_fake = tf.zeros_like(d_logits_fake)

d_loss_real = loss_fn(y_true=d_labels_real, y_pred=d_logits_real)
d_loss_fake = loss_fn(y_true=d_labels_fake, y_pred=d_logits_fake)
print('Discriminator Losses: Real {:.4f} Fake {:.4f}'
      .format(d_loss_real.numpy(), d_loss_fake.numpy()))



import time                                                         @ Final training

num_epochs = 40
batch_size = 64
image_size = (28,28)
z_size = 20
mode_z = "uniform"
gen_hidden_layers = 1
gen_hidden_size = 100
disc_hidden_layers =1
disc_hidden_size = 100

tf.random.set_seed(1)
np.random.seed(1)

if mode_z == "uniform":
    fixed_z = tf.random.uniform(
        shape=(batch_size, z_size),
        minval=-1, maxval=1)
elif mode_z == "normal":
    fixed_z = tf.random.normal(
        shape=(batch_size, z_size))

def create_samples(g_model, input_z):
    g_output = g_model(input_z, training=False)
    images = tf.reshape(g_output, (batch_size, *image_size))
    return (images+1)/2.0

## Set-up the dataset
mnist_trainset = mnist["train"]
mnist_trainset = mnist_trainset.map(
    lambda ex: preprocess(ex, mode=mode_z))

mnist_trainset = mnist_trainset.shuffle(10000)
mnist_trainset = mnist_trainset.batch(
    batch_size, drop_remainder=True)

device_name = "GPU"

## Set-up the model
with tf.device(device_name):
    gen_model = make_generator_network(
        num_hidden_layers=gen_hidden_layers,
        num_hidden_units=gen_hidden_size,
        num_output_units=np.prod(image_size))                 #它用於計算給定數組中所有元素的乘積:
    gen_model.build(input_shape=(None, z_size))               #例如，如果 image_size 是 (height, width, channels)，則 np.prod(image_size) 將等於 height * width * channels，這代表圖像的總像素數量。

    disc_model = make_discriminator_network(
        num_hidden_layers=disc_hidden_layers,
        num_hidden_units=disc_hidden_size)
    disc_model.build(input_shape=(None, np.prod(image_size)))

## Loss function and optimizers:
loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
g_optimizer = tf.keras.optimizers.Adam()
d_optimizer = tf.keras.optimizers.Adam()

all_losses = []
all_d_vals = []
epoch_samples = []

start_time = time.time()
for epoch in range(1, num_epochs+1):
    epoch_losses, epoch_d_vals=[], []
    for i, (input_z, input_real) in enumerate(mnist_trainset):

        ## Compute generator's loss
        with tf.GradientTape() as g_tape:
            g_output = gen_model(input_z)
            d_logits_fake = disc_model(g_output, training=True)
            labels_real = tf.ones_like(d_logits_fake)
            g_loss = loss_fn(y_true=labels_real, y_pred=d_logits_fake)

        g_grads = g_tape.gradient(g_loss, gen_model.trainable_variables)      #用於根據生成器的損失計算梯度
        g_optimizer.apply_gradients(                                          #然後使用優化器更新生成器的權重
            grads_and_vars=zip(g_grads, gen_model.trainable_variables))

        ## Compute discriminator's loss
        with tf.GradientTape() as d_tape:
            d_logits_real = disc_model(input_real, training=True)

            d_labels_real = tf.ones_like(d_logits_real)

            d_loss_real = loss_fn(
                y_true=d_labels_real, y_pred=d_logits_real)

            d_logits_fake = disc_model(g_output, training=True)
            d_labels_fake = tf.zeros_like(d_logits_fake)

            d_loss_fake = loss_fn(
                y_true=d_labels_fake, y_pred=d_logits_fake)

            d_loss = d_loss_real + d_loss_fake

        ## Compute the gradients of d_loss
        d_grads = d_tape.gradient(d_loss, disc_model.trainable_variables)

        ## Optimization: Apply the gradients
        d_optimizer.apply_gradients(
            grads_and_vars=zip(d_grads, disc_model.trainable_variables))

        epoch_losses.append(
            (g_loss.numpy(), d_loss.numpy(),
             d_loss_real.numpy(), d_loss_fake.numpy()))

        d_probs_real = tf.reduce_mean(tf.sigmoid(d_logits_real)) #sigmoid 激活通常用於將 logits 壓縮到介於 0 到 1 之間，表示概率，並計算其平均
        d_probs_fake = tf.reduce_mean(tf.sigmoid(d_logits_fake))
        epoch_d_vals.append((d_probs_real.numpy(), d_probs_fake.numpy()))
    all_losses.append(epoch_losses)
    all_d_vals.append(epoch_d_vals)
    print(
        "Epoch {:03d} | ET {:.2f} min | Avg Losses >>"
        "G/D {: .4f}/{: .4f} [D-Real: {:.4f} D-Fake: {:.4f}]"
        .format(
            epoch, (time.time() - start_time)/60,
            *list(np.mean(all_losses[-1],axis=0))))
    epoch_samples.append(
        create_samples(gen_model, fixed_z).numpy())


