import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers, models, optimizers, Input
from tensorflow.keras.saving import register_keras_serializable
from tensorflow.keras.layers import (
    Conv1D,
    MaxPooling1D,
    Dense,
    Bidirectional,
    BatchNormalization,
    Activation,
    Add,
    LSTM,
    Flatten,
    Attention,
    Concatenate,
    concatenate,
    Conv1DTranspose,
    GlobalAveragePooling1D,
    MultiHeadAttention,
)

@register_keras_serializable()
class Residual1DConv(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides=1, activation="relu", **kwargs):
        super(Residual1DConv, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.activation_name = activation
        self.conv1 = layers.Conv1D(
            filters, kernel_size, strides=strides, padding="same"
        )
        self.bn1 = layers.BatchNormalization()
        self.activation = layers.Activation(activation)
        self.conv2 = layers.Conv1D(
            filters, kernel_size, strides=strides, padding="same"
        )
        self.bn2 = layers.BatchNormalization()
        self.add = layers.Add()
        self.final_activation = layers.Activation(activation)

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.add([x, inputs])
        return self.final_activation(x)

    def get_config(self):
        config = super(Residual1DConv, self).get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "strides": self.strides,
                "activation": self.activation_name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

@register_keras_serializable()
class MidPointResidual1DConv(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides=1, activation="relu", **kwargs):
        super(MidPointResidual1DConv, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.activation_name = activation
        self.conv = layers.Conv1D(filters, kernel_size, strides=strides, padding="same")
        self.bn = layers.BatchNormalization()
        self.activation = layers.Activation(activation)
        self.add = layers.Add()

    def call(self, inputs, training=False):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        x = self.activation(x)
        x = self.conv(x)
        x = self.bn(x, training=training)
        x = self.activation(x)
        x = self.add([x, inputs])

        x = self.conv(inputs)
        x = self.bn(x, training=training)
        x = self.activation(x)
        x = self.conv(x)
        x = self.bn(x, training=training)
        x = self.activation(x)
        x = self.add([x, inputs])

        return x

    def get_config(self):
        config = super(MidPointResidual1DConv, self).get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "strides": self.strides,
                "activation": self.activation_name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

#### List of Model
# MRLAN
# AMPCNN
# MCLDRL
# CNNBiLSTM
# MSCMpResCNN
# MSCNNLSTM

# MRLAN Model
def MRLAN(input_shape, num_classes):
    # Input layer
    inputs = Input(shape=input_shape)

    # Main feature extraction layers
    conv1d_1 = Conv1D(filters=10, kernel_size=8, activation="relu", padding="same", strides=1)(inputs)
    maxpool1d_1 = MaxPooling1D(pool_size=2, strides=2, padding="same")(conv1d_1)
    conv1d_2 = Conv1D(filters=20, kernel_size=16, activation="relu", padding="same", strides=1)(maxpool1d_1)
    maxpool1d_2 = MaxPooling1D(pool_size=2, strides=2, padding="same")(conv1d_2)

    # Head 1
    conv1d_1a = Conv1D(filters=40, kernel_size=32, activation="relu", padding="same", strides=1)(maxpool1d_2)
    maxpool1d_1a = MaxPooling1D(pool_size=2, strides=2)(conv1d_1a)
    res1d_1a = Residual1DConv(filters=40, kernel_size=3, activation="relu")(maxpool1d_1a)
    maxpool1d_2a = MaxPooling1D(pool_size=2, strides=2, padding="same")(res1d_1a)
    lstm_1a = LSTM(40, return_sequences=True)(maxpool1d_2a)
    mha1a = MultiHeadAttention(num_heads=10, key_dim=40)(lstm_1a, lstm_1a)

    # Head 2
    conv1d_1b = Conv1D(filters=20, kernel_size=8, activation="relu", padding="same", strides=1)(maxpool1d_2)
    maxpool1d_1b = MaxPooling1D(pool_size=2, strides=2)(conv1d_1b)
    res1d_1b = Residual1DConv(filters=20, kernel_size=3, activation="relu")(maxpool1d_1b)
    maxpool1d_2b = MaxPooling1D(pool_size=2, strides=2, padding="same")(res1d_1b)
    lstm_1b = LSTM(20, return_sequences=True)(maxpool1d_2b)
    mha1b = MultiHeadAttention(num_heads=10, key_dim=20)(lstm_1b, lstm_1b)

    # Head 3
    conv1d_1c = Conv1D(filters=10, kernel_size=2, activation="relu", padding="same", strides=1)(maxpool1d_2)
    maxpool1d_1c = MaxPooling1D(pool_size=2, strides=2)(conv1d_1c)
    res1d_1c = Residual1DConv(filters=10, kernel_size=3, activation="relu")(maxpool1d_1c)
    maxpool1d_2c = MaxPooling1D(pool_size=2, strides=2, padding="same")(res1d_1c)
    lstm_1c = LSTM(10, return_sequences=True)(maxpool1d_2c)
    mha1c = MultiHeadAttention(num_heads=10, key_dim=10)(lstm_1c, lstm_1c)

    # Concat and Compile Layer
    concat_1 = concatenate([mha1a, mha1b, mha1c])

    maxpool1d_3 = MaxPooling1D(pool_size=2, strides=2, padding="same")(concat_1)
    flatten1 = Flatten()(maxpool1d_3)
    dense_1 = Dense(64, activation="relu")(flatten1)
    outputs = Dense(num_classes, activation="softmax")(dense_1)

    optimizer = optimizers.Adam(learning_rate=1e-5,beta_1=0.9,beta_2=0.99,epsilon=1e-08,amsgrad=False,name="Adam")
    model = models.Model(inputs, outputs)
    model.compile(loss="categorical_crossentropy",optimizer=optimizer,metrics=["accuracy"])
    return model


# AMPCNN Model
def AMPCNN(input_shape, num_classes):
    # Input layer
    inputs = Input(shape=input_shape)

    # Head1_Part1
    conv1d_11 = Conv1D(filters=16, kernel_size=128, activation="relu", strides=8, padding="same")(inputs)
    mp11 = MaxPooling1D(pool_size=2, strides=2, padding="same")(conv1d_11)
    am11 = Attention()([mp11, mp11])
    dense11 = Dense(16, activation="relu")(am11)
    concat11 = Concatenate()([mp11, dense11])

    conv1d_13 = Conv1D(filters=32, kernel_size=3, activation="relu", strides=1, padding="same")(concat11)
    mp13 = MaxPooling1D(pool_size=2, strides=2, padding="same")(conv1d_13)
    conv1d_14 = Conv1D(filters=64, kernel_size=3, activation="relu", strides=1, padding="same")(mp13)
    mp14 = MaxPooling1D(pool_size=2, strides=2, padding="same")(conv1d_14)
    conv1d_15 = Conv1D(filters=64, kernel_size=3, activation="relu", strides=1, padding="same")(mp14)
    mp15 = MaxPooling1D(pool_size=2, strides=2, padding="same")(conv1d_15)
    conv1d_16 = Conv1D(filters=64, kernel_size=3, activation="relu", strides=2, padding="valid")(mp15)
    mp16 = MaxPooling1D(pool_size=2, strides=2, padding="same")(conv1d_16)
    gap11 = GlobalAveragePooling1D()(mp15)
    gap12 = GlobalAveragePooling1D()(mp16)

    # Head1_Part2
    conv1d_21 = Conv1D(filters=16, kernel_size=12, activation="relu", strides=4, padding="same")(inputs)
    mp21 = MaxPooling1D(pool_size=2, strides=2, padding="same")(conv1d_21)
    am21 = Attention()([mp21, mp21])
    dense21 = Dense(16, activation="relu")(am21)
    concat21 = Concatenate()([mp21, dense21])
    conv1d_22 = Conv1D(filters=32, kernel_size=6, activation="relu", strides=4, padding="same")(concat21)
    mp22 = MaxPooling1D(pool_size=2, strides=2, padding="same")(conv1d_22)
    conv1d_23 = Conv1D(filters=64, kernel_size=3, activation="relu", strides=2, padding="same")(mp22)
    tranconv1d_21 = Conv1DTranspose(filters=64, kernel_size=3, activation="relu", strides=2, padding="same")(conv1d_23)
    mp23 = MaxPooling1D(pool_size=2, strides=2, padding="same")(tranconv1d_21)
    tranconv1d_22 = Conv1DTranspose(filters=32, kernel_size=3, activation="relu", strides=2, padding="same")(mp23)
    mp24 = MaxPooling1D(pool_size=2, strides=2, padding="same")(tranconv1d_22)
    tranconv1d_23 = Conv1DTranspose(filters=16, kernel_size=3, activation="relu", strides=2, padding="same")(mp24)
    gap21 = GlobalAveragePooling1D()(tranconv1d_23)
    gap22 = GlobalAveragePooling1D()(mp24)

    # Combine Model
    concatfinal = Concatenate()([gap11, gap12, gap21, gap22])
    dense = Dense(100, activation="relu")(concatfinal)
    outputs = Dense(num_classes, activation="softmax")(dense)

    optimizer = optimizers.Adam(learning_rate=0.001,beta_1=0.9,beta_2=0.99,epsilon=1e-08,amsgrad=False,name="Adam",)
    model = models.Model(inputs, outputs)
    model.compile(loss="categorical_crossentropy",optimizer=optimizer,metrics=["accuracy"])

    return model


# MCLDRL Model
def MCLDRL(input_shape, num_classes):
    # Input layer
    inputs = Input(shape=input_shape)

    conv1d_1 = Conv1D(filters=64, kernel_size=64, activation="relu", padding="valid", strides=7)(inputs)
    maxpool_1 = MaxPooling1D(pool_size=3, strides=1, padding="valid")(conv1d_1)

    # Head 1
    conv1d_1a = Conv1D(filters=32, kernel_size=32, activation="relu", padding="same", strides=3)(maxpool_1)
    bn1a = BatchNormalization()(conv1d_1a)
    lstm_1a = LSTM(32, return_sequences=True)(bn1a)

    # Head 2
    conv1d_2a = Conv1D(filters=32, kernel_size=16, activation="relu", padding="same", strides=3)(maxpool_1)
    bn2a = BatchNormalization()(conv1d_2a)
    lstm_2a = LSTM(32, return_sequences=True)(bn2a)

    # Head 3
    conv1d_3a = Conv1D(filters=32, kernel_size=8, activation="relu", padding="same", strides=3)(maxpool_1)
    bn3a = BatchNormalization()(conv1d_3a)
    lstm_3a = LSTM(32, return_sequences=True)(bn3a)
    concat1 = concatenate([lstm_1a, lstm_2a, lstm_3a], axis=1)
    maxpool_2 = MaxPooling1D(pool_size=3, strides=1, padding="valid")(concat1)

    conv1d_5 = Conv1D(filters=32, kernel_size=58, activation="relu", padding="same", strides=3)(maxpool_2)
    bn5 = BatchNormalization()(conv1d_5)
    con1d_6 = Conv1D(filters=32, kernel_size=8, activation="relu", padding="same", strides=3)(bn5)
    bn6 = BatchNormalization()(con1d_6)
    lstm1 = LSTM(32, return_sequences=True)(bn6)
    concat2 = concatenate([lstm1, maxpool_2], axis=1)
    flatten1 = Flatten()(concat2)
    dense1 = Dense(64, activation="relu")(flatten1)
    outputs = Dense(num_classes, activation="softmax")(dense1)

    optimizer = optimizers.Adam(learning_rate=1e-5, beta_1=0.9, beta_2=0.99, epsilon=1e-08, amsgrad=False, name="Adam",)
    model = models.Model(inputs, outputs)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    return model


# CNN-BiLSTM Model
def CNNBiLSTM(input_shape, num_classes):
    # Input layer
    inputs = Input(shape=input_shape)

    # Main feature extraction layers
    conv1d_1 = Conv1D(filters=16, kernel_size=8, activation="relu", padding="same", strides=1)(inputs)
    maxpool1d_1 = MaxPooling1D(pool_size=2, strides=2, padding="same")(conv1d_1)
    conv1d_2 = Conv1D(filters=32, kernel_size=16, activation="relu", padding="same", strides=1)(maxpool1d_1)
    maxpool1d_2 = MaxPooling1D(pool_size=2, strides=2, padding="same")(conv1d_2)

    # Head 1
    conv1d_1a = Conv1D(filters=64, kernel_size=32, activation="relu", padding="same", strides=1)(maxpool1d_2)
    maxpool1d_1a = MaxPooling1D(pool_size=2, strides=2)(conv1d_1a)
    conv1d_2a = Conv1D(filters=128, kernel_size=32, activation="relu", padding="same", strides=1)(maxpool1d_1a)
    maxpool1d_2a = MaxPooling1D(pool_size=2, strides=2, padding="same")(conv1d_2a)
    bilstm_1a = Bidirectional(LSTM(128))(maxpool1d_2a)

    # Concat and Compile Layer
    # flatten1 = Flatten()(bilstm_1a)
    dense_1 = Dense(512, activation="relu")(bilstm_1a)
    outputs = Dense(num_classes, activation="softmax")(dense_1)

    optimizer = optimizers.Adam(learning_rate=0.0003,beta_1=0.9, beta_2=0.99,epsilon=1e-08,amsgrad=False,name="Adam",)
    model = models.Model(inputs, outputs)
    model.compile(loss="categorical_crossentropy",optimizer=optimizer, metrics=["accuracy"],)

    return model


# MSC-MpResCNN Model
def MSCMpResCNN(input_shape, num_classes):
    model = Sequential()

    # Input layer
    inputs = Input(shape=input_shape)

    # Main
    conv1d_1 = Conv1D(filters=16, kernel_size=64, activation="relu", padding="same", strides=16)(inputs)
    bn1 = BatchNormalization()(conv1d_1)
    act1 = Activation("relu")(bn1)
    mp1 = MaxPooling1D(pool_size=2, strides=2, padding="same")(act1)

    # Head 1
    mprb1a = MidPointResidual1DConv(filters=16, kernel_size=9, strides=1)(mp1)
    mprb1b = MidPointResidual1DConv(filters=16, kernel_size=6, strides=1)(mp1)
    mprb1c = MidPointResidual1DConv(filters=16, kernel_size=3, strides=1)(mp1)

    concat1 = concatenate([mprb1a, mprb1b, mprb1c], axis=1)
    gap = GlobalAveragePooling1D()(concat1)
    outputs = Dense(num_classes, activation="softmax")(gap)

    model = models.Model(inputs=inputs, outputs=outputs)
    optimizer = optimizers.Adam(learning_rate=1e-4,beta_1=0.9,beta_2=0.99,epsilon=1e-08, amsgrad=False, name="Adam",)
    model = models.Model(inputs, outputs)
    model.compile( loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"],)

    return model


# MSCNN-LSTM Model
def MSCNNLSTM(input_shape, num_classes):
    # Input layer
    inputs = Input(shape=input_shape)

    # Head 1
    conv1d_1 = Conv1D(filters=64, kernel_size=5, activation="relu", strides=1, padding="same")(inputs)
    conv1d_2 = Conv1D(filters=64, kernel_size=5, activation="relu", strides=1, padding="same")(conv1d_1)
    conv1d_3 = Conv1D(filters=64, kernel_size=5, activation="relu", strides=1, padding="same")(conv1d_2)
    conv1d_4 = Conv1D(filters=64, kernel_size=5, activation="relu", strides=1, padding="same")(conv1d_3)
    conv1d_5 = Conv1D(filters=64, kernel_size=5, activation="relu", strides=1, padding="same")(conv1d_4)
    conv1d_6 = Conv1D(filters=64, kernel_size=5, activation="relu", strides=1, padding="same")(conv1d_5)
    conv1d_7 = Conv1D(filters=64, kernel_size=5, activation="relu", strides=1, padding="same")(conv1d_6)
    conv1d_8 = Conv1D(filters=1, kernel_size=64, activation="relu", strides=1, padding="same")(conv1d_7)
    add1 = Add()([conv1d_8, conv1d_1])

    conv1d_a1 = Conv1D(filters=8, kernel_size=7, activation="relu", strides=1, padding="same")(add1)
    conv1d_a2 = Conv1D(filters=8, kernel_size=7, strides=1, padding="same")(conv1d_a1)
    conv1d_a3 = Conv1D(filters=8, kernel_size=1, strides=1, padding="same")(add1)
    adda1 = Add()([conv1d_a2, conv1d_a3])
    acta1 = Activation("relu")(adda1)
    mpa1 = MaxPooling1D(pool_size=2, strides=2, padding="same")(acta1)

    conv1d_a4 = Conv1D(filters=16, kernel_size=7, activation="relu", strides=1, padding="same")(mpa1)
    conv1d_a5 = Conv1D(filters=16, kernel_size=7, strides=1, padding="same")(conv1d_a4)
    conv1d_a6 = Conv1D(filters=16, kernel_size=1, strides=1, padding="same")(mpa1)
    adda2 = Add()([conv1d_a5, conv1d_a6])
    acta2 = Activation("relu")(adda2)
    mpa2 = MaxPooling1D(pool_size=2, strides=2, padding="same")(acta2)

    conv1d_a7 = Conv1D(filters=32, kernel_size=7, activation="relu", strides=1, padding="same")(mpa2)
    conv1d_a8 = Conv1D(filters=32, kernel_size=7, strides=1, padding="same")(conv1d_a7)
    conv1d_a9 = Conv1D(filters=32, kernel_size=1, strides=1, padding="same")(mpa2)
    adda3 = Add()([conv1d_a8, conv1d_a9])
    acta3 = Activation("relu")(adda3)
    mpa3 = MaxPooling1D(pool_size=2, strides=2, padding="same")(acta3)

    conv1d_a10 = Conv1D(filters=64, kernel_size=7, activation="relu", strides=1, padding="same")(mpa3)
    conv1d_a11 = Conv1D(filters=64, kernel_size=7, strides=1, padding="same")(conv1d_a10)
    conv1d_a12 = Conv1D(filters=64, kernel_size=1, strides=1, padding="same")(mpa3)
    adda4 = Add()([conv1d_a11, conv1d_a12])
    acta4 = Activation("relu")(adda4)
    mpa4 = MaxPooling1D(pool_size=2, strides=2, padding="same")(acta4)

    lstm1 = LSTM(128, return_sequences=True)(mpa4)
    flatten = Flatten()(lstm1)
    outputs = Dense(num_classes, activation="softmax")(flatten)

    optimizer = optimizers.Adam(learning_rate=0.001,beta_1=0.9,beta_2=0.99,epsilon=1e-08,amsgrad=False,name="Adam")
    model = models.Model(inputs, outputs)
    model.compile(loss="categorical_crossentropy",optimizer=optimizer,metrics=["accuracy"])

    return model

# Basic CNN Model
def CNN(input_shape, num_classes):
    # Input layer
    inputs = Input(shape=input_shape)

    # Main feature extraction layers
    conv1d_1 = Conv1D(filters=10, kernel_size=8, activation="relu", padding="same", strides=1)(inputs)
    maxpool1d_1 = MaxPooling1D(pool_size=2, strides=2, padding="same")(conv1d_1)
    conv1d_2 = Conv1D(filters=20, kernel_size=16, activation="relu", padding="same", strides=1)(maxpool1d_1)
    maxpool1d_2 = MaxPooling1D(pool_size=2, strides=2, padding="same")(conv1d_2)

    # Head 1
    conv1d_1a = Conv1D(filters=40, kernel_size=32, activation="relu", padding="same", strides=1)(maxpool1d_2)
    maxpool1d_1a = MaxPooling1D(pool_size=2, strides=2)(conv1d_1a)
    flatten1 = Flatten()(maxpool1d_1a)
    dense_1 = Dense(64, activation="relu")(flatten1)
    outputs = Dense(num_classes, activation="softmax")(dense_1)

    optimizer = optimizers.Adam(learning_rate=1e-4,beta_1=0.9,beta_2=0.99,epsilon=1e-08,amsgrad=False,name="Adam",)

    model = models.Model(inputs, outputs)
    model.compile(loss="categorical_crossentropy",optimizer=optimizer,metrics=["accuracy"],)
    return model

# CNN-Res Model
def CNNRes(input_shape, num_classes):
    # Input layer
    inputs = Input(shape=input_shape)

    # Main feature extraction layers
    conv1d_1 = Conv1D(filters=10, kernel_size=8, activation="relu", padding="same", strides=1)(inputs)
    maxpool1d_1 = MaxPooling1D(pool_size=2, strides=2, padding="same")(conv1d_1)
    conv1d_2 = Conv1D(filters=20, kernel_size=16, activation="relu", padding="same", strides=1)(maxpool1d_1)
    maxpool1d_2 = MaxPooling1D(pool_size=2, strides=2, padding="same")(conv1d_2)

    # Head 1
    conv1d_1a = Conv1D(filters=40, kernel_size=32, activation="relu", padding="same", strides=1)(maxpool1d_2)
    maxpool1d_1a = MaxPooling1D(pool_size=2, strides=2)(conv1d_1a)
    res1d_1a = Residual1DConv(filters=40, kernel_size=3, activation="relu")(maxpool1d_1a)
    maxpool1d_2a = MaxPooling1D(pool_size=2, strides=2, padding="same")(res1d_1a)

    flatten1 = Flatten()(maxpool1d_2a)
    dense_1 = Dense(64, activation="relu")(flatten1)
    outputs = Dense(num_classes, activation="softmax")(dense_1)
    optimizer = optimizers.Adam(learning_rate=1e-4,beta_1=0.9,beta_2=0.99,epsilon=1e-08,amsgrad=False,name="Adam")

    model = models.Model(inputs, outputs)
    model.compile(loss="categorical_crossentropy",optimizer=optimizer,metrics=["accuracy"])
    return model

# CNN-LSTM Model
def CNNLSTM(input_shape, num_classes):
    # Input layer
    inputs = Input(shape=input_shape)

    # Main feature extraction layers
    conv1d_1 = Conv1D(filters=10, kernel_size=8, activation="relu", padding="same", strides=1)(inputs)
    maxpool1d_1 = MaxPooling1D(pool_size=2, strides=2, padding="same")(conv1d_1)

    lstm_1a = LSTM(40, return_sequences=True)(maxpool1d_1)
    maxpool1d_3 = MaxPooling1D(pool_size=2, strides=2, padding="same")(lstm_1a)
    flatten1 = Flatten()(maxpool1d_3)
    dense_1 = Dense(64, activation="relu")(flatten1)
    outputs = Dense(num_classes, activation="softmax")(dense_1)

    optimizer = optimizers.Adam(learning_rate=1e-4,beta_1=0.9,beta_2=0.99,epsilon=1e-08,amsgrad=False,name="Adam")

    model = models.Model(inputs, outputs)
    model.compile(loss="categorical_crossentropy",optimizer=optimizer,metrics=["accuracy"])
    return model

# CNN-MHA Model
def CNNMHA(input_shape, num_classes):
    # Input layer
    inputs = Input(shape=input_shape)

    # Main feature extraction layers
    conv1d_1 = Conv1D(filters=10, kernel_size=8, activation="relu", padding="same", strides=1)(inputs)
    maxpool1d_1 = MaxPooling1D(pool_size=2, strides=2, padding="same")(conv1d_1)
    selfAttention1a = MultiHeadAttention(num_heads=10, key_dim=20)(maxpool1d_1, maxpool1d_1)
    maxpool1d_3 = MaxPooling1D(pool_size=2, strides=2, padding="same")(selfAttention1a)
    flatten1 = Flatten()(maxpool1d_3)
    dense_1 = Dense(64, activation="relu")(flatten1)
    outputs = Dense(num_classes, activation="softmax")(dense_1)

    optimizer = optimizers.Adam(learning_rate=1e-5,beta_1=0.9,beta_2=0.99,epsilon=1e-08,amsgrad=False,name="Adam")
    model = models.Model(inputs, outputs)
    model.compile(loss="categorical_crossentropy",optimizer=optimizer,metrics=["accuracy"])
    return model
