import tensorflow as tf

from ._custom_layers_and_blocks import ConvolutionBnActivation, Upsample_x2_Block
from ..backbones.tf_backbones import create_base_model

class UNet(tf.keras.Model):
    def __init__(self, n_classes, base_model, output_layers, height=None, width=None, filters=128,
                 final_activation="softmax", backbone_trainable=False,
                 up_filters=[32, 64, 128, 256, 512], include_top_conv=True, **kwargs):
        super().__init__(**kwargs)

        self.n_classes = n_classes
        self.filters = filters
        self.up_filters = up_filters
        self.include_top_conv = include_top_conv
        self.height = height
        self.width = width

        base_model.trainable = backbone_trainable
        self.backbone = tf.keras.Model(inputs=base_model.input, outputs=output_layers)

        if include_top_conv:
            self.conv3x3_bn_relu1 = ConvolutionBnActivation(filters, kernel_size=3, post_activation="relu")
            self.conv3x3_bn_relu2 = ConvolutionBnActivation(filters, kernel_size=3, post_activation="relu")

        self.upsample_blocks = [
            Upsample_x2_Block(up_filters[4]),
            Upsample_x2_Block(up_filters[3]),
            Upsample_x2_Block(up_filters[2]),
            Upsample_x2_Block(up_filters[1]),
            Upsample_x2_Block(up_filters[0])
        ]

        self.final_conv = tf.keras.layers.Conv2D(n_classes, 3, padding="same")
        self.final_activation = tf.keras.layers.Activation(final_activation)

    def call(self, inputs, training=False):
        features = self.backbone(inputs, training=training)
        x = features[4]

        if self.include_top_conv:
            conv1 = self.conv3x3_bn_relu1(inputs, training=training)
            conv1 = self.conv3x3_bn_relu2(conv1, training=training)
        else:
            conv1 = None

        x = self.upsample_blocks[0](x, features[3], training=training)
        x = self.upsample_blocks[1](x, features[2], training=training)
        x = self.upsample_blocks[2](x, features[1], training=training)
        x = self.upsample_blocks[3](x, features[0], training=training)
        x = self.upsample_blocks[4](x, conv1, training=training)

        x = self.final_conv(x)
        return self.final_activation(x)

    def model(self):
        x = tf.keras.Input(shape=(self.height, self.width, 3))
        return tf.keras.Model(inputs=x, outputs=self.call(x, training=False))
