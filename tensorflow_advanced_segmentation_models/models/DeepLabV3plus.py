import tensorflow as tf

from ._custom_layers_and_blocks import (
    ConvolutionBnActivation,
    AtrousSeparableConvolutionBnReLU,
    AtrousSpatialPyramidPoolingV3
)
from ..backbones.tf_backbones import create_base_model

class DeepLabV3plus(tf.keras.Model):
    def __init__(self, n_classes, base_model, output_layers, height=None, width=None, filters=256,
                 final_activation="softmax", backbone_trainable=False,
                 output_stride=8, dilations=[6, 12, 18], **kwargs):
        super().__init__(**kwargs)

        self.n_classes = n_classes
        self.filters = filters
        self.final_activation_str = final_activation
        self.output_stride = output_stride
        self.dilations = dilations
        self.height = height
        self.width = width

        if self.output_stride == 8:
            self.upsample_encoder = tf.keras.layers.UpSampling2D(size=2, interpolation="bilinear")
            output_layers = output_layers[:3]
            self.dilations = [2 * rate for rate in dilations]
        elif self.output_stride == 16:
            self.upsample_encoder = tf.keras.layers.UpSampling2D(size=4, interpolation="bilinear")
            output_layers = output_layers[:4]
        else:
            raise ValueError(f"'output_stride' must be one of (8, 16), got {self.output_stride}")

        base_model.trainable = backbone_trainable
        self.backbone = tf.keras.Model(inputs=base_model.input, outputs=output_layers)

        # Encoder
        self.atrous_sepconv_bn_relu_1 = AtrousSeparableConvolutionBnReLU(dilation=2, filters=filters, kernel_size=3)
        self.aspp = AtrousSpatialPyramidPoolingV3(self.dilations, filters)
        self.conv1x1_bn_relu_1 = ConvolutionBnActivation(filters, 1)

        # Decoder
        self.atrous_sepconv_bn_relu_2 = AtrousSeparableConvolutionBnReLU(dilation=2, filters=filters, kernel_size=3)
        self.conv1x1_bn_relu_2 = ConvolutionBnActivation(64, 1)

        self.concat = tf.keras.layers.Concatenate(axis=-1)
        self.conv3x3_bn_relu_1 = ConvolutionBnActivation(filters, 3)
        self.conv3x3_bn_relu_2 = ConvolutionBnActivation(filters, 3)
        self.conv1x1_logits = ConvolutionBnActivation(n_classes, 1, post_activation="linear")

        self.upsample_output = tf.keras.layers.UpSampling2D(size=4, interpolation="bilinear")
        self.final_activation = tf.keras.layers.Activation(final_activation)

    def call(self, inputs, training=False):
        features = self.backbone(inputs, training=training)
        x = features[-1]
        low_level = features[1]

        x = self.atrous_sepconv_bn_relu_1(x, training=training)
        x = self.aspp(x, training=training)
        x = self.conv1x1_bn_relu_1(x, training=training)
        x = self.upsample_encoder(x)

        low_level = self.atrous_sepconv_bn_relu_2(low_level, training=training)
        low_level = self.conv1x1_bn_relu_2(low_level, training=training)

        x = self.concat([x, low_level])
        x = self.conv3x3_bn_relu_1(x, training=training)
        x = self.conv3x3_bn_relu_2(x, training=training)
        x = self.conv1x1_logits(x, training=training)
        x = self.upsample_output(x)
        return self.final_activation(x)

    def model(self):
        x = tf.keras.Input(shape=(self.height, self.width, 3))
        return tf.keras.Model(inputs=x, outputs=self.call(x, training=False))
