import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.layers import (Input, Conv2D, Flatten, Dense,
                                    Conv2DTranspose, Reshape, Lambda, Activation, 
                                    BatchNormalization, LeakyReLU, Dropout)

from tensorflow.keras.models import Model

import numpy as np


def get_encoder(
    input_dim, 
    encoder_conv_filters,
    encoder_conv_kernel_size,
    encoder_conv_strides,
    z_dim,
    use_batch_norm=False,
    use_dropout=False):

    num_layers = len(encoder_conv_filters)
    

    encoder_input = Input(shape=input_dim, name='encoder_input')

    x = encoder_input
    #print(x.get_shape())

    for i in range(num_layers):

        conv_layer = Conv2D(
            filters=encoder_conv_filters[i],
            kernel_size=encoder_conv_kernel_size[i],
            strides=encoder_conv_strides[i],
            padding='same',
            name='encoder_conv_{}'.format(i))

        x = conv_layer(x)

        if use_batch_norm:
            x = BatchNormalization()(x)

        x = LeakyReLU()(x)

        if use_dropout:
            x = Dropout(rate=0.25)(x)

    shape_before_flattening = x.get_shape().as_list()[1:]

    x = Flatten()(x)

    mu = Dense(z_dim, name='mu')(x)
    log_var = Dense(z_dim, name='log_var')(x)

    encoder_mu_log_var = Model(encoder_input, (mu, log_var))

    def sampling(args):
        mu, log_var = args
        epsilon = tf.random.normal(
            shape=tf.shape(mu),
            mean=0,
            stddev=1.0)

        return mu + tf.exp(log_var/2)*epsilon

    encoder_output = Lambda(sampling, name='encoder_output')([mu, log_var])
    

    return Model(encoder_input, encoder_output, name='Encoder'), encoder_input, encoder_output, shape_before_flattening, mu, log_var

def get_decoder(
    z_dim,
    shape_before_flattening,
    decoder_conv_t_filters,
    decoder_conv_t_kernel_size,
    decoder_conv_t_strides,
    use_batch_norm=False,
    use_dropout=False):

    num_layers = len(decoder_conv_t_filters)
    
    decoder_input = Input(shape=(z_dim), name='decoder_input')

    x = Dense(np.prod(shape_before_flattening))(decoder_input)
    x = Reshape(shape_before_flattening)(x)

    for i in range(num_layers):
        conv_t_layer = Conv2DTranspose(
            filters=decoder_conv_t_filters[i],
            kernel_size=decoder_conv_t_kernel_size[i],
            strides=decoder_conv_t_strides[i],
            padding='same',
            name='decoder_conv_t_{}'.format(i))

        x = conv_t_layer(x)

        if i < num_layers-1 :
            if use_batch_norm:
                x = BatchNormalization()(x)

            x = LeakyReLU()(x)

            if use_dropout:
                x = Dropout(rate=0.25)(x)

        else:
            x = Activation('sigmoid')(x)

    decoder_output = x

    return Model(decoder_input, decoder_output, name='Decoder'), decoder_input, decoder_output

def get_full_vae(encoder_input, encoder_output, decoder):

    assert isinstance(decoder, Model), "expected decoder to be an instance of {}".format(Model)

    model_input = encoder_input
    model_output = decoder(encoder_output)
    
    return Model(model_input, model_output, name='Variational Auto Encoder')


def main():

    z_dim = 200
    enc_model, enc_input, enc_output, shape_before_flattening = get_encoder(
        input_dim=(128,128,3),
        encoder_conv_filters=[32,64,64, 64],
        encoder_conv_kernel_size=[3,3,3,3],
        encoder_conv_strides=[2,2,2,2],
        z_dim=z_dim)

    #print(enc_model.summary())

    dec_model, dec_input, dec_output = get_decoder(
        z_dim=z_dim,
        shape_before_flattening=shape_before_flattening,
        decoder_conv_t_filters=[64,64,32,3],
        decoder_conv_t_kernel_size=[3,3,3,3],
        decoder_conv_t_strides=[2,2,2,2])

    vae_model = get_full_vae(
        encoder_input=enc_input, 
        encoder_output=enc_output, 
        decoder=dec_model)

    print(vae_model.summary())


if __name__ == '__main__':
    main()