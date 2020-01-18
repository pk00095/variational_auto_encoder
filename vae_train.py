#from .VAE import get_encoder, get_decoder
#from .losses import VaeLoss
from VAE import get_encoder, get_decoder, get_full_vae
from losses import VaeLoss
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

def get_vae_model():
    z_dim = 200
    enc_model, enc_input, enc_output, shape_before_flattening, mu, log_var = get_encoder(
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

    return vae_model, mu, log_var

def train():

	learning_rate = 0.0005


	model, mu, log_var = get_vae_model()
	vae_loss = VaeLoss(mu=mu, log_var=log_var, r_loss_factor=10000)

	model.compile(optimizer=Adam(learning_rate=learning_rate), loss=vae_loss)

	print(model.summary())


if __name__ == '__main__':
	train()