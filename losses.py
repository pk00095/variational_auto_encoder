import tensorflow as tf
from tensorflow.python.keras.losses import LossFunctionWrapper
from tensorflow.python.keras.utils import losses_utils


class VaeLoss(LossFunctionWrapper):
	"""docstring for VaeLoss"""
	def __init__(self, mu, log_var, r_loss_factor, name='vae_loss'):
		super(VaeLoss, self).__init__(
			fn=self.vae_loss,
			reduction=losses_utils.ReductionV2.AUTO,
			name=name)
			#mu=mu,
			#log_var=log_var,
			#r_loss_factor=r_loss_factor)
		self.mu = mu
		self.log_var = log_var
		self.r_loss_factor = r_loss_factor

	def vae_r_loss(self, y_true, y_pred):
		r_loss = tf.reduce_mean(tf.square(y_true - y_pred), axis=[1,2,3])
		return r_loss* self.r_loss_factor

	def vae_kl_loss(self, y_true, y_pred):
		#1 + self.log_var - K.square(self.mu) - K.exp(self.log_var)
		kl_loss = -0.5 * tf.reduce_sum(1+self.log_var - tf.square(self.mu) - tf.exp(self.log_var), axis=1)
		return kl_loss	

	def vae_loss(self, y_true, y_pred):
		r_loss = self.vae_r_loss(y_true, y_pred)
		kl_loss = self.vae_kl_loss(y_true, y_pred)

		return r_loss+kl_loss	



