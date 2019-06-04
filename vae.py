# Copyright 2018 The TensorFlow Probability Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Trains a variational auto-encoder (VAE) on binarized MNIST.
The VAE defines a generative model in which a latent code `Z` is sampled from a
prior `p(Z)`, then used to generate an observation `X` by way of a decoder
`p(X|Z)`. The full reconstruction follows
```none
   X ~ p(X)              # A random image from some dataset.
   Z ~ q(Z | X)          # A random encoding of the original image ("encoder").
Xhat ~ p(Xhat | Z)       # A random reconstruction of the original image
                         #   ("decoder").
```
To fit the VAE, we assume an approximate representation of the posterior in the
form of an encoder `q(Z|X)`. We minimize the KL divergence between `q(Z|X)` and
the true posterior `p(Z|X)`: this is equivalent to maximizing the evidence lower
bound (ELBO),
```none
-log p(x)
= -log int dz p(x|z) p(z)
= -log int dz q(z|x) p(x|z) p(z) / q(z|x)
<= int dz q(z|x) (-log[ p(x|z) p(z) / q(z|x) ])   # Jensen's Inequality
=: KL[q(Z|x) || p(x|Z)p(Z)]
= -E_{Z~q(Z|x)}[log p(x|Z)] + KL[q(Z|x) || p(Z)]
```
-or-
```none
-log p(x)
= KL[q(Z|x) || p(x|Z)p(Z)] - KL[q(Z|x) || p(Z|x)]
<= KL[q(Z|x) || p(x|Z)p(Z)                        # Positivity of KL
= -E_{Z~q(Z|x)}[log p(x|Z)] + KL[q(Z|x) || p(Z)]
```
The `-E_{Z~q(Z|x)}[log p(x|Z)]` term is an expected reconstruction loss and
`KL[q(Z|x) || p(Z)]` is a kind of distributional regularizer. See
[Kingma and Welling (2014)][1] for more details.
This script supports both a (learned) mixture of Gaussians prior as well as a
fixed standard normal prior. You can enable the fixed standard normal prior by
setting `mixture_components` to 1. Note that fixing the parameters of the prior
(as opposed to fitting them with the rest of the model) incurs no loss in
generality when using only a single Gaussian. The reasoning for this is
two-fold:
  * On the generative side, the parameters from the prior can simply be absorbed
    into the first linear layer of the generative net. If `z ~ N(mu, Sigma)` and
    the first layer of the generative net is given by `x = Wz + b`, this can be
    rewritten,
      s ~ N(0, I)
      x = Wz + b
        = W (As + mu) + b
        = (WA) s + (W mu + b)
    where Sigma has been decomposed into A A^T = Sigma. In other words, the log
    likelihood of the model (E_{Z~q(Z|x)}[log p(x|Z)]) is independent of whether
    or not we learn mu and Sigma.
  * On the inference side, we can adjust any posterior approximation
    q(z | x) ~ N(mu[q], Sigma[q]), with
    new_mu[p] := 0
    new_Sigma[p] := eye(d)
    new_mu[q] := inv(chol(Sigma[p])) @ (mu[p] - mu[q])
    new_Sigma[q] := inv(Sigma[q]) @ Sigma[p]
    A bit of algebra on the KL divergence term `KL[q(Z|x) || p(Z)]` reveals that
    it is also invariant to the prior parameters as long as Sigma[p] and
    Sigma[q] are invertible.
This script also supports using the analytic KL (KL[q(Z|x) || p(Z)]) with the
`analytic_kl` flag. Using the analytic KL is only supported when
`mixture_components` is set to 1 since otherwise no analytic form is known.
Here we also compute tighter bounds, the IWAE [Burda et. al. (2015)][2].
These as well as image summaries can be seen in Tensorboard. For help using
Tensorboard see
https://www.tensorflow.org/guide/summaries_and_tensorboard
which can be run with
  `python -m tensorboard.main --logdir=MODEL_DIR`
#### References
[1]: Diederik Kingma and Max Welling. Auto-Encoding Variational Bayes. In
     _International Conference on Learning Representations_, 2014.
     https://arxiv.org/abs/1312.6114
[2]: Yuri Burda, Roger Grosse, Ruslan Salakhutdinov. Importance Weighted
     Autoencoders. In _International Conference on Learning Representations_,
     2015.
     https://arxiv.org/abs/1509.00519
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os

# Dependency imports
from absl import flags
import numpy as np
from six.moves import urllib
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.python.ops import variable_scope
from tensorflow.contrib.framework.python.ops import variables as variable_lib

tfd = tfp.distributions

IMAGE_SHAPE = [28, 28, 1]

flags.DEFINE_float(
    "learning_rate", default=0.001, help="Initial learning rate.")
flags.DEFINE_integer(
    "max_steps", default=5001, help="Number of training steps to run.")
flags.DEFINE_integer(
    "latent_size",
    default=16,
    help="Number of dimensions in the latent code (z).")
flags.DEFINE_integer("base_depth", default=32, help="Base depth for layers.")
flags.DEFINE_string(
    "activation",
    default="leaky_relu",
    help="Activation function for all hidden layers.")
flags.DEFINE_integer(
    "batch_size",
    default=32,
    help="Batch size.")
flags.DEFINE_integer(
    "n_samples", default=16, help="Number of samples to use in encoding.")
flags.DEFINE_integer(
    "mixture_components",
    default=100,
    help="Number of mixture components to use in the prior. Each component is "
         "a diagonal normal distribution. The parameters of the components are "
         "intialized randomly, and then learned along with the rest of the "
         "parameters. If `analytic_kl` is True, `mixture_components` must be "
         "set to `1`.")
flags.DEFINE_bool(
    "analytic_kl",
    default=False,
    help="Whether or not to use the analytic version of the KL. When set to "
         "False the E_{Z~q(Z|X)}[log p(Z)p(X|Z) - log q(Z|X)] form of the ELBO "
         "will be used. Otherwise the -KL(q(Z|X) || p(Z)) + "
         "E_{Z~q(Z|X)}[log p(X|Z)] form will be used. If analytic_kl is True, "
         "then you must also specify `mixture_components=1`.")
flags.DEFINE_string(
    "data_dir",
    default=os.path.join(os.getenv("TEST_TMPDIR", "/tmp"), "vae/data"),
    help="Directory where data is stored (if using real data).")
flags.DEFINE_string(
    "model_dir",
    default=os.path.join(os.getenv("TEST_TMPDIR", "/tmp"), "vae/"),
    help="Directory to put the model's fit.")
flags.DEFINE_integer(
    "viz_steps", default=500, help="Frequency at which to save visualizations.")
flags.DEFINE_bool(
    "fake_data",
    default=False,
    help="If true, uses fake data instead of MNIST.")
flags.DEFINE_bool(
    "delete_existing",
    default=False,
    help="If true, deletes existing `model_dir` directory.")
flags.DEFINE_integer(
    "glow_num_levels", 
    default=2,
    help="Number of Glow levels in the flow.")
flags.DEFINE_integer(
    "glow_level_depth", 
    default=2,
    help="Number of flow steps in each Glow level.")

FLAGS = flags.FLAGS


def _softplus_inverse(x):
  """Helper which computes the function inverse of `tf.nn.softplus`."""
  return tf.math.log(tf.math.expm1(x))


def make_encoder(activation, latent_size, base_depth, scale_min=0.01, scale_range=0.1):
  """Creates the encoder function.
  Args:
    activation: Activation function in hidden layers.
    latent_size: The dimensionality of the encoding.
    base_depth: The lowest depth for a layer.
  Returns:
    encoder: A `callable` mapping a `Tensor` of images to a
      `tfd.Distribution` instance over encodings.
  """
  conv = functools.partial(
      tf.keras.layers.Conv2D, padding="SAME", activation=activation)

  encoder_net = tf.keras.Sequential([
      conv(base_depth, 5, 1),
      conv(base_depth, 5, 2),
      conv(2 * base_depth, 5, 1),
      conv(2 * base_depth, 5, 2),
      conv(4 * latent_size, 7, padding="VALID"),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(2 * latent_size, activation=None),
  ], name="EncoderSequential")

  def encoder(images):
    images = 2 * tf.cast(images, dtype=tf.float32) - 1
    net = encoder_net(images)
    return tfd.MultivariateNormalDiag(
        loc=net[..., :latent_size],
        scale_diag=scale_min + scale_range*tf.nn.softplus(net[..., latent_size:] +
                                  _softplus_inverse(1.0)),
        name="code")

  return encoder


def make_decoder(activation, latent_size, output_shape, base_depth, scale_min=0.01, scale_range=0.1):
  """Creates the decoder function.
  Args:
    activation: Activation function in hidden layers.
    latent_size: Dimensionality of the encoding.
    output_shape: The output image shape.
    base_depth: Smallest depth for a layer.
  Returns:
    decoder: A `callable` mapping a `Tensor` of encodings to a
      `tfd.Distribution` instance over images.
  """
  deconv = functools.partial(
      tf.keras.layers.Conv2DTranspose, padding="SAME", activation=activation)
  conv = functools.partial(
      tf.keras.layers.Conv2D, padding="SAME", activation=activation)

  output_depth = output_shape[-1]

  decoder_net = tf.keras.Sequential([
      deconv(2 * base_depth, 7, padding="VALID"),
      deconv(2 * base_depth, 5),
      deconv(2 * base_depth, 5, 2),
      deconv(base_depth, 5),
      deconv(base_depth, 5, 2),
      deconv(base_depth, 5),
      conv(2*output_depth, 5, activation=None),
  ], name="DecoderSequential")

  def decoder(codes, label):
    original_shape = tf.shape(input=codes)
    # Collapse the sample and batch dimension and convert to rank-4 tensor for
    # use with a convolutional decoder network.
    codes = tf.reshape(codes, (-1, 1, 1, latent_size))
    
    #logits = decoder_net(codes)
    #logits = tf.reshape(
    #    logits, shape=tf.concat([original_shape[:-1], output_shape], axis=0))
    #return tfd.Independent(tfd.Bernoulli(logits=logits),
    #                       reinterpreted_batch_ndims=len(output_shape),
    #                       name="image")

    logits = decoder_net(codes)
    logits = tf.reshape(logits, shape=tf.concat([original_shape[:-1], output_shape[:-1], [output_depth*2]], axis=0))

    loc=logits[...,:output_depth]
    scale_raw=tf.compat.v1.get_variable(name="scale_input", dtype=tf.float32, shape=[])
    scale=tf.scalar_mul(scale_min + scale_range*tf.nn.softplus(scale_raw), tf.ones_like(loc))
    #scale=scale_min+scale_range*tf.nn.sigmoid(logits[...,output_depth:])

    tf.compat.v1.summary.scalar(label + "loc", tf.reduce_mean(input_tensor=loc))
    tf.compat.v1.summary.scalar(label + "loc_max", tf.reduce_max(input_tensor=loc))
    tf.compat.v1.summary.scalar(label + "loc_min", tf.reduce_min(input_tensor=loc))
    tf.compat.v1.summary.scalar(label + "scale", tf.reduce_mean(input_tensor=scale))
    tf.compat.v1.summary.scalar(label + "scale_max", tf.reduce_max(input_tensor=scale))
    tf.compat.v1.summary.scalar(label + "scale_min", tf.reduce_min(input_tensor=scale))

    #return tfd.Independent(tfd.Logistic(loc=loc,scale=scale),reinterpreted_batch_ndims=len(output_shape),name="image")
    return tfd.Independent(tfd.Normal(loc=loc,scale=scale),reinterpreted_batch_ndims=len(output_shape),name="image")

  return decoder


def make_mixture_prior(latent_size, mixture_components):
  """Creates the mixture of Gaussians prior distribution.
  Args:
    latent_size: The dimensionality of the latent representation.
    mixture_components: Number of elements of the mixture.
  Returns:
    random_prior: A `tfd.Distribution` instance representing the distribution
      over encodings in the absence of any evidence.
  """
  if mixture_components == 1:
    # See the module docstring for why we don't learn the parameters here.
    return tfd.MultivariateNormalDiag(
        loc=tf.zeros([latent_size]),
        scale_identity_multiplier=1.0)

  loc = tf.compat.v1.get_variable(
      name="loc", shape=[mixture_components, latent_size])
  raw_scale_diag = tf.compat.v1.get_variable(
      name="raw_scale_diag", shape=[mixture_components, latent_size])
  mixture_logits = tf.compat.v1.get_variable(
      name="mixture_logits", shape=[mixture_components])

  return tfd.MixtureSameFamily(
      components_distribution=tfd.MultivariateNormalDiag(
          loc=loc,
          scale_diag=tf.nn.softplus(raw_scale_diag)),
      mixture_distribution=tfd.Categorical(logits=mixture_logits),
      name="prior")

def make_generator(activation, latent_size, output_shape, base_depth):
  """Creates the decoder function.
  Args:
    activation: Activation function in hidden layers.
    latent_size: Dimensionality of the encoding.
    output_shape: The output image shape.
    base_depth: Smallest depth for a layer.
  Returns:
    generator: A `callable` mapping a `Tensor` of encodings to an image.
  """
  deconv = functools.partial(
      tf.keras.layers.Conv2DTranspose, padding="SAME", activation=activation)
  conv = functools.partial(
      tf.keras.layers.Conv2D, padding="SAME", activation=activation)

  output_depth = output_shape[-1]

  generator_net = tf.keras.Sequential([
      deconv(2 * base_depth, 7, padding="VALID"),
      deconv(2 * base_depth, 5),
      deconv(2 * base_depth, 5, 2),
      deconv(base_depth, 5),
      deconv(base_depth, 5, 2),
      deconv(base_depth, 5),
      conv(output_depth, 5, activation=None),
  ], name="GeneratorSequential")

  def generator(codes):
    original_shape = tf.shape(input=codes)
    # Collapse the sample and batch dimension and convert to rank-4 tensor for
    # use with a convolutional decoder network.
    codes = tf.reshape(codes, (-1, 1, 1, latent_size))
   
    output = generator_net(codes)
    output = tf.reshape(output, shape=tf.concat([original_shape[:-1], output_shape[:-1], [output_depth]], axis=0))

    return output

  return generator

def make_density(num_levels, level_depth):
  def density():
      density_base_distribution = tfd.MultivariateNormalDiag(
          loc=tf.zeros(np.prod(features.shape[-3:])),
          scale_diag=tf.ones(np.prod(features.shape[-3:])))

      density_flow = GlowFlow(
          num_levels=params['glow_num_levels'],
          level_depth=params['glow_level_depth'])

      density_distribution = tfd.TransformedDistribution(
          distribution=
            tfd.TransformedDistribution(
                distribution=density_base_distribution,
                bijector=tfp.bijectors.Reshape(event_shape_out=features.shape[-3:])
            ),
          bijector=density_flow,
          name="transformed_glow_flow")

      def density_log_probs(features):
          density_z = density_flow.inverse(features)
          density_prior_log_probs = density_base_distribution.log_prob(density_z)
          generaotr_prior_log_likelihood = -tf.reduce_mean(density_prior_log_probs)
          density_log_det_jacobians = density_flow.inverse_log_det_jacobian(features, event_ndims=3)
          density_log_probs_alt = density_log_det_jacobians + density_log_det_jacobians
          density_log_probs = density.log_prob(features)
          
          # Sanity check, remove when tested
          with tf.control_dependencies([tf.equal(density_log_probs, density_log_probs_alt)]):
            return 0 + density_log_probs

      return density_distribution, density_log_probs

  return density

def bits_per_dim(negative_log_likelihood, image_shape):
    image_size = tf.cast(tf.reduce_prod(image_shape),dtype=tf.float32)
    return ((negative_log_likelihood + tf.log(256.0) * image_size)
            / (image_size * tf.log(2.0)))

def pack_images(images, rows, cols):
  """Helper utility to make a field of images."""
  shape = tf.shape(input=images)
  width = shape[-3]
  height = shape[-2]
  depth = shape[-1]
  images = tf.reshape(images, (-1, width, height, depth))
  batch = tf.shape(input=images)[0]
  rows = tf.minimum(rows, batch)
  cols = tf.minimum(batch // rows, cols)
  images = images[:rows * cols]
  images = tf.reshape(images, (rows, cols, width, height, depth))
  images = tf.transpose(a=images, perm=[0, 2, 1, 3, 4])
  images = tf.reshape(images, [1, rows * width, cols * height, depth])
  return images


def image_tile_summary(name, tensor, rows=8, cols=8):
  tf.compat.v1.summary.image(
      name, pack_images(tensor, rows, cols), max_outputs=1)

def model_fn(features, labels, mode, params, config):
  """Builds the model function for use in an estimator.
  Arguments:
    features: The input features for the estimator.
    labels: The labels, unused here.
    mode: Signifies whether it is train or test or predict.
    params: Some hyperparameters as a dictionary.
    config: The RunConfig, unused here.
  Returns:
    EstimatorSpec: A tf.estimator.EstimatorSpec instance.
  """
  del labels, config

  if params["analytic_kl"] and params["mixture_components"] != 1:
    raise NotImplementedError(
        "Using `analytic_kl` is only supported when `mixture_components = 1` "
        "since there's no closed form otherwise.")

  image_tile_summary(
      "input", tf.cast(features, dtype=tf.float32), rows=1, cols=16)
 
  def build_vae(vae_label, deterministic_encoder=False):
      with variable_scope.variable_scope(vae_label + "encoder", reuse=tf.AUTO_REUSE) as encoder_scope:
          encoder = make_encoder(params["activation"],
                                 params["latent_size"],
                                 params["base_depth"])

      with variable_scope.variable_scope(vae_label + "decoder", reuse=tf.AUTO_REUSE) as decoder_scope:
          decoder = make_decoder(params["activation"],
                                 params["latent_size"],
                                 IMAGE_SHAPE,
                                 params["base_depth"])
          latent_prior = make_mixture_prior(params["latent_size"],
                                            params["mixture_components"])

      def elbo_fn(inputs, label):
          with variable_scope.variable_scope(encoder_scope, reuse=tf.AUTO_REUSE):
              approx_posterior = encoder(inputs)

          with variable_scope.variable_scope(decoder_scope, reuse=tf.AUTO_REUSE):
              if deterministic_encoder:
                  approx_posterior_sample = tf.expand_dims(approx_posterior.mean(), axis=0)
              else:
                  approx_posterior_sample = approx_posterior.sample(params["n_samples"])
              decoder_likelihood = decoder(approx_posterior_sample, label)

          viz_posterior_samples = min(3,approx_posterior_sample.shape[0])

          image_tile_summary(
              label + "recon/input",
              tf.cast(inputs, dtype=tf.float32),
              rows=1,
              cols=16)
          image_tile_summary(
              label + "recon/sample",
              tf.cast(decoder_likelihood.sample()[:viz_posterior_samples, :16], dtype=tf.float32),
              rows=viz_posterior_samples,
              cols=16)
          image_tile_summary(
              label + "recon/mean",
              decoder_likelihood.mean()[:viz_posterior_samples, :16],
              rows=viz_posterior_samples,
              cols=16)
          image_tile_summary(
              label + "recon/stddev",
              decoder_likelihood.stddev()[:viz_posterior_samples, :16],
              rows=viz_posterior_samples,
              cols=16)
          
          tf.compat.v1.summary.scalar(label + "encoder/stddev/min", tf.reduce_min(approx_posterior.stddev()))
          tf.compat.v1.summary.scalar(label + "decoder/stddev/min", tf.reduce_min(decoder_likelihood.stddev()))

          # `distortion` is just the negative log likelihood.
          distortion = -decoder_likelihood.log_prob(inputs)
          avg_distortion = tf.reduce_mean(input_tensor=distortion)
          tf.compat.v1.summary.scalar(label + "distortion", avg_distortion)

          tf.compat.v1.summary.scalar(label + "distortion_max", tf.reduce_max(input_tensor=distortion))
          tf.compat.v1.summary.scalar(label + "distortion_min", tf.reduce_min(input_tensor=distortion))

          if params["analytic_kl"]:
            rate = tfd.kl_divergence(approx_posterior, latent_prior)
          else:
            if deterministic_encoder:
                rate = (- latent_prior.log_prob(approx_posterior_sample))
            else:
                rate = (approx_posterior.log_prob(approx_posterior_sample)
                        - latent_prior.log_prob(approx_posterior_sample))
          avg_rate = tf.reduce_mean(input_tensor=rate)
          tf.compat.v1.summary.scalar(label + "rate", avg_rate)

          elbo_local = -(rate + distortion)

          elbo = tf.reduce_mean(input_tensor=elbo_local)
          tf.compat.v1.summary.scalar(label + "elbo", elbo)

          importance_weighted_elbo = tf.reduce_mean(
              input_tensor=tf.reduce_logsumexp(input_tensor=elbo_local, axis=0) -
              tf.math.log(tf.cast(params["n_samples"], dtype=tf.float32)))
          tf.compat.v1.summary.scalar(label + "elbo/importance_weighted",
                                      importance_weighted_elbo)
          eval_metric_ops = {
                  label + "elbo":
                      tf.compat.v1.metrics.mean(elbo),
                  label + "elbo/importance_weighted":
                      tf.compat.v1.metrics.mean(importance_weighted_elbo),
                  label + "rate":
                      tf.compat.v1.metrics.mean(avg_rate),
                  label + "distortion":
                      tf.compat.v1.metrics.mean(avg_distortion),
              }
          return elbo, importance_weighted_elbo, eval_metric_ops

      return decoder_scope, decoder, latent_prior, encoder_scope, encoder, elbo_fn

  def build_entropy(entropy_label):
      with variable_scope.variable_scope(entropy_label + "encoder", reuse=tf.AUTO_REUSE) as encoder_scope:
          encoder = make_encoder(params["activation"],
                                 params["latent_size"],
                                 params["base_depth"])

      def entropy_fn(decoder_input, decoder_output, label):
          with variable_scope.variable_scope(encoder_scope, reuse=tf.AUTO_REUSE):
              approx_posterior = encoder(decoder_output)

          reconstruction = approx_posterior.log_prob(decoder_input)
          avg_reconstruction = tf.reduce_mean(input_tensor=reconstruction)
          tf.compat.v1.summary.scalar(label + "reconstruction", avg_reconstruction)

          tf.compat.v1.summary.scalar(label + "reconstruction_max", tf.reduce_max(input_tensor=reconstruction))
          tf.compat.v1.summary.scalar(label + "reconstruction_min", tf.reduce_min(input_tensor=reconstruction))

          entropy = tf.reduce_mean(reconstruction)

          eval_metric_ops = {
                  label + "entropy":
                      tf.compat.v1.metrics.mean(entropy),
                  label + "reconstruction":
                      tf.compat.v1.metrics.mean(avg_reconstruction)
              }
          return entropy, eval_metric_ops

      return encoder_scope, encoder, entropy_fn

  decoder_scope, decoder, latent_prior, encoder_scope, encoder, elbo_fn = build_vae("vae_", deterministic_encoder=True)
  entropy_encoder_scope, entropy_encoder, entropy_fn = build_entropy("entropy_")

  # Decode samples from the prior for visualization.
  with variable_scope.variable_scope(decoder_scope, reuse=tf.AUTO_REUSE):
      decoder_random_image = decoder(latent_prior.sample(16), "sample/")

  image_tile_summary(
      "decoder/random/sample",
      tf.cast(decoder_random_image.sample(), dtype=tf.float32),
      rows=4,
      cols=4)
  image_tile_summary("decoder/random/mean", decoder_random_image.mean(), rows=4, cols=4)
  image_tile_summary("decoder/random/stddev", decoder_random_image.stddev(), rows=4, cols=4)

  # Generator
  with variable_scope.variable_scope("generator", reuse=tf.AUTO_REUSE) as generator_scope:
      generator_prior = make_mixture_prior(params["latent_size"],
                                        params["mixture_components"])
      generator = make_generator(params["activation"],
                             generator_prior.event_shape[-1],
                             IMAGE_SHAPE,
                             params["base_depth"])

      generator_input = generator_prior.sample(params["n_samples"])
      generator_output = generator(generator_input)

      generator_random_directions = tf.math.l2_normalize(tf.random_normal(tf.shape(generator_input)), axis=1)
      from tensorflow.contrib.nn.python.ops import fwd_gradients
      generator_direction_gradients = fwd_gradients.fwd_gradients([generator_output], [generator_input], [generator_random_directions])[0]

  # Generator samples for visualization.
  with variable_scope.variable_scope(generator_scope, reuse=tf.AUTO_REUSE):
      generator_random_image = generator(generator_prior.sample(16))
  image_tile_summary(
      "generator/random/sample",
      tf.cast(generator_random_image, dtype=tf.float32),
      rows=4,
      cols=4)

  # Build positive loss
  positive_elbo, positive_importance_weighted_elbo, positive_eval_metric_ops = elbo_fn(features, "positive/")
  positive_loss = -positive_elbo

  # Build negative loss
  negative_elbo, negative_importance_weighted_elbo, negative_eval_metric_ops = elbo_fn(generator_output, "negative/")
  negative_loss = -negative_elbo

  # Build entropy loss
  entropy, entropy_eval_metric_ops = entropy_fn(generator_input, generator_output, "generator/")
  entropy_loss = -entropy

  # Build generator loss
  generator_reg = 0
  #generator_reg += -tf.reduce_mean(np.prod(IMAGE_SHAPE)*tf.math.log(1e-3 + tf.norm(generator_direction_gradients, axis=1)))
  generator_reg = -entropy
  generator_loss = -negative_elbo
  generator_reg_loss = generator_loss + generator_reg
  tf.compat.v1.summary.scalar("generator/generator_reg", generator_reg)
  tf.compat.v1.summary.scalar("generator/generator_loss", generator_loss)
  tf.compat.v1.summary.scalar("generator/generator_reg_loss", generator_reg_loss)

  # Build global step
  global_step = tf.compat.v1.train.get_or_create_global_step()

  # Train encoder 
  encoder_learning_rate = tf.compat.v1.train.cosine_decay(params["learning_rate"], global_step, params["max_steps"])
  tf.compat.v1.summary.scalar("encoder/learning_rate", encoder_learning_rate)
  encoder_optimizer = tf.compat.v1.train.AdamOptimizer(encoder_learning_rate)
  encoder_train_op = encoder_optimizer.minimize(positive_loss, var_list=variable_lib.get_trainable_variables(encoder_scope))

  # Train decoder 
  decoder_learning_rate = tf.compat.v1.train.cosine_decay(params["learning_rate"], global_step, params["max_steps"])
  tf.compat.v1.summary.scalar("decoder/learning_rate", decoder_learning_rate)
  decoder_optimizer = tf.compat.v1.train.AdamOptimizer(decoder_learning_rate)
  decoder_train_op = encoder_optimizer.minimize(positive_loss, var_list=variable_lib.get_trainable_variables(decoder_scope))

  # Train entropy
  entropy_learning_rate = tf.compat.v1.train.cosine_decay(params["learning_rate"], global_step, params["max_steps"])
  tf.compat.v1.summary.scalar("entropy/learning_rate", entropy_learning_rate)
  entropy_optimizer = tf.compat.v1.train.AdamOptimizer(entropy_learning_rate)
  entropy_train_op = entropy_optimizer.minimize(entropy_loss, var_list=variable_lib.get_trainable_variables(entropy_encoder_scope))

  # Train generator
  generator_learning_rate = tf.compat.v1.train.cosine_decay(params["learning_rate"], global_step, params["max_steps"])
  tf.compat.v1.summary.scalar("generator/learning_rate", generator_learning_rate)
  generator_optimizer = tf.compat.v1.train.AdamOptimizer(decoder_learning_rate)
  generator_train_op = encoder_optimizer.minimize(generator_loss, var_list=variable_lib.get_trainable_variables(generator_scope))

  from tensorflow.contrib.gan import RunTrainOpsHook
  return tf.estimator.EstimatorSpec(
      mode=mode,
      loss=positive_loss,
      train_op=global_step.assign_add(1),
      training_hooks=[RunTrainOpsHook(encoder_train_op,1), RunTrainOpsHook(decoder_train_op,1), RunTrainOpsHook(entropy_train_op,1), RunTrainOpsHook(generator_train_op,1)],
      eval_metric_ops={**positive_eval_metric_ops, **negative_eval_metric_ops, **entropy_eval_metric_ops},
  )


ROOT_PATH = "http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/"
FILE_TEMPLATE = "binarized_mnist_{split}.amat"


def download(directory, filename):
  """Downloads a file."""
  filepath = os.path.join(directory, filename)
  if tf.io.gfile.exists(filepath):
    return filepath
  if not tf.io.gfile.exists(directory):
    tf.io.gfile.makedirs(directory)
  url = os.path.join(ROOT_PATH, filename)
  print("Downloading %s to %s" % (url, filepath))
  urllib.request.urlretrieve(url, filepath)
  return filepath


def static_mnist_dataset(directory, split_name):
    import tensorflow.examples.tutorials.mnist
    mnist_datasets = tensorflow.examples.tutorials.mnist.input_data.read_data_sets(directory)
    if split_name == "train":
        result =  mnist_datasets.train
    elif split_name == "valid":
        result = mnist_datasets.validation
    elif split_name == "test":
        result = mnist.test
    else:
        assert False
    result = tf.data.Dataset.from_tensor_slices((result.images, result.labels))
    return result.map(lambda image, label: (tf.cast(tf.reshape(image, [28,28,1]), dtype=tf.float32), tf.cast(label, tf.int32)))

def static_binary_mnist_dataset(directory, split_name):
  """Returns binary static MNIST tf.data.Dataset."""
  amat_file = download(directory, FILE_TEMPLATE.format(split=split_name))
  dataset = tf.data.TextLineDataset(amat_file)
  str_to_arr = lambda string: np.array([c == b"1" for c in string.split()])

  def _parser(s):
    booltensor = tf.compat.v1.py_func(str_to_arr, [s], tf.bool)
    reshaped = tf.reshape(booltensor, [28, 28, 1])
    return tf.cast(reshaped, dtype=tf.float32), tf.constant(0, tf.int32)

  return dataset.map(_parser)


def build_fake_input_fns(batch_size):
  """Builds fake MNIST-style data for unit testing."""
  random_sample = np.random.rand(batch_size, *IMAGE_SHAPE).astype("float32")

  def train_input_fn():
    dataset = tf.data.Dataset.from_tensor_slices(
        random_sample).map(lambda row: (row, 0)).batch(batch_size).repeat()
    return tf.compat.v1.data.make_one_shot_iterator(dataset).get_next()

  def eval_input_fn():
    dataset = tf.data.Dataset.from_tensor_slices(
        random_sample).map(lambda row: (row, 0)).batch(batch_size)
    return tf.compat.v1.data.make_one_shot_iterator(dataset).get_next()

  return train_input_fn, eval_input_fn


def build_input_fns(data_dir, batch_size):
  """Builds an Iterator switching between train and heldout data."""

  # Build an iterator over training batches.
  def train_input_fn():
    dataset = static_mnist_dataset(data_dir, "train")
    dataset = dataset.shuffle(50000).repeat().batch(batch_size)
    return tf.compat.v1.data.make_one_shot_iterator(dataset).get_next()

  # Build an iterator over the heldout set.
  def eval_input_fn():
    eval_dataset = static_mnist_dataset(data_dir, "valid")
    eval_dataset = eval_dataset.batch(batch_size)
    return tf.compat.v1.data.make_one_shot_iterator(eval_dataset).get_next()

  return train_input_fn, eval_input_fn


def main(argv):
  del argv  # unused

  tf.compat.v1.logging.set_verbosity(tf.logging.DEBUG)

  params = FLAGS.flag_values_dict()
  params["activation"] = getattr(tf.nn, params["activation"])
  if FLAGS.delete_existing and tf.io.gfile.exists(FLAGS.model_dir):
    tf.compat.v1.logging.warn("Deleting old log directory at {}".format(
        FLAGS.model_dir))
    tf.io.gfile.rmtree(FLAGS.model_dir)
  tf.io.gfile.makedirs(FLAGS.model_dir)

  if FLAGS.fake_data:
    train_input_fn, eval_input_fn = build_fake_input_fns(FLAGS.batch_size)
  else:
    train_input_fn, eval_input_fn = build_input_fns(FLAGS.data_dir,
                                                    FLAGS.batch_size)

  session_config = tf.ConfigProto()
  session_config.gpu_options.allow_growth=True
  estimator = tf.estimator.Estimator(
      model_fn,
      params=params,
      config=tf.estimator.RunConfig(
          session_config=session_config,
          model_dir=FLAGS.model_dir,
          save_checkpoints_steps=FLAGS.viz_steps,
          #save_summary_steps=1
      ),
  )

  for _ in range(FLAGS.max_steps // FLAGS.viz_steps):
    estimator.train(train_input_fn, steps=FLAGS.viz_steps)
    eval_results = estimator.evaluate(eval_input_fn)
    print("Evaluation_results:\n\t%s\n" % eval_results)


if __name__ == "__main__":
  tf.compat.v1.app.run()
