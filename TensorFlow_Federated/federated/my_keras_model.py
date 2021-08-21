"""
This contains the modified script of the _KerasModel class from the source file keras_utils.py.
That allows to take the client's metrics during the training and evaluation.

"""
import collections
from typing import List, Union

import tensorflow as tf
import tensorflow_federated as tff
from tensorflow_federated.python.learning import model as model_lib

Loss = Union[tf.keras.losses.Loss, List[tf.keras.losses.Loss]]

def get_local_metrics(variables):
  local_metrics = collections.OrderedDict()
  for metric in metrics_name:
    if metric == 'num_examples':
      local_metrics[metric] = variables[metric]
    else:
      local_metrics[metric] = variables[metric]/variables["num_examples"]
  return local_metrics

@tff.federated_computation
def aggregate_metrics_across_clients(metrics):
  global metrics_name
  output = collections.OrderedDict()

  for metric in metrics_name:
    if metric == 'num_examples':
      output[metric] = tff.federated_sum(getattr(metrics, metric))
      output['per_client/' + metric] = tff.federated_collect(getattr(metrics, metric))
    else:
      output[metric]= tff.federated_mean(getattr(metrics, metric), metrics.num_examples)
      output['per_client/'+ metric]= tff.federated_collect(getattr(metrics, metric))
  return output

class My_keras_model(model_lib.Model):
  """Internal wrapper class for tf.keras.Model objects."""

  def __init__(self, keras_model: tf.keras.Model, input_spec,
               loss_fns: List[tf.keras.losses.Loss], loss_weights: List[float],
               Metrics: List[tf.keras.metrics.Metric]):

    self._keras_model = keras_model

    self._input_spec = input_spec
    self._loss_fns = loss_fns
    self._loss_weights = loss_weights
    self._metrics = Metrics
    self._variables = collections.OrderedDict()
    
    global metrics_name
    metrics_name =  ['num_examples', 'loss'] + [str(metric.name) for metric in self._metrics]      # Modifier cette ligne pour retirer la variable metrics_name et ainsi trouver comment obtenir la liste des attribue de metrics dans la fonction "aggregate_metrics_across_clients"

    for metric in metrics_name:
      self._variables[metric] = tf.Variable(0.0, name=metric, trainable=False)

    # This is defined here so that it closes over the `loss_fn`.
    class _WeightedMeanLossMetric(tf.keras.metrics.Mean):
      """A `tf.keras.metrics.Metric` wrapper for the loss function."""

      def __init__(self, name='loss', dtype=tf.float32):
        super().__init__(name, dtype)
        self._loss_fns = loss_fns
        self._loss_weights = loss_weights

      def update_state(self, y_true, y_pred, sample_weight=None):
        if isinstance(y_pred, list):
          batch_size = tf.shape(y_pred[0])[0]
        else:
          batch_size = tf.shape(y_pred)[0]

        if len(self._loss_fns) == 1:
          batch_loss = self._loss_fns[0](y_true, y_pred)
        else:
          batch_loss = tf.zeros(())
          for i in range(len(self._loss_fns)):
            batch_loss += self._loss_weights[i] * self._loss_fns[i](y_true[i],
                                                                    y_pred[i])

        return super().update_state(batch_loss, batch_size)

    self._loss_metric = _WeightedMeanLossMetric()

  @property
  def trainable_variables(self):
    return self._keras_model.trainable_variables

  @property
  def non_trainable_variables(self):
    return self._keras_model.non_trainable_variables

  @property
  def local_variables(self):
    local_variables = []
    for metric in metrics_name:
      local_variables.extend(self._variables[metric])
    # for metric in self._variables._fields:
    #   local_variables.extend(getattr(self._variables,metric))
    return local_variables

  @property
  def input_spec(self):
    return self._input_spec

  def _forward_pass(self, variables, batch_input, training=True):
    if hasattr(batch_input, '_asdict'):
      batch_input = batch_input._asdict()
    if isinstance(batch_input, collections.abc.Mapping):
      inputs = batch_input.get('x')
    else:
      inputs = batch_input[0]
    if inputs is None:
      raise KeyError('Received a batch_input that is missing required key `x`. '
                     'Instead have keys {}'.format(list(batch_input.keys())))
    predictions = self._keras_model(inputs, training=training)

    if isinstance(batch_input, collections.abc.Mapping):
      y_true = batch_input.get('y')
    else:
      y_true = batch_input[1]
    if y_true is not None:
      if len(self._loss_fns) == 1:
        loss_fn = self._loss_fns[0]
        batch_loss = tf.add_n([loss_fn(y_true=y_true, y_pred=predictions)] +
                              self._keras_model.losses)

      else:
        batch_loss = tf.add_n([tf.zeros(())] + self._keras_model.losses)
        for i in range(len(self._loss_fns)):
          loss_fn = self._loss_fns[i]
          loss_wt = self._loss_weights[i]
          batch_loss += loss_wt * loss_fn(
              y_true=y_true[i], y_pred=predictions[i])
    else:
      batch_loss = None

    # TODO(b/145308951): Follow up here to pass through sample_weight in the
    # case that we have a model supporting masking.

    num_examples = tf.cast(tf.size(y_true), tf.float32)
    variables['num_examples'].assign_add(num_examples)
    self._loss_metric.update_state(y_true=y_true, y_pred=predictions)
    variables['loss'].assign_add(self._loss_metric.result()*num_examples)
    for metric in self._metrics:
      metric_value = metric.update_state(y_true=y_true, y_pred=predictions)
      variables[str(metric.name)].assign_add(metric.result()*num_examples)

    return model_lib.BatchOutput(
        loss=batch_loss,
        predictions=predictions,
        num_examples=tf.shape(tf.nest.flatten(inputs)[0])[0])

  @tf.function
  def forward_pass(self, batch_input, training=True):
    return self._forward_pass(self._variables, batch_input, training=training)

  @tf.function
  def report_local_outputs(self):
    return get_local_metrics(self._variables)

  @property
  def federated_output_computation(self):
    return aggregate_metrics_across_clients

  @classmethod
  def make_batch(cls, x, y):
    return cls.Batch(x=x, y=y)