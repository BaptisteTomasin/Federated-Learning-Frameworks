import collections
from typing import List, Optional, Sequence, Union

import tensorflow as tf
import tensorflow_federated as tff
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.api import intrinsics
from tensorflow_federated.python.core.api import placements
from tensorflow_federated.python.learning import model as model_lib
from tensorflow_federated.python.learning import model_utils


Loss = Union[tf.keras.losses.Loss, List[tf.keras.losses.Loss]]


def from_keras_model(
    keras_model: tf.keras.Model,
    loss: Loss,
    input_spec,
    loss_weights: Optional[List[float]] = None,
    metrics: Optional[List[tf.keras.metrics.Metric]] = None) -> model_lib.Model:
 
  # Validate `keras_model`
  py_typecheck.check_type(keras_model, tf.keras.Model)
  if keras_model._is_compiled:  # pylint: disable=protected-access
    raise ValueError('`keras_model` must not be compiled')

  # Validate and normalize `loss` and `loss_weights`
  if not isinstance(loss, list):
    py_typecheck.check_type(loss, tf.keras.losses.Loss)
    if loss_weights is not None:
      raise ValueError('`loss_weights` cannot be used if `loss` is not a list.')
    loss = [loss]
    loss_weights = [1.0]
  else:
    if len(loss) != len(keras_model.outputs):
      raise ValueError('If a loss list is provided, `keras_model` must have '
                       'equal number of outputs to the losses.\nloss: {}\nof '
                       'length: {}.\noutputs: {}\nof length: {}.'.format(
                           loss, len(loss), keras_model.outputs,
                           len(keras_model.outputs)))
    for loss_fn in loss:
      py_typecheck.check_type(loss_fn, tf.keras.losses.Loss)

    if loss_weights is None:
      loss_weights = [1.0] * len(loss)
    else:
      if len(loss) != len(loss_weights):
        raise ValueError(
            '`keras_model` must have equal number of losses and loss_weights.'
            '\nloss: {}\nof length: {}.'
            '\nloss_weights: {}\nof length: {}.'.format(loss, len(loss),
                                                        loss_weights,
                                                        len(loss_weights)))
      for loss_weight in loss_weights:
        py_typecheck.check_type(loss_weight, float)

  if len(input_spec) != 2:
    raise ValueError('The top-level structure in `input_spec` must contain '
                     'exactly two top-level elements, as it must specify type '
                     'information for both inputs to and predictions from the '
                     'model. You passed input spec {}.'.format(input_spec))
  if not isinstance(input_spec, computation_types.Type):
    for input_spec_member in tf.nest.flatten(input_spec):
      py_typecheck.check_type(input_spec_member, tf.TensorSpec)
  else:
    for type_elem in input_spec:
      py_typecheck.check_type(type_elem, computation_types.TensorType)
  if isinstance(input_spec, collections.abc.Mapping):
    if 'x' not in input_spec:
      raise ValueError(
          'The `input_spec` is a collections.abc.Mapping (e.g., a dict), so it '
          'must contain an entry with key `\'x\'`, representing the input(s) '
          'to the Keras model.')
    if 'y' not in input_spec:
      raise ValueError(
          'The `input_spec` is a collections.abc.Mapping (e.g., a dict), so it '
          'must contain an entry with key `\'y\'`, representing the label(s) '
          'to be used in the Keras loss(es).')

  if metrics is None:
    metrics = []
  else:
    py_typecheck.check_type(metrics, list)
    for metric in metrics:
      py_typecheck.check_type(metric, tf.keras.metrics.Metric)

  return model_utils.enhance(
      _KerasModel(
          keras_model,
          input_spec=input_spec,
          loss_fns=loss,
          loss_weights=loss_weights,
          Metrics=metrics))

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

class _KerasModel(model_lib.Model):
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