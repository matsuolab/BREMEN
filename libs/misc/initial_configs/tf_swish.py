# The content of this file is ported from a new version of tensorflow.python.ops.nn_impl
# The version of tensorflow on the CSLab cluster does not have the swish implementation
# so it's ported over here.

from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops


def _swish_shape(op):
  """Shape helper function for swish and _swish_grad function below."""
  return [op.inputs[0].shape]


@function.Defun(shape_func=_swish_shape, func_name="swish_grad", noinline=True)
def _swish_grad(features, grad):
  """Gradient of Swish function defined below."""
  sigmoid_features = math_ops.sigmoid(features)
  activation_grad = (
      sigmoid_features * (1.0 + features * (1.0 - sigmoid_features)))
  return grad * activation_grad


@function.Defun(
    grad_func=_swish_grad,
    shape_func=_swish_shape,
    func_name="swish",
    noinline=True)
def swish(features):
  # pylint: disable=g-doc-args
  """Computes the Swish activation function: `x * sigmoid(x)`.

  Source: "Searching for Activation Functions" (Ramachandran et al. 2017)
  https://arxiv.org/abs/1710.05941

  Args:
    features: A `Tensor` representing preactivation values.
    name: A name for the operation (optional).

  Returns:
    The activation value.
  """
  # pylint: enable=g-doc-args
  features = ops.convert_to_tensor(features, name="features")
  return features * math_ops.sigmoid(features)
