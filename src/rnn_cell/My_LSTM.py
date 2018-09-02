#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author:zhoukaiyin
@Time:2018/5/11
"""
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.layers import base as base_layer
import collections
from tensorflow.python.util.tf_export import tf_export
try:
  from tensorflow.python.ops.rnn_cell_impl import LayerRNNCell
except ImportError:
  from tensorflow.python.ops.rnn_cell_impl import _LayerRNNCell as LayerRNNCell
_LSTMStateTuple = collections.namedtuple("LSTMStateTuple", ("c", "h"))
@tf_export("nn.rnn_cell.LSTMStateTuple")
class LSTMStateTuple(_LSTMStateTuple):
  """Tuple used by LSTM Cells for `state_size`, `zero_state`, and output state.

  Stores two elements: `(c, h)`, in that order. Where `c` is the hidden state
  and `h` is the output.

  Only used when `state_is_tuple=True`.
  """

  __slots__ = ()

  @property
  def dtype(self):
    (c, h) = self
    if c.dtype != h.dtype:
      raise TypeError("Inconsistent internal state: %s vs %s" %
                      (str(c.dtype), str(h.dtype)))
    return c.dtype
class NLSTM(LayerRNNCell):
  def __init__(self,
               num_units,
               bias =0.0,
               hide_kernel_initializer=None,
               input_kernel_initializer=None,
               cell_kernel_intializer = None,
               state_is_tuple=True,
               cell_clip=True,
               activation=None,
               reuse=None,
               name=None):
    super(NLSTM, self).__init__(_reuse=reuse, name=name)

    # Inputs must be 2-dimensional.
    self.input_spec = base_layer.InputSpec(ndim=2)
    self._state_is_tuple=state_is_tuple
    self._num_units = num_units
    self._bias=bias
    self._cell_clip=cell_clip
    self._hide_initializer = hide_kernel_initializer
    self._cell_initializer = cell_kernel_intializer
    self._input_initializer = input_kernel_initializer

    self._activation = activation or nn_ops.relu
    self._state_size = (
        LSTMStateTuple(num_units, num_units)
        if state_is_tuple else 2 * num_units)
  @property
  def state_size(self):
      return self._state_size
  @property
  def output_size(self):
    return self._num_units
  def set_variable(self,inputs_shape,gate):
      if inputs_shape[1].value is None:
          raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                           % inputs_shape)
      input_depth = inputs_shape[1].value
      # input weights

      if self._input_initializer is None:
          self._input_initializer = init_ops.random_normal_initializer(mean=0.0,
                                                                       stddev=0.001)
      # input weights variable
      self._input_kernel = self.add_variable(
          gate+"input_kernel",
          shape=[input_depth, self._num_units],
          initializer=self._input_initializer)
      # hide weight
      if self._hide_initializer is None:
          self._recurrent_initializer = init_ops.random_normal_initializer(mean=0.0,
                                                                           stddev=0.001)
      self._hide_kernel = self.add_variable(
          gate+"hide_kernel",
          shape=[self._num_units, self._num_units],
          initializer=self._recurrent_initializer)
      # cell weight
      if self._cell_initializer is None:
          self._cell_kernel_initializer = init_ops.orthogonal_initializer()
      self._cell_initializer = self.add_variable(
          gate+"cell_kernel",
          shape=[self._num_units, self._num_units],
          initializer=self._cell_kernel_initializer
      )

      if self._bias == 1.0:
          self._bias = self.add_variable(
              gate+"bias",
              shape=[self._num_units],
              initializer=init_ops.ones_initializer(dtype=self.dtype))
      else:
          self._bias = self.add_variable(
              gate+"bias",
              shape=[self._num_units],
              initializer=init_ops.zeros_initializer(dtype=self.dtype))
      return self._input_kernel,self._hide_kernel,self._cell_initializer,self._bias

  def build(self, inputs_shape):
    self.igate = self.set_variable(inputs_shape,"i")
    self.fgate = self.set_variable(inputs_shape,"f")
    self.cgate = self.set_variable(inputs_shape,"c")
    self.ogate = self.set_variable(inputs_shape,"o")
    self.built = True

  def call(self, inputs, state):
      """
      :param inputs:
      :param state:
      :return:
      """
      if self._state_is_tuple:
          (c_prev, h_prev) = state
      else:
          c_prev = array_ops.slice(state, [0, 0], [-1, self._num_units])
          h_prev = array_ops.slice(state, [0, self._num_units], [-1, self._num_units])
      i_need = math_ops.matmul(inputs, self.igate[0])+math_ops.matmul(h_prev, self.igate[1])+math_ops.matmul(c_prev, self.igate[2])+self.igate[3]
      i = math_ops.sigmoid(i_need)
      f_need = math_ops.matmul(inputs, self.fgate[0])+math_ops.matmul(h_prev, self.fgate[1])+math_ops.matmul(c_prev, self.igate[2])+self.fgate[3]
      f = math_ops.sigmoid(f_need)
      c_need = math_ops.matmul(inputs, self.cgate[0])+math_ops.matmul(h_prev, self.cgate[1])+self.cgate[3]
      c = math_ops.tanh(c_need)
      current_c = f*c_prev+i*c
      #use clip
      if self._cell_clip is not None:
          current_c = clip_ops.clip_by_value(current_c, -self._cell_clip, self._cell_clip)
      o_need = math_ops.matmul(inputs, self.ogate[0])+math_ops.matmul(h_prev, self.ogate[1])+math_ops.matmul(current_c, self.igate[2])+self.ogate[3]
      o=math_ops.sigmoid(o_need)
      output = o*math_ops.tanh(current_c)
      new_state =(LSTMStateTuple(current_c,output) if self._state_is_tuple else array_ops.concat([current_c,output],1))
      return output,new_state
#定义解码层
class TLSTM(LayerRNNCell):
  def __init__(self,
               num_units,
               bias =0.0,
               hide_kernel_initializer=None,
               input_kernel_initializer=None,
               cell_kernel_intializer = None,
               state_is_tuple=True,
               ti_initializer=None,
               cell_clip=True,
               activation=None,
               reuse=None,
               name=None):
    super(TLSTM, self).__init__(_reuse=reuse, name=name)

    # Inputs must be 2-dimensional.
    self.input_spec = base_layer.InputSpec(ndim=2)
    self._state_is_tuple=state_is_tuple
    self._num_units = num_units
    self._bias=bias
    self._cell_clip=cell_clip
    self._hide_initializer = hide_kernel_initializer
    self._cell_initializer = cell_kernel_intializer
    self._input_initializer = input_kernel_initializer
    self._ti_initializer = ti_initializer
    self._activation = activation or nn_ops.relu
    self._state_size = (
        LSTMStateTuple(num_units, num_units)
        if state_is_tuple else 2 * num_units)
  @property
  def state_size(self):
      return self._state_size
  @property
  def output_size(self):
    return self._num_units
  def set_variable(self,inputs_shape,gate):
      if inputs_shape[1].value is None:
          raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                           % inputs_shape)
      input_depth = inputs_shape[1].value
      # input weights

      if self._input_initializer is None:
          self._input_initializer = init_ops.random_normal_initializer(mean=0.0,
                                                                       stddev=0.001)
      # input weights variable
      self._input_kernel = self.add_variable(
          gate+"_input_kernel",
          shape=[input_depth, self._num_units],
          initializer=self._input_initializer)
      # hide weight
      if self._hide_initializer is None:
          self._recurrent_initializer = init_ops.random_normal_initializer(mean=0.0,
                                                                       stddev=0.001)
      self._hide_kernel = self.add_variable(
          gate+"_hide_kernel",
          shape=[self._num_units, self._num_units],
          initializer=self._recurrent_initializer)
      # cell weight
      if self._cell_initializer is None:
          self._cell_kernel_initializer = init_ops.random_normal_initializer(mean=0.0,
                                                                       stddev=0.001)
      self._cell_initializer = self.add_variable(
          gate+"_cell_kernel",
          shape=[self._num_units, self._num_units],
          initializer=self._cell_kernel_initializer
      )

      if self._bias == 1.0:
          self._bias = self.add_variable(
              gate+"bias",
              shape=[self._num_units],
              initializer=init_ops.ones_initializer(dtype=self.dtype))
      else:
          self._bias = self.add_variable(
              gate+"bias",
              shape=[self._num_units],
              initializer=init_ops.zeros_initializer(dtype=self.dtype))

      return self._input_kernel,self._hide_kernel,self._cell_initializer,self._bias
  def build(self, inputs_shape):
    self.igate = self.set_variable(inputs_shape,"ti")
    self.fgate = self.set_variable(inputs_shape,"tf")
    self.cgate = self.set_variable(inputs_shape,"tc")
    self.ogate = self.set_variable(inputs_shape,"to")
    if self._ti_initializer is None:
        self._ti_kernel_initializer = init_ops.orthogonal_initializer()
    self._Ti = self.add_variable(
        "Ti",
        shape=[self._num_units, self._num_units],
        initializer=self._ti_kernel_initializer
    )
    self._tb = self.add_variable(
        "tb",
        shape=[self._num_units],
        initializer=init_ops.zeros_initializer(dtype=self.dtype))
    self.built = True

  def call(self, inputs, state):
      """
      :param inputs:
      :param state:
      :return:
      """
      if self._state_is_tuple:
          (c_prev, h_prev) = state
      else:
          c_prev = array_ops.slice(state, [0, 0], [-1, self._num_units])
          h_prev = array_ops.slice(state, [0, self._num_units], [-1, self._num_units])
      T = math_ops.matmul(h_prev,self._Ti)+self._tb

      i_need = math_ops.matmul(inputs, self.igate[0])+math_ops.matmul(h_prev, self.igate[1])+math_ops.matmul(T, self.igate[2])+self.igate[3]
      i = math_ops.sigmoid(i_need)
      f_need = math_ops.matmul(inputs, self.fgate[0])+math_ops.matmul(h_prev, self.fgate[1])+math_ops.matmul(c_prev, self.igate[2])+self.fgate[3]
      f = math_ops.sigmoid(f_need)
      c_need = math_ops.matmul(inputs, self.cgate[0])+math_ops.matmul(h_prev, self.cgate[1])+self.cgate[3]
      c = math_ops.tanh(c_need)
      current_c = f*c_prev+i*c
      #use clip
      if self._cell_clip is not None:
          current_c = clip_ops.clip_by_value(current_c, -self._cell_clip, self._cell_clip)
      o_need = math_ops.matmul(inputs, self.ogate[0])+math_ops.matmul(h_prev, self.ogate[1])+math_ops.matmul(current_c, self.igate[2])+self.ogate[3]
      o=math_ops.sigmoid(o_need)
      output = o*math_ops.tanh(current_c)
      new_state =(LSTMStateTuple(current_c,output) if self._state_is_tuple else array_ops.concat([current_c,output],1))
      return output,new_state