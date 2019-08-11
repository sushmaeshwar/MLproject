
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from inception.slim import scopes

# Collection containing all the variables created using slim.variables
MODEL_VARIABLES = '_model_variables_'

# Collection containing the slim.variables that are created with restore=True.
VARIABLES_TO_RESTORE = '_variables_to_restore_'


def add_variable(var, restore=True):
  
  collections = [MODEL_VARIABLES]
  if restore:
    collections.append(VARIABLES_TO_RESTORE)
  for collection in collections:
    if var not in tf.get_collection(collection):
      tf.add_to_collection(collection, var)


def get_variables(scope=None, suffix=None):
  
  candidates = tf.get_collection(MODEL_VARIABLES, scope)[:]
  if suffix is not None:
    candidates = [var for var in candidates if var.op.name.endswith(suffix)]
  return candidates


def get_variables_to_restore():
 
  return tf.get_collection(VARIABLES_TO_RESTORE)[:]


def get_variables_by_name(given_name, scope=None):
  
  return get_variables(scope=scope, suffix=given_name)


def get_unique_variable(name):
  
  candidates = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, name)
  if not candidates:
    raise ValueError('Couldnt find variable %s' % name)

  for candidate in candidates:
    if candidate.op.name == name:
      return candidate
  raise ValueError('Variable %s does not uniquely identify a variable', name)


class VariableDeviceChooser(object):
  

  def __init__(self,
               num_parameter_servers=0,
               ps_device='/job:ps',
               placement='CPU:0'):
    
    self._num_ps = num_parameter_servers
    self._ps_device = ps_device
    self._placement = placement if num_parameter_servers == 0 else 'CPU:0'
    self._next_task_id = 0

  def __call__(self, op):
    device_string = ''
    if self._num_ps > 0:
      task_id = self._next_task_id
      self._next_task_id = (self._next_task_id + 1) % self._num_ps
      device_string = '%s/task:%d' % (self._ps_device, task_id)
    device_string += '/%s' % self._placement
    return device_string


# TODO(sguada) Remove once get_variable is able to colocate op.devices.
def variable_device(device, name):
  """Fix the variable device to colocate its ops."""
  if callable(device):
    var_name = tf.get_variable_scope().name + '/' + name
    var_def = tf.NodeDef(name=var_name, op='Variable')
    device = device(var_def)
  if device is None:
    device = ''
  return device


@scopes.add_arg_scope
def global_step(device=''):
  
  global_step_ref = tf.get_collection(tf.GraphKeys.GLOBAL_STEP)
  if global_step_ref:
    return global_step_ref[0]
  else:
    collections = [
        VARIABLES_TO_RESTORE,
        tf.GraphKeys.GLOBAL_VARIABLES,
        tf.GraphKeys.GLOBAL_STEP,
    ]
    # Get the device for the variable.
    with tf.device(variable_device(device, 'global_step')):
      return tf.get_variable('global_step', shape=[], dtype=tf.int64,
                             initializer=tf.zeros_initializer(),
                             trainable=False, collections=collections)


@scopes.add_arg_scope
def variable(name, shape=None, dtype=tf.float32, initializer=None,
             regularizer=None, trainable=True, collections=None, device='',
             restore=True):
  
  collections = list(collections or [])

  # Make sure variables are added to tf.GraphKeys.GLOBAL_VARIABLES and MODEL_VARIABLES
  collections += [tf.GraphKeys.GLOBAL_VARIABLES, MODEL_VARIABLES]
  # Add to VARIABLES_TO_RESTORE if necessary
  if restore:
    collections.append(VARIABLES_TO_RESTORE)
  # Remove duplicates
  collections = set(collections)
  # Get the device for the variable.
  with tf.device(variable_device(device, name)):
    return tf.get_variable(name, shape=shape, dtype=dtype,
                           initializer=initializer, regularizer=regularizer,
                           trainable=trainable, collections=collections)
