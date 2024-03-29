
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import functools

from tensorflow.python.framework import ops

_ARGSTACK_KEY = ("__arg_stack",)

_DECORATED_OPS = set()


def _get_arg_stack():
  stack = ops.get_collection(_ARGSTACK_KEY)
  if stack:
    return stack[0]
  else:
    stack = [{}]
    ops.add_to_collection(_ARGSTACK_KEY, stack)
    return stack


def _current_arg_scope():
  stack = _get_arg_stack()
  return stack[-1]


def _add_op(op):
  key_op = (op.__module__, op.__name__)
  if key_op not in _DECORATED_OPS:
    _DECORATED_OPS.add(key_op)


@contextlib.contextmanager
def arg_scope(list_ops_or_scope, **kwargs):
  """Stores the default arguments for the given set of list_ops.

  For usage, please see examples at top of the file.

  Args:
    list_ops_or_scope: List or tuple of operations to set argument scope for or
      a dictionary containg the current scope. When list_ops_or_scope is a dict,
      kwargs must be empty. When list_ops_or_scope is a list or tuple, then
      every op in it need to be decorated with @add_arg_scope to work.
    **kwargs: keyword=value that will define the defaults for each op in
              list_ops. All the ops need to accept the given set of arguments.

  Yields:
    the current_scope, which is a dictionary of {op: {arg: value}}
  Raises:
    TypeError: if list_ops is not a list or a tuple.
    ValueError: if any op in list_ops has not be decorated with @add_arg_scope.
  """
  if isinstance(list_ops_or_scope, dict):
    # Assumes that list_ops_or_scope is a scope that is being reused.
    if kwargs:
      raise ValueError("When attempting to re-use a scope by suppling a"
                       "dictionary, kwargs must be empty.")
    current_scope = list_ops_or_scope.copy()
    try:
      _get_arg_stack().append(current_scope)
      yield current_scope
    finally:
      _get_arg_stack().pop()
  else:
    # Assumes that list_ops_or_scope is a list/tuple of ops with kwargs.
    if not isinstance(list_ops_or_scope, (list, tuple)):
      raise TypeError("list_ops_or_scope must either be a list/tuple or reused"
                      "scope (i.e. dict)")
    try:
      current_scope = _current_arg_scope().copy()
      for op in list_ops_or_scope:
        key_op = (op.__module__, op.__name__)
        if not has_arg_scope(op):
          raise ValueError("%s is not decorated with @add_arg_scope", key_op)
        if key_op in current_scope:
          current_kwargs = current_scope[key_op].copy()
          current_kwargs.update(kwargs)
          current_scope[key_op] = current_kwargs
        else:
          current_scope[key_op] = kwargs.copy()
      _get_arg_stack().append(current_scope)
      yield current_scope
    finally:
      _get_arg_stack().pop()


def add_arg_scope(func):
  """Decorates a function with args so it can be used within an arg_scope.

  Args:
    func: function to decorate.

  Returns:
    A tuple with the decorated function func_with_args().
  """
  @functools.wraps(func)
  def func_with_args(*args, **kwargs):
    current_scope = _current_arg_scope()
    current_args = kwargs
    key_func = (func.__module__, func.__name__)
    if key_func in current_scope:
      current_args = current_scope[key_func].copy()
      current_args.update(kwargs)
    return func(*args, **current_args)
  _add_op(func)
  return func_with_args


def has_arg_scope(func):
  """Checks whether a func has been decorated with @add_arg_scope or not.

  Args:
    func: function to check.

  Returns:
    a boolean.
  """
  key_op = (func.__module__, func.__name__)
  return key_op in _DECORATED_OPS
