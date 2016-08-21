import os
import sys

import asciitree
import collections

from concurrent.futures import ThreadPoolExecutor

PROJECT_ROOT = os.getcwd()
runfiles_path = os.path.join(PROJECT_ROOT, 'models', 'syntaxnet', 'bazel-bin', 'syntaxnet', 'parser_eval.runfiles')
tensorflow_path = os.path.join(PROJECT_ROOT, 'models', 'syntaxnet', 'bazel-bin', 'syntaxnet', 'parser_eval.runfiles',
                               'external', 'tf')

sys.path.append(runfiles_path)
sys.path.append(tensorflow_path)

import os.path
import time

import tensorflow as tf

from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from syntaxnet import sentence_pb2
from syntaxnet import graph_builder
from syntaxnet import structured_graph_builder
from syntaxnet.ops import gen_parser_ops

# from io import TextIOWrapper, BytesIO
# from StringIO import StringIO
# from subprocess import Popen, PIPE, STDOUT

# CURRENT_PATH = os.getcwd()
# other_option = os.path.dirname(os.path.realpath(__file__))
# print("current_path=%s, other_option=%s", CURRENT_PATH, other_option)

# runfiles_path = os.path.join(CURRENT_PATH, 'models', 'syntaxnet', 'bazel-bin', 'syntaxnet', 'parser_eval.runfiles')
# sys.path.append(runfiles_path)

# TODO: which of these paths is actually needed?
# sys.path.append(CURRENT_PATH + '/models/syntaxnet/bazel-bin/syntaxnet/parser_eval.runfiles')
# sys.path.append(os.path.join(CURRENT_PATH, 'models/syntaxnet/bazel-bin/syntaxnet/parser_eval.runfiles/'))
# sys.path.append('/Users/plowman/projects/parseface/models/syntaxnet/bazel-bin/syntaxnet/parser_eval.runfiles/')
# sys.path.append(
#   '/Users/plowman/projects/parseface/models/syntaxnet/bazel-bin/syntaxnet/parser_eval.runfiles/external/six_archive')

# tensorflow_path = os.path.join(CURRENT_PATH, 'models', 'syntaxnet', 'bazel-bin', 'syntaxnet', 'parser_eval.runfiles',
#                                'external', 'tf')
#
# sys.path.append(tensorflow_path)
# sys.path.append(
#   '/Users/plowman/projects/parseface/models/syntaxnet/bazel-bin/syntaxnet/parser_eval.runfiles/external/tf')


project_root = os.path.dirname(os.path.realpath(__file__))
input_file_path = os.path.join(project_root, "input-file.txt")
output_file_path = os.path.join(project_root, "output-file.txt")
parser_path = os.path.join(project_root, 'models', 'syntaxnet', 'bazel-bin', 'syntaxnet', 'parser_eval')
mcparseface_path = os.path.join(project_root, 'models', 'syntaxnet', 'syntaxnet', 'models', 'parsey_mcparseface')
tagger_params_path = os.path.join(mcparseface_path, 'tagger-params')
parser_params_path = os.path.join(mcparseface_path, 'parser-params')
task_context_path = os.path.join(project_root, "custom_context.pbtxt")

"""
PARSER_EVAL=bazel-bin/syntaxnet/parser_eval
MODEL_DIR=syntaxnet/models/parsey_mcparseface
[[ "$1" == "--conll" ]] && INPUT_FORMAT=stdin-conll || INPUT_FORMAT=stdin

$PARSER_EVAL \
  --input=$INPUT_FORMAT \
  --output=stdout-conll \
  --hidden_layer_sizes=64 \
  --arg_prefix=brain_tagger \
  --graph_builder=structured \
  --task_context=$MODEL_DIR/context.pbtxt \
  --model_path=$MODEL_DIR/tagger-params \
  --slim_model \
  --batch_size=1024 \
  --alsologtostderr \
   | \
  $PARSER_EVAL \
  --input=stdin-conll \
  --output=stdout-conll \
  --hidden_layer_sizes=512,512 \
  --arg_prefix=brain_parser \
  --graph_builder=structured \
  --task_context=$MODEL_DIR/context.pbtxt \
  --model_path=$MODEL_DIR/parser-params \
  --slim_model \
  --batch_size=1024 \
  --alsologtostderr \
  | \
  bazel-bin/syntaxnet/conll2tree \
  --task_context=$MODEL_DIR/context.pbtxt \
  --alsologtostderr
"""


def parse_sentence(sentence):
  tagged_sentence = _perform_action_in_thread(action="brain_tagger", sentence=sentence)

  parsed_sentence = _perform_action_in_thread(action="brain_parser", sentence=tagged_sentence)

  sentence_dict = _get_sentence_dict_in_thread(parsed_sentence)

  return sentence_dict


def to_dict(sentence):
  """Builds a dictionary representing the parse tree of a sentence.

  Args:
    sentence: Sentence protocol buffer to represent.
  Returns:
    Dictionary mapping tokens to children.
  """
  token_str = ['%s %s %s' % (token.word, token.tag, token.label)
               for token in sentence.token]
  children = [[] for token in sentence.token]
  root = -1
  for i in range(0, len(sentence.token)):
    token = sentence.token[i]
    if token.head == -1:
      root = i
    else:
      children[token.head].append(i)

  def _get_dict(i):
    d = collections.OrderedDict()
    for c in children[i]:
      d[token_str[c]] = _get_dict(c)
    return d

  tree = collections.OrderedDict()
  tree[token_str[root]] = _get_dict(root)
  return tree


def _get_sentence_dict_in_thread(sentence_input):
  # The file doesn't get written unless the calling thread dies.
  # So this is a hack to be able to read from the file we are writing to.
  input_file = open(input_file_path, mode="w")
  input_file.write(sentence_input)
  input_file.close()

  executor = ThreadPoolExecutor(max_workers=1)
  future = executor.submit(_get_sentence_dict)
  return future.result()


def _get_sentence_dict():
  logging.set_verbosity(logging.INFO)
  with tf.Session() as sess:
    src = gen_parser_ops.document_source(batch_size=32,
                                         corpus_name='input-from-file-conll',
                                         task_context=task_context_path)
    sentence = sentence_pb2.Sentence()
    result_dict = None
    while True:
      documents, finished = sess.run(src)
      logging.info('Read %d documents', len(documents))
      for d in documents:
        sentence.ParseFromString(d)
        d = to_dict(sentence)
        result_dict = d

      if finished:
        break

  return result_dict


def _perform_action_in_thread(action=None, sentence=None):
  # The file doesn't get written unless the calling thread dies.
  # So this is a hack to be able to read from the file we are writing to.
  input_file = open(input_file_path, mode="w")
  input_file.write(sentence)
  input_file.close()

  executor = ThreadPoolExecutor(max_workers=1)
  future = executor.submit(_perform_action, action)
  future.result()

  output_file = open(output_file_path, mode="r")
  result = output_file.read()
  output_file.close()

  return result


def _perform_action(action=None):
  arg_prefix = action
  task_context = task_context_path

  if action == "brain_tagger":
    hidden_layer_sizes = [64]
    model_path = tagger_params_path
    output = 'output-to-file'
    input = 'input-from-file'
  elif action == "brain_parser":
    hidden_layer_sizes = [512, 512]
    model_path = parser_params_path
    output = 'output-to-file-conll'
    input = 'input-from-file-conll'
  else:
    raise Exception("Do not recognize action %s" % action)

  with tf.Session() as sess:
    feature_sizes, domain_sizes, embedding_dims, num_actions = sess.run(
      gen_parser_ops.feature_size(task_context=task_context,
                                  arg_prefix=arg_prefix))

  beam_size = 8
  max_steps = 1000
  batch_size = 1024
  slim_model = True

  parser = structured_graph_builder.StructuredGraphBuilder(
    num_actions,
    feature_sizes,
    domain_sizes,
    embedding_dims,
    hidden_layer_sizes,
    gate_gradients=True,
    arg_prefix=arg_prefix,
    beam_size=beam_size,
    max_steps=max_steps)

  parser.AddEvaluation(task_context,
                       batch_size,
                       corpus_name=input,
                       evaluation_max_steps=max_steps)

  with tf.Session() as sess:

    parser.AddSaver(slim_model)
    sess.run(parser.inits.values())
    parser.saver.restore(sess, model_path)

    sink_documents = tf.placeholder(tf.string)
    sink = gen_parser_ops.document_sink(sink_documents,
                                        task_context=task_context,
                                        corpus_name=output)
    t = time.time()
    num_epochs = None
    num_tokens = 0
    num_correct = 0
    num_documents = 0
    while True:
      tf_eval_epochs, tf_eval_metrics, tf_documents = sess.run([
        parser.evaluation['epochs'],
        parser.evaluation['eval_metrics'],
        parser.evaluation['documents'],
      ])

      if len(tf_documents):
        logging.info('Processed %d documents', len(tf_documents))
        num_documents += len(tf_documents)
        sess.run(sink, feed_dict={sink_documents: tf_documents})

      num_tokens += tf_eval_metrics[0]
      num_correct += tf_eval_metrics[1]
      if num_epochs is None:
        num_epochs = tf_eval_epochs
      elif num_epochs < tf_eval_epochs:
        break

    logging.info('Total processed documents: %d', num_documents)
    if num_tokens > 0:
      eval_metric = 100.0 * num_correct / num_tokens
      logging.info('num correct tokens: %d', num_correct)
      logging.info('total tokens: %d', num_tokens)
      logging.info('Seconds elapsed in evaluation: %.2f, '
                   'eval metric: %.2f%%', time.time() - t, eval_metric)


  # with open(output_file_path, mode="w") as output_file:
  #   output_file.write(".")

  # with open(output_file_path, mode="r") as output_file:
  #   print output_file.read()

"""
/usr/local/bin/python /Users/plowman/projects/parseface/models/syntaxnet/bazel-bin/syntaxnet/parser_eval.runfiles/syntaxnet/parser_eval.py
--input=stdin --output=stdout-conll --hidden_layer_sizes=64 --arg_prefix=brain_tagger
--graph_builder=structured --task_context=syntaxnet/models/parsey_mcparseface/context.pbtxt
--model_path=syntaxnet/models/parsey_mcparseface/tagger-params --slim_model --batch_size=1024
--alsologtostderr
"""


def pretty_print_dict(input_dict):
  tr = asciitree.LeftAligned()
  print tr(input_dict)


def check_install():
  pass

if __name__ == "__main__":
  parsed_sentence_dict = parse_sentence("Maybe there was once a human who looked like you, and somewhere along the "
                                        "line you killed him and took his place. And your superiors don't know.")
  print(parsed_sentence_dict)
  pretty_print_dict(parsed_sentence_dict)
