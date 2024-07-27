"""Full AlphaFold protein structure prediction script."""
import enum
import json
import os
import pathlib
import pickle
import random
import shutil
import sys
import time
from typing import Any, Dict, Union, Optional

from absl import app
from absl import flags
from absl import logging
from alphafold.common import protein
from alphafold.common import residue_constants
from alphafold.model import config
from alphafold.model import data
from alphafold.model import model
from alphafold.relax import relax
import jax.numpy as jnp
import numpy as np

logging.set_verbosity(logging.INFO)

@enum.unique
class ModelsToRelax(enum.Enum):
  ALL = 0
  BEST = 1
  NONE = 2

flags.DEFINE_list(
    'fasta_paths', None, 'Paths to FASTA files, each containing a prediction '
    'target that will be folded one after another. If a FASTA file contains '
    'multiple sequences, then it will be folded as a multimer. Paths should be '
    'separated by commas. All FASTA paths must have a unique basename as the '
    'basename is used to name the output directories for each prediction.')
flags.DEFINE_string('output_dir', None, 'Path to a directory that will '
                    'store the results.')
flags.DEFINE_string('parameter_path', None, 'Path to directory of supporting data.')
flags.DEFINE_list('feature_files', None, 'Paths to pre-computed feature.pkl files, '
                  'corresponding to the FASTA files.')
flags.DEFINE_list('model_names', None, 'Names of models to use.')
flags.DEFINE_boolean('benchmark', False, 'Run multiple JAX model evaluations '
                     'to obtain a timing that excludes the compilation time, '
                     'which should be more indicative of the time required for '
                     'inferencing many proteins.')
flags.DEFINE_integer('random_seed', None, 'The random seed for the data '
                     'pipeline. By default, this is randomly generated. Note '
                     'that even if this is set, Alphafold may still not be '
                     'deterministic, because processes like GPU inference are '
                     'nondeterministic.')
flags.DEFINE_integer('num_multimer_predictions_per_model', 5, 'How many '
                     'predictions (each with a different random seed) will be '
                     'generated per model. E.g. if this is 2 and there are 5 '
                     'models then there will be 10 predictions per input. '
                     'Note: this FLAG only applies if model_preset=multimer')
flags.DEFINE_enum_class('models_to_relax', ModelsToRelax.BEST, ModelsToRelax,
                        'The models to run the final relaxation step on. '
                        'If `all`, all models are relaxed, which may be time '
                        'consuming. If `best`, only the most confident model '
                        'is relaxed. If `none`, relaxation is not run. Turning '
                        'off relaxation might result in predictions with '
                        'distracting stereochemical violations but might help '
                        'in case you are having issues with the relaxation '
                        'stage.')
flags.DEFINE_boolean('use_gpu_relax', None, 'Whether to relax on GPU. '
                     'Relax on GPU can be much faster than CPU, so it is '
                     'recommended to enable if possible. GPUs must be available'
                     ' if this setting is enabled.')
flags.DEFINE_enum('model_preset', 'monomer',
                  ['monomer', 'monomer_casp14', 'monomer_ptm', 'multimer'],
                  'Choose preset model configuration - the monomer model, '
                  'the monomer model with extra ensembling, monomer model with '
                  'pTM head, or multimer model')

FLAGS = flags.FLAGS

MAX_TEMPLATE_HITS = 20
RELAX_MAX_ITERATIONS = 0
RELAX_ENERGY_TOLERANCE = 2.39
RELAX_STIFFNESS = 10.0
RELAX_EXCLUDE_RESIDUES = []
RELAX_MAX_OUTER_ITERATIONS = 3

def _jnp_to_np(output: Dict[str, Any]) -> Dict[str, Any]:
  """Recursively changes jax arrays to numpy arrays."""
  for k, v in output.items():
    if isinstance(v, dict):
      output[k] = _jnp_to_np(v)
    elif isinstance(v, jnp.ndarray):
      output[k] = np.array(v)
  return output

def predict_structure(
    fasta_path: str,
    fasta_name: str,
    output_dir_base: str,
    feature_file: str,
    model_runners: Dict[str, model.RunModel],
    amber_relaxer: relax.AmberRelaxation,
    benchmark: bool,
    random_seed: int,
    models_to_relax: ModelsToRelax):
  """Predicts structure using AlphaFold for the given sequence."""
  logging.info('Predicting %s', fasta_name)
  timings = {}
  output_dir = os.path.join(output_dir_base, fasta_name)
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  # Get features
  with open(feature_file, 'rb') as f:
    feature_dict = pickle.load(f)

  # Write out features as a pickled dictionary.
  features_output_path = os.path.join(output_dir, 'features.pkl')
  with open(features_output_path, 'wb') as f:
    pickle.dump(feature_dict, f, protocol=4)

  unrelaxed_pdbs = {}
  unrelaxed_proteins = {}
  relaxed_pdbs = {}
  relax_metrics = {}
  ranking_confidences = {}

  # Run the models.
  num_models = len(model_runners)
  for model_index, (model_name, model_runner) in enumerate(
      model_runners.items()):
    logging.info('Running model %s on %s', model_name, fasta_name)
    t_0 = time.time()
    model_random_seed = model_index + random_seed * num_models
    processed_feature_dict = model_runner.process_features(
        feature_dict, random_seed=model_random_seed)
    timings[f'process_features_{model_name}'] = time.time() - t_0

    t_0 = time.time()
    prediction_result = model_runner.predict(processed_feature_dict,
                                             random_seed=model_random_seed)
    t_diff = time.time() - t_0
    timings[f'predict_and_compile_{model_name}'] = t_diff
    logging.info(
        'Total JAX model %s on %s predict time (includes compilation time, see --benchmark): %.1fs',
        model_name, fasta_name, t_diff)

    if benchmark:
      t_0 = time.time()
      model_runner.predict(processed_feature_dict,
                           random_seed=model_random_seed)
      t_diff = time.time() - t_0
      timings[f'predict_benchmark_{model_name}'] = t_diff
      logging.info(
          'Total JAX model %s on %s predict time (excludes compilation time): %.1fs',
          model_name, fasta_name, t_diff)

    plddt = prediction_result['plddt']
    ranking_confidences[model_name] = prediction_result['ranking_confidence']

    # Remove jax dependency from results.
    np_prediction_result = _jnp_to_np(dict(prediction_result))

    # Save the model outputs.
    result_output_path = os.path.join(output_dir, f'result_{model_name}.pkl')
    with open(result_output_path, 'wb') as f:
      pickle.dump(np_prediction_result, f, protocol=4)

    # Add the predicted LDDT in the b-factor column.
    # Note that higher predicted LDDT value means higher model confidence.
    plddt_b_factors = np.repeat(
        plddt[:, None], residue_constants.atom_type_num, axis=-1)
    unrelaxed_protein = protein.from_prediction(
        features=processed_feature_dict,
        result=prediction_result,
        b_factors=plddt_b_factors,
        remove_leading_feature_dimension=not model_runner.multimer_mode)

    unrelaxed_proteins[model_name] = unrelaxed_protein
    unrelaxed_pdbs[model_name] = protein.to_pdb(unrelaxed_protein)
    unrelaxed_pdb_path = os.path.join(output_dir, f'unrelaxed_{model_name}.pdb')
    with open(unrelaxed_pdb_path, 'w') as f:
      f.write(unrelaxed_pdbs[model_name])

  # Rank by model confidence.
  ranked_order = [
      model_name for model_name, confidence in
      sorted(ranking_confidences.items(), key=lambda x: x[1], reverse=True)]

  # Relax predictions.
  if models_to_relax == ModelsToRelax.BEST:
    to_relax = [ranked_order[0]]
  elif models_to_relax == ModelsToRelax.ALL:
    to_relax = ranked_order
  elif models_to_relax == ModelsToRelax.NONE:
    to_relax = []

  for model_name in to_relax:
    t_0 = time.time()
    relaxed_pdb_str, _, violations = amber_relaxer.process(
        prot=unrelaxed_proteins[model_name])
    relax_metrics[model_name] = {
        'remaining_violations': violations,
        'remaining_violations_count': sum(violations)
    }
    timings[f'relax_{model_name}'] = time.time() - t_0

    relaxed_pdbs[model_name] = relaxed_pdb_str

    # Save the relaxed PDB.
    relaxed_output_path = os.path.join(
        output_dir, f'relaxed_{model_name}.pdb')
    with open(relaxed_output_path, 'w') as f:
      f.write(relaxed_pdb_str)

  # Write out relaxed PDBs in rank order.
  for idx, model_name in enumerate(ranked_order):
    ranked_output_path = os.path.join(output_dir, f'ranked_{idx}.pdb')
    with open(ranked_output_path, 'w') as f:
      if model_name in relaxed_pdbs:
        f.write(relaxed_pdbs[model_name])
      else:
        f.write(unrelaxed_pdbs[model_name])

  ranking_output_path = os.path.join(output_dir, 'ranking_debug.json')
  with open(ranking_output_path, 'w') as f:
    label = 'iptm+ptm' if 'iptm' in prediction_result else 'plddts'
    f.write(json.dumps(
        {label: ranking_confidences, 'order': ranked_order}, indent=4))

  logging.info('Final timings for %s: %s', fasta_name, timings)

  timings_output_path = os.path.join(output_dir, 'timings.json')
  with open(timings_output_path, 'w') as f:
    f.write(json.dumps(timings, indent=4))
  if models_to_relax != ModelsToRelax.NONE:
    relax_metrics_path = os.path.join(output_dir, 'relax_metrics.json')
    with open(relax_metrics_path, 'w') as f:
      f.write(json.dumps(relax_metrics, indent=4))

def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # Check for duplicate FASTA file names.
  fasta_names = [pathlib.Path(p).stem for p in FLAGS.fasta_paths]
  if len(fasta_names) != len(set(fasta_names)):
    raise ValueError('All FASTA paths must have a unique basename.')

  if len(FLAGS.fasta_paths) != len(FLAGS.feature_files):
    raise ValueError('The number of feature files must match the number of FASTA files.')

  run_multimer_system = 'multimer' in FLAGS.model_preset

  if FLAGS.model_preset == 'monomer_casp14':
    num_ensemble = 8
  else:
    num_ensemble = 1

  if run_multimer_system:
    num_predictions_per_model = FLAGS.num_multimer_predictions_per_model
  else:
    num_predictions_per_model = 1

  model_runners = {}
  model_names = ['model_1']  # Only include model_1
  for model_name in model_names:
    model_config = config.model_config(model_name)
    if run_multimer_system:
      model_config.model.num_ensemble_eval = num_ensemble
    else:
      model_config.data.eval.num_ensemble = num_ensemble
    model_params = data.get_model_haiku_params(
        model_name=model_name, parameter_path=FLAGS.parameter_path)
    model_runner = model.RunModel(model_config, model_params)
    for i in range(num_predictions_per_model):
      model_runners[f'{model_name}_pred_{i}'] = model_runner

  logging.info('Have %d models: %s', len(model_runners),
               list(model_runners.keys()))

  amber_relaxer = relax.AmberRelaxation(
      max_iterations=RELAX_MAX_ITERATIONS,
      tolerance=RELAX_ENERGY_TOLERANCE,
      stiffness=RELAX_STIFFNESS,
      exclude_residues=RELAX_EXCLUDE_RESIDUES,
      max_outer_iterations=RELAX_MAX_OUTER_ITERATIONS,
      use_gpu=FLAGS.use_gpu_relax)

  random_seed = FLAGS.random_seed
  if random_seed is None:
    random_seed = random.randrange(sys.maxsize // len(model_runners))
  logging.info('Using random seed %d for the data pipeline', random_seed)

  # Predict structure for each of the sequences.
  for i, (fasta_path, feature_file) in enumerate(zip(FLAGS.fasta_paths, FLAGS.feature_files)):
    fasta_name = fasta_names[i]
    predict_structure(
        fasta_path=fasta_path,
        fasta_name=fasta_name,
        output_dir_base=FLAGS.output_dir,
        feature_file=feature_file,
        model_runners=model_runners,
        amber_relaxer=amber_relaxer,
        benchmark=FLAGS.benchmark,
        random_seed=random_seed,
        models_to_relax=FLAGS.models_to_relax)
    logging.info('%s AlphaFold structure prediction COMPLETE', fasta_name)


if __name__ == '__main__':
    flags.mark_flags_as_required([
        'fasta_paths',
        'output_dir',
        'parameter_path',
        'feature_files',
        'use_gpu_relax',
    ])
    app.run(main)
