

from object_detection.core import region_similarity_calculator
from object_detection.protos import region_similarity_calculator_pb2


def build(region_similarity_calculator_config):
  

  if not isinstance(
      region_similarity_calculator_config,
      region_similarity_calculator_pb2.RegionSimilarityCalculator):
    raise ValueError(
        'region_similarity_calculator_config not of type '
        'region_similarity_calculator_pb2.RegionsSimilarityCalculator')

  similarity_calculator = region_similarity_calculator_config.WhichOneof(
      'region_similarity')
  if similarity_calculator == 'iou_similarity':
    return region_similarity_calculator.IouSimilarity()
  if similarity_calculator == 'ioa_similarity':
    return region_similarity_calculator.IoaSimilarity()
  if similarity_calculator == 'neg_sq_dist_similarity':
    return region_similarity_calculator.NegSqDistSimilarity()
  if similarity_calculator == 'thresholded_iou_similarity':
    return region_similarity_calculator.ThresholdedIouSimilarity(
        region_similarity_calculator_config.thresholded_iou_similarity
        .iou_threshold)

  raise ValueError('Unknown region similarity calculator.')
