from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os.path
import sys
import xml.etree.ElementTree as ET
from six.moves import xrange  # pylint: disable=redefined-builtin


class BoundingBox(object):
  pass


def GetItem(name, root, index=0):
  count = 0
  for item in root.iter(name):
    if count == index:
      return item.text
    count += 1
  # Failed to find "index" occurrence of item.
  return -1


def GetInt(name, root, index=0):
  return int(GetItem(name, root, index))


def FindNumberBoundingBoxes(root):
  index = 0
  while True:
    if GetInt('xmin', root, index) == -1:
      break
    index += 1
  return index


def ProcessXMLAnnotation(xml_file):
  """Process a single XML file containing a bounding box."""
  # pylint: disable=broad-except
  try:
    tree = ET.parse(xml_file)
  except Exception:
    print('Failed to parse: ' + xml_file, file=sys.stderr)
    return None
  # pylint: enable=broad-except
  root = tree.getroot()

  num_boxes = FindNumberBoundingBoxes(root)
  boxes = []

  for index in xrange(num_boxes):
    box = BoundingBox()
    # Grab the 'index' annotation.
    box.xmin = GetInt('xmin', root, index)
    box.ymin = GetInt('ymin', root, index)
    box.xmax = GetInt('xmax', root, index)
    box.ymax = GetInt('ymax', root, index)

    box.width = GetInt('width', root)
    box.height = GetInt('height', root)
    box.filename = GetItem('filename', root) + '.JPEG'
    box.label = GetItem('name', root)

    xmin = float(box.xmin) / float(box.width)
    xmax = float(box.xmax) / float(box.width)
    ymin = float(box.ymin) / float(box.height)
    ymax = float(box.ymax) / float(box.height)

    # Some images contain bounding box annotations that
    # extend outside of the supplied image. See, e.g.
    # n03127925/n03127925_147.xml
    # Additionally, for some bounding boxes, the min > max
    # or the box is entirely outside of the image.
    min_x = min(xmin, xmax)
    max_x = max(xmin, xmax)
    box.xmin_scaled = min(max(min_x, 0.0), 1.0)
    box.xmax_scaled = min(max(max_x, 0.0), 1.0)

    min_y = min(ymin, ymax)
    max_y = max(ymin, ymax)
    box.ymin_scaled = min(max(min_y, 0.0), 1.0)
    box.ymax_scaled = min(max(max_y, 0.0), 1.0)

    boxes.append(box)

  return boxes

if __name__ == '__main__':
  if len(sys.argv) < 2 or len(sys.argv) > 3:
    print('Invalid usage\n'
          'usage: process_bounding_boxes.py <dir> [synsets-file]',
          file=sys.stderr)
    sys.exit(-1)

  xml_files = glob.glob(sys.argv[1] + '/*/*.xml')
  print('Identified %d XML files in %s' % (len(xml_files), sys.argv[1]),
        file=sys.stderr)

  if len(sys.argv) == 3:
    labels = set([l.strip() for l in open(sys.argv[2]).readlines()])
    print('Identified %d synset IDs in %s' % (len(labels), sys.argv[2]),
          file=sys.stderr)
  else:
    labels = None

  skipped_boxes = 0
  skipped_files = 0
  saved_boxes = 0
  saved_files = 0
  for file_index, one_file in enumerate(xml_files):
    # Example: <...>/n06470073/n00141669_6790.xml
    label = os.path.basename(os.path.dirname(one_file))

    # Determine if the annotation is from an ImageNet Challenge label.
    if labels is not None and label not in labels:
      skipped_files += 1
      continue

    bboxes = ProcessXMLAnnotation(one_file)
    assert bboxes is not None, 'No bounding boxes found in ' + one_file

    found_box = False
    for bbox in bboxes:
      if labels is not None:
        if bbox.label != label:
          
          if bbox.label in labels:
            skipped_boxes += 1
            continue

      # Guard against improperly specified boxes.
      if (bbox.xmin_scaled >= bbox.xmax_scaled or
          bbox.ymin_scaled >= bbox.ymax_scaled):
        skipped_boxes += 1
        continue

      # Note bbox.filename occasionally contains '%s' in the name. This is
      # data set noise that is fixed by just using the basename of the XML file.
      image_filename = os.path.splitext(os.path.basename(one_file))[0]
      print('%s.JPEG,%.4f,%.4f,%.4f,%.4f' %
            (image_filename,
             bbox.xmin_scaled, bbox.ymin_scaled,
             bbox.xmax_scaled, bbox.ymax_scaled))

      saved_boxes += 1
      found_box = True
    if found_box:
      saved_files += 1
    else:
      skipped_files += 1

    if not file_index % 5000:
      print('--> processed %d of %d XML files.' %
            (file_index + 1, len(xml_files)),
            file=sys.stderr)
      print('--> skipped %d boxes and %d XML files.' %
            (skipped_boxes, skipped_files), file=sys.stderr)

  print('Finished processing %d XML files.' % len(xml_files), file=sys.stderr)
  print('Skipped %d XML files not in ImageNet Challenge.' % skipped_files,
        file=sys.stderr)
  print('Skipped %d bounding boxes not in ImageNet Challenge.' % skipped_boxes,
        file=sys.stderr)
  print('Wrote %d bounding boxes from %d annotated images.' %
        (saved_boxes, saved_files),
        file=sys.stderr)
  print('Finished.', file=sys.stderr)
