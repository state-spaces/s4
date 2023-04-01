import argparse
import random
import shutil
import uuid

from natsort import natsorted
from pathlib import Path
from types import SimpleNamespace

rd = random.Random()
rd.seed(0)
uuid.uuid4 = lambda: uuid.UUID(int=rd.getrandbits(128))

class Experiment:
  
  def __init__(
    self,
    condition,
    input_dir,
    output_dir,
    url_templ,
    methods,
  ):
      self.condition = condition
      self.input_dir = Path(input_dir)
      self.output_dir = Path(output_dir)
      self.url_templ = url_templ
      self.methods = methods

      self._verify_input_dir()
      self._verify_input_filenames()
      self._shuffle_files()
      self.uids = self.create_uids()

  def _verify_input_dir(self):
    # Check that input_dir exists and contains folders for each method
    for method in self.methods:
      assert self.input_dir.joinpath(method).exists()

  def _verify_input_filenames(self):
    # Check that each method has the same number of files and identical filenames
    filenames = set([file.name for file in self.input_dir.joinpath(self.methods[0]).glob('*.wav')])
    for method in self.methods[1:]:
      files = set(self.input_dir.joinpath(method).glob('*.wav'))
      assert len(files) == len(filenames)
      for file in files:
        assert file.name in filenames, f'{file.name} is not in the set of filenames'

    self.filenames = list(natsorted(filenames))
    print("Found {} files".format(len(self.filenames)))

  def _shuffle_files(self):
    # Shuffle the filenames
    random.seed(42)
    random.shuffle(self.filenames)

  def create_uids(self):
    # Construct a table mapping each (method, filename) to a unique ID
    random.seed(42)
    uids = {}
    _uuids = set()
    for method in self.methods:
      for filename in self.filenames:
        # Generate a unique ID
        uid = uuid.uuid4().hex
        while uid in _uuids:
          uid = uuid.uuid4().hex
        _uuids.add(uid)

        uids[(method, filename)] = uid
    return uids

  def construct_batches(self, batch_size: int):
    assert batch_size > 0
    assert len(self.filenames) % batch_size == 0, 'batch_size must evenly divide the number of files'

    # Split the files into batches
    batches = [self.filenames[i:i+batch_size] for i in range(0, len(self.filenames), batch_size)]

    return batches

  def create_output_dir(self):
    # Create the output directory
    self.output_dir.mkdir(parents=True, exist_ok=True)

    # Create a subdirectory for the condition
    self.output_dir.joinpath(self.condition).mkdir(parents=True, exist_ok=False)
    
  def process_data(self, batch_size: int):
    # Set random seed
    random.seed(42)

    # Create output directory
    self.create_output_dir()

    # Construct batches
    # Each batch contains a fixed set of filenames, and includes those filenames for all methods
    batches = self.construct_batches(batch_size)

    # 1. Create a subdirectory for each batch
    # 2. For each method, copy the waveforms into the subdirectory after renaming them using self.uids
    # 3. Create a list of URLs for each batch
    urls_by_batch = {}
    for i, batch in enumerate(batches):
      urls = []

      # Create output directory
      batch_dir = self.output_dir.joinpath(self.condition).joinpath(str(i))
      batch_dir.mkdir(parents=True, exist_ok=False)

      # Copy files into the batch directory
      for method in self.methods:
        for filename in batch:
          src = self.input_dir.joinpath(method).joinpath(filename)
          dst = batch_dir.joinpath(f'{self.uids[(method, filename)]}.wav')
          shutil.copy(src, dst)

          urls.append(self.get_url(self.condition, str(i), self.uids[(method, filename)]))

      # Shuffle the URLs to randomize the order of the waveforms
      random.shuffle(urls)
      urls_by_batch[str(i)] = urls

    # Store `batches` to disk
    with open(self.output_dir.joinpath(self.condition).joinpath('batches.txt'), 'w') as f:
      for batch in batches:
        f.write(' '.join(batch) + '\n')

    # Store `urls_by_batch` to disk
    with open(self.output_dir.joinpath(self.condition).joinpath('urls.csv'), 'w') as f:
      urls = [','.join([f'recording_{i}_url' for i in range(len(self.methods) * batch_size)])] + [",".join(urls) for _, urls in urls_by_batch.items()]
      f.write('\n'.join(urls))
    
    # Store `urls_by_batch` by batch
    for batch, urls in urls_by_batch.items():
      with open(self.output_dir.joinpath(self.condition).joinpath(f'urls_{batch}.csv'), 'w') as f:
        urls = [','.join([f'recording_{i}_url' for i in range(len(self.methods) * batch_size)])] + [",".join(urls)]
        f.write('\n'.join(urls))

    # Store information in `self.uids` to disk
    with open(self.output_dir.joinpath(self.condition).joinpath('uids.txt'), 'w') as f:
      for (method, filename), uid in self.uids.items():
        f.write(f'{method} {filename} {uid}\n')

    return SimpleNamespace(
      batches=batches,
      urls_by_batch=urls_by_batch,
    )

  def get_url(self, condition, batch, uid):
    return self.url_templ.format(condition, batch, uid)

  def upload_data():
    pass


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--condition', type=str, required=True)
  parser.add_argument('--input_dir', type=str, required=True)
  parser.add_argument('--output_dir', type=str, required=True)
  parser.add_argument('--url_templ', type=str, default='https://storage.googleapis.com/mturk-experiments/{}/{}/{}.wav')
  parser.add_argument('--methods', nargs='+', required=True)
  parser.add_argument('--batch_size', type=int, default=1)
  args = parser.parse_args()

  experiment = Experiment(
    condition=args.condition,
    input_dir=args.input_dir,
    output_dir=args.output_dir,
    url_templ=args.url_templ,
    methods=args.methods,
  )

  experiment.process_data(batch_size=args.batch_size)
