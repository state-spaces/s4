# SaShiMi

![SaShiMi](/assets/sashimi.png "SaShiMi Architecture")
> **It's Raw! Audio Generation with State-Space Models**\
> Karan Goel, Albert Gu, Chris Donahue, Christopher Ré\
> Paper: https://arxiv.org/abs/2202.09729

## Table of Contents
- [Standalone Implementation](#standalone-implementation)
- [SaShiMi+DiffWave](#diffwave)
- [Datasets](#datasets)
- [Model Training](#model-training)
- [Audio Generation](#audio-generation)
    - [Download Checkpoints](#download-checkpoints)
    - [Unconditional Generation](#unconditional-generation)
    - [Conditional Generation](#conditional-generation)
- [Automated Metrics](#automated-metrics)
    - [SC09 Classifier Training](#sc09-classifier-training)
        - [Install Dependencies](#install-dependencies)
        - [Download Dataset](#download-dataset)
        - [Train](#train)
    - [Downloads](#downloads)
    - [Calculating Automated Metrics](#calculating-automated-metrics)
        - [Dataset Metrics](#dataset-metrics)
        - [Metrics on Autoregressive Models (SaShiMi, SampleRNN, WaveNet)](#metrics-on-autoregressive-models-sashimi-samplernn-wavenet)
        - [Metrics on Non-Autoregressive Models (DiffWave variants, WaveGAN)](#metrics-on-non-autoregressive-models-diffwave-variants-wavegan)
- [Mean Opinion Scores (Amazon MTurk)](#mean-opinion-scores-amazon-mturk)
    - [Template for HITs](#template-for-hits)
    - [Download Artifacts](#download-artifacts)
    - [Music (YouTubeMix)](#music-youtubemix)
    - [Speech (SC09)](#speech-sc09)
    - [Posting HITs](#posting-hits)

Samples of SaShiMi and baseline audio can be found [online](https://hazyresearch.stanford.edu/sashimi-examples).

## Standalone Implementation
We provide a standalone PyTorch implementation of the SaShiMi architecture backbone in [sashimi.py](sashimi.py), which you can use in your own code. Note that you'll need to also copy over the standalone S4 layer implementation, which can be found at [[/models/s4/s4.py](/models/s4/)].
Note that our experiments do not use this standalone and instead use the modular model construction detailed in [[/src/models/README.md](/src/models/)], so this standalone is less tested; if running experiments from this codebase, it is recommended to use the normal training infrastructure ([README](/README.md#training)).

You can treat the SaShiMi module as a sequence-to-sequence map taking `(batch, seq, dim)` inputs to `(batch, seq, dim)` outputs i.e.
```python
sashimi = Sashimi().cuda() # refer to docstring for arguments
x = torch.randn(batch_size, seq_len, dim).cuda()
# Run forward
y = sashimi(x) # y.shape == x.shape
```

If you use SaShiMi for autoregressive generation, you can convert it to a recurrent model at inference time and then step it to generate samples one at a time. The following is a simple example that processes an input in recurrent mode instead of convolutional mode.
See the main [README](/README.md#generation) for the main generation script used for autoregressive generation.
```python
with torch.no_grad():
    sashimi.eval()
    sashimi.setup_rnn() # setup S4 layers in recurrent mode
    # alternately, use sashimi.setup_rnn('diagonal') for a speedup

    # Run recurrence
    ys = []
    state = sashimi.default_state(*x.shape[:1], device='cuda')
    for i in tqdm(range(seq_len)):
        y_, state = sashimi.step(x[:, i], state)
        ys.append(y_.detach().cpu())

    y = torch.stack(ys, dim=1) # y.shape == x.shape
    # y should be equal to sashimi(x)!
```

## DiffWave

The DiffWave and DiffWave+SaShiMi experiments used an alternative pipeline to handle diffusion logic, and is supported in another codebase located here: https://github.com/albertfgu/diffwave-sashimi

## Datasets
You can download the Beethoven, YouTubeMix and SC09 datasets from the following links on the Huggingface Hub. Details about the datasets can be found in the README files on the respective dataset pages.

- [Beethoven](https://huggingface.co/datasets/krandiash/beethoven)
- [YouTubeMix](https://huggingface.co/datasets/krandiash/youtubemix)
- [SC09](https://huggingface.co/datasets/krandiash/sc09)

For each dataset, you only need to download and unzip the` <dataset>.zip` file inside the `/data/` directory at the top-level of the repository.

Details about the training-validation-test splits used are also included in the README files on the dataset pages. If you reproduce our results, this splitting will be handled automatically by our training scripts. The specific data processing code that we use can be found at `/src/dataloaders/audio.py`, and dataset definitions for use with our training code are included in `/src/dataloaders/datasets.py`. The dataset configs can be found at `/configs/datasets/`.

### SC09 alternative
There is an easier way to set up SC09 through the standard SpeechCommands dataset available in this repository, without duplicating data:
```
python -m train wandb=null experiment=sc/s4-sc  # Start SC experiment, which auto-downloads data; can kill experiment once started
cd data
mkdir sc09
ln -s $PWD/SpeechCommands/{zero,one,two,three,four,five,six,seven,eight,nine} sc09/  # Symlink relevant SC subsets
cp SpeechCommands/{testing_list.txt,validation_list.txt} sc09
```

## Model Training
SaShiMi models rely on the same training framework as S4 (see the [README](/README.md) for details). To reproduce our results or train new SaShiMi models, you can use the following commands:
```bash
# Train SaShiMi models on YouTubeMix, Beethoven and SC09
python -m train experiment=audio/sashimi-youtubemix wandb=null
python -m train experiment=audio/sashimi-beethoven wandb=null
python -m train experiment=audio/sashimi-sc09 wandb=null
```
If you encounter GPU OOM errors on either Beethoven or YouTubeMix, we recommend reducing the sequence length used for training by setting `dataset.sample_len` to a lower value e.g. `dataset.sample_len=120000`. For SC09, we recommend reducing batch size if GPU memory is an issue, by setting `loader.batch_size` to a lower value.

We also include implementations of SampleRNN and WaveNet models, which can be trained easily using the following commands:
```bash
# Train SampleRNN models on YouTubeMix, Beethoven and SC09
python -m train experiment=audio/samplernn-youtubemix wandb=null
python -m train experiment=audio/samplernn-beethoven wandb=null
python -m train experiment=audio/samplernn-sc09 wandb=null

# Train WaveNet models on YouTubeMix, Beethoven and SC09
python -m train experiment=audio/wavenet-youtubemix wandb=null
python -m train experiment=audio/wavenet-beethoven wandb=null
python -m train experiment=audio/wavenet-sc09 wandb=null
```
Audio generation models are generally slow to train, e.g. YouTubeMix SaShiMi models take up to a week to train on a single V100 GPU.

<!--
> _Note on model performance_: due to limited compute resources, our results involved a best-effort reproduction of the baselines, and relatively limited hyperparameter tuning for SaShiMi. We expect that aggressive hyperparameter tuning should lead to improved results for all models. If you're interested in pushing these models to the limits and have compute $ or GPUs, please reach out to us!
-->

## Audio Generation

To generate audio, use the `/generation.py` script.
More instructions can be found in the main [README](/README.md#generation).

### Download Checkpoints
We provide checkpoints for SaShiMi, SampleRNN and WaveNet on YouTubeMix and SC09 on the [Huggingface Hub](https://huggingface.co/krandiash/sashimi-release). The checkpoint files are named `checkpoints/<model>_<dataset>.pt` and are provided for use with our generation script.

### Unconditional Generation
First, put the checkpoints you downloaded at `/checkpoints/`.

Then, run the following command to generate audio
```bash
python -m generate experiment=audio/<model>-<dataset> checkpoint_path=<path/to/model.ckpt> l_sample=<sample_len_in_steps> load_data=false
```

For example, to generate 32 unconditional samples of 1 second 16kHz audio from the SaShiMi model on YouTubeMix, run the following command:
```bash
python -m generate experiment=audio/sashimi-youtubemix checkpoint_path=checkpoints/sashimi_youtubemix.pt n_samples=32 l_sample=16000 load_data=false
```
The generated `.wav` files will be saved to `sashimi/samples/`. You can generate audio for all models and datasets in a similar way.

> _Note 1 (log-likehoods)_: the saved `.wav` files will be ordered by their (exact) log-likelihoods, with the first sample having the lowest log-likelihood. This is possible since all models considered here have exact, tractable likelihoods. We found that samples near the bottom (i.e. those with lowest likelihoods) or close to the top (i.e. those with highest likelihoods) tend to have worse quality. Concretely, throwing out the bottom 40% and top 5% of samples is a simple heuristic for improving average sample quality, and performance on automated generation metrics (see Appendix C.3.1 of our paper for details).

> _Note 2 (runaway noise)_: samples generated by autoregressive models can often have "runaway noise", where a sample suddenly degenerates into pure noise. Intuitively, this happens when the model finds itself in an unseen state that it struggles to generalize to, which is often the case for extremely long sequence generation. We found that SaShiMi also suffers from this problem when generating long sequences, and fixing this issue for autoregressive generation is an interesting research direction.

### Conditional Generation
You can also generate conditional samples, e.g. to generate 32 samples conditioned on 0.5 seconds of audio from the SaShiMi model on YouTubeMix, run the following command:
```bash
python -m generate experiment=audio/sashimi-youtubemix checkpoint_path=checkpoints/sashimi_youtubemix.pt n_samples=8 n_reps=4 l_sample=16000 l_prefix=8000
```
The `prefix` flag specifies the number of steps to condition on. The script selects the first `n_samples` examples of the specified `split` (defaults to `val`) of the dataset. `n_reps` specifies how many generated samples will condition on a prefix from a single example (i.e. the total number of generated samples is `n_samples x n_reps`).

Note that it is necessary to pass the `load_data` flag and you will need to make sure the datasets are available in the `data/` directory when running conditional generation.

## Automated Metrics
We provide a standalone implementations of automated evaluation metrics for evaluating the quality of generated samples on the SC09 dataset in `metrics.py`. Following [Kong et al. (2021)](https://arxiv.org/pdf/2009.09761.pdf), we implemented the Frechet Inception Distance (FID), Inception Score (IS), Modified Inception Score (mIS), AM Score (AM) and the number of statistically different bins score (NDB). Details about the metrics and the procedure followed by us can be found in Appendix C.3 of the paper.

### SC09 Classifier Training
We use a modified version of the training/testing script provided by the [pytorch-speech-commands](https://github.com/tugstugi/pytorch-speech-commands) repository, which we include under `sc09_classifier`. Following [Kong et al. (2021)](https://arxiv.org/pdf/2009.09761.pdf), we used a ResNeXt model trained on SC09 spectrograms.

This classifier has two purposes:
1. To calculate the automated metrics, each SC09 audio clip must be converted into a feature vector.
2. Following [Donahue et al. (2019)](https://arxiv.org/pdf/1802.04208.pdf), we use classifier confidence as a proxy for the quality and intelligibility of the generated audio. Roughly, we sample a large number of samples from each model, and then select the top samples (as ranked by classifier confidence) per class (as assigned by the classifier). These are then used in MOS experiments.

#### Install Dependencies
Requirements are included in the `/requirements.txt` file for reference. We recommend running your `torch` and `torchvision` install using whatever best practices you follow before installing other requirements.
```bash
pip install -r requirements.txt
```
> This code is provided as-is, so depending on your `torch` version, you may need to make slight tweaks to the code to get it running. It's been tested with `torch` version `1.9.0+cu102`.

#### Download Dataset
For convenience, we recommend redownloading the Speech Commands dataset for classifier training using the commands below. Downloading and extraction should take a few minutes.
```bash
cd sashimi/sc09_classifier/
bash download_speech_commands_dataset.sh
```

#### Train
To train the classifier, run the following command:
```bash
mkdir checkpoints/
python train_speech_commands.py --batch-size 96 --learning-rate 1e-2
```
Training is fast and should take only a few hours on a T4 GPU. The best model checkpoint should be saved under `sc09_classifier/` with a leading timestamp. Note that we provide these instructions for completeness, and you should be able to reuse our classifier checkpoint directly (see `Downloads` next).

### Downloads
To reproduce our evaluation results on the SC09 dataset, we provide sample directories, a classifier checkpoint and a preprocessed cache of classifier outputs.

1. _Samples:_ We provide all samples generated by all models on the [Huggingface Hub](https://huggingface.co/krandiash/sashimi-release) under `samples/`. Download and unzip all the sample directories in `samples/`.
2. _Classifier:_ You can use our SC09 classifier checkpoint rather than training your own. We provide this for convenience on the [Huggingface Hub](https://huggingface.co/krandiash/sashimi-release) at `sc09_classifier/resnext.pth`. This model achieves `98.08%` accuracy on the SC09 test set. Download this checkpoint and place it in `sc09_classifier/`.
3. _Cache:_ We provide a cache of classifier outputs that are used to speed up automated evaluation, as well as used in MTurk experiments for gathering mean opinion scores. You can find these on the [Huggingface Hub](https://huggingface.co/krandiash/sashimi-release) at `sc09_classifier/cache`. Download and place the `cache` directory under `sc09_classifier/`.

At the end of this your directory structure should look something like this:
```bash
state-spaces/
├── sashimi/
│   ├── samples/
│   │   ├── sc09/
│   │   ├── youtubemix/
│   ├── sc09_classifier/
│   │   ├── resnext.pth
│   │   ├── cache/
...
```

### Calculating Automated Metrics
We provide instructions for calculating the automated SC09 metrics next.

#### Dataset Metrics
To generate the automated metrics for the dataset, run the following command from the `sashimi/sc09_classifier` folder:
```bash
python test_speech_commands.py resnext.pth
```
If you didn't correctly place the `cache` folder under `sc09_classifier`, this will be a little slow to run the first time, as it caches features and predictions (`train_probs.npy`, `test_probs.npy`, `train_activations.npy`, `test_activations.npy`) for the train and test sets under `sc09_classifier/cache/`. Subsequent runs reuse this and are much faster.

#### Metrics on Autoregressive Models (SaShiMi, SampleRNN, WaveNet)
For autoregressive models, we follow a threshold tuning procedure that is outlined in `Appendix C.3` of our paper. We generated `10240` samples for each model, using `5120` to tune thresholds for rejecting samples with the lowest and highest likelihoods, and evaluating the metrics on the `5120` samples that are held out. This is all taken care of automatically by the `test_speech_commands.py` script (with the `--threshold` flag passed in).

```bash
# SaShiMi (4.1M parameters)
python test_speech_commands.py resnext.pth --threshold --sample-dir ../samples/sc09/sashimi/

# SampleRNN (35.0M parameters)
python test_speech_commands.py resnext.pth --threshold --sample-dir ../samples/sc09/samplernn/

# WaveNet (4.2M parameters)
python test_speech_commands.py resnext.pth --threshold --sample-dir ../samples/sc09/wavenet/
```
> _Important:_ the commands above assume that samples inside the `sample-dir` directory are sorted by log-likelihoods (in increasing order), since the `--threshold` flag is being passed. Our autoregressive generation script does this automatically, but if you generated samples manually through a separate script, you should sort them by log-likelihoods before running the above commands. If you cannot sort the samples by log-likelihoods, you can simply omit the `--threshold` flag.

For example, in order to generate the SaShiMi samples above:
```bash
python -m generate experiment=audio/sashimi-sc09 checkpoint_path=sashimi/sashimi_sc09.pt save_dir=sashimi/samples/sc09/sashimi load_data=false n_samples=10240

python -m generate experiment=audio/wavenet-sc09 checkpoint_path=sashimi/wavenet_sc09.pt save_dir=sashimi/samples/sc09/wavenet load_data=false n_samples=10240
```

**Update (Sept. 2022 - V3)**: A slightly larger SaShiMi model (6.8M parameters) with improved hyperparameters was trained and a checkpoint is provided. It is slightly too large to generate 10240 samples at once, so the `n_batch` flag can be used:
```
python -m generate experiment=audio/sashimi-sc09 checkpoint_path=sashimi/sashimi_sc09_unet.pt save_dir=sashimi/samples/sc09/sashimi load_data=false n_samples=10240 n_batch=5120
```

The results are better than the original model reported in the paper.
|                | NLL   | FID  | IS   | mIS   | AM   |
| ---            | ---   | ---  | ---  | ---   | ---  |
| Sashimi (orig) | 1.891 | 1.99 | 4.12 | 24.57 | 0.90 |
| Sashimi (new)  | 1.873 | 1.99 | 5.13 | 42.57 | 0.74 |


#### Metrics on Non-Autoregressive Models (DiffWave variants, WaveGAN)
For DiffWave models and WaveGAN (which don't provide exact likelihoods), we simply calculate metrics directly on `2048` samples.
```bash
# DiffWave with WaveNet backbone (24.1M parameters), trained for 500K steps
python test_speech_commands.py --sample-dir ../samples/sc09/2048-diffwave-500k/ resnext.pth
# DiffWave with WaveNet backbone (24.1M parameters), trained for 1M steps
python test_speech_commands.py --sample-dir ../samples/sc09/2048-diffwave-1m/ resnext.pth
# Small DiffWave with WaveNet backbone (6.8M parameters), trained for 500K steps
python test_speech_commands.py --sample-dir ../samples/sc09/2048-diffwave-small-500k/ resnext.pth

# DiffWave with bidirectional SaShiMi backbone (23.0M parameters), trained for 500K steps
python test_speech_commands.py --sample-dir ../samples/sc09/2048-sashimi-diffwave-500k/ resnext.pth
# DiffWave with bidirectional SaShiMi backbone (23.0M parameters), trained for 800K steps
python test_speech_commands.py --sample-dir ../samples/sc09/2048-sashimi-diffwave-800k/ resnext.pth
# Small DiffWave with bidirectional SaShiMi backbone (7.5M parameters), trained for 500K steps
python test_speech_commands.py --sample-dir ../samples/sc09/2048-sashimi-diffwave-small-500k/ resnext.pth

# Small DiffWave with unidirectional SaShiMi backbone (7.1M parameters), trained for 500K steps
python test_speech_commands.py --sample-dir ../samples/sc09/2048-sashimi-diffwave-small-uni-500k/ resnext.pth

# WaveGAN model (19.1M parameters)
python test_speech_commands.py --sample-dir ../samples/sc09/2048-wavegan/ resnext.pth
```

## Mean Opinion Scores (Amazon MTurk)

Details and instructions on how we ran our MOS studies on Amazon Mechnical Turk can be found in `Appendix C.4` of our paper. We strongly recommend referring to that section while going through the instructions below. We provide both code and samples that you can use to compare against SaShiMi, as well as run your own MOS studies.

### Template for HITs
We provide templates for generating HTML required for constructing HITs on Amazon MTurk at `sashimi/mturk/templates/`. Our templates are largely derived and repurposed from the templates provided by [Neekhara et al. (2019)](https://arxiv.org/pdf/1904.07944.pdf). The template in `template_music.py` can be used to generate HITs for evaluating any music generation model, while the template in `template_speech.py` can be used for evaluating models on SC09 (and could likely be repurposed for evaluating other types of speech generation models). Each HTML template corresponds to what is shown in a single HIT to a crowdworker.

### Download Artifacts
The SaShiMi release page for the [Huggingface Hub](https://huggingface.co/krandiash/sashimi-release) contains the final set of `.wav` files that we use in our MOS MTurk experiments at `mturk/sc09` and `mturk/youtubemix`. You should download and unzip these files and place them at `mturk/sc09/` and `mturk/youtubemix/` respectively.

### Music (YouTubeMix)
If you want to run MOS studies on YouTubeMix (or for any music generation dataset and models), the instructions below should help you get setup using our code. The steps that you will follow are:
1. Generating samples from each model that is being compared. We use 16 second samples for our YouTubeMix MOS study.
2. Selecting a few samples from each model to use in the MOS study. In our work, we use a simple filtering criteria to select `30` samples for each model that you can find in Appendix C.4 of our paper.
3. Randomizing and generating the batches of samples that will constitute each HIT in the MOS study using `turk_create_batch.py`.
4. Generating an HTML template using `template_music.py`.
5. Uploading the final batches to public storage.
6. Posting the HITs to Amazon MTurk (see [Posting HITs](#posting-hits)).
7. Downloading results and calculating the MOS scores using our `MTurk YouTubeMix MOS` notebook at `mturk/mos`.

For Steps 1 and 2, you can use your own generation scripts or refer to our generation code (see [Audio Generation](#audio-generation)).

For Step 3, we provide the `turk_create_batch.py` script that takes a directory of samples for a collection of methods and organizes them as a batch of HITs. Note that this script only organizes the data that will be contained in each HIT and does not actually post the HITs to MTurk.

Assuming you have the following directory structure (make sure you followed [Download Artifacts](#download-artifacts) above):
```bash
state-spaces/
├── sashimi/
│   ├── mturk/
│   │   ├── turk_create_batch.py
│   │   └── youtubemix/
│   │       ├── youtubemix-unconditional-16s-exp-1
│   │       |   ├── sashimi-2
│   │       |   |   ├── 0.wav
│   │       |   |   ├── 1.wav
│   │       |   |   ├── ...
│   │       |   |   ├── 29.wav
│   │       |   ├── sashimi-8
│   │       |   |   ├── 0.wav
│   │       |   |   ├── 1.wav
│   │       |   |   ├── ...
│   │       |   |   ├── 29.wav
│   │       |   ├── samplernn-3
│   │       |   |   ├── 0.wav
│   │       |   |   ├── 1.wav
│   │       |   |   ├── ...
│   │       |   |   ├── 29.wav
│   │       |   ├── wavenet-1024
│   │       |   |   ├── 0.wav
│   │       |   |   ├── 1.wav
│   │       |   |   ├── ...
│   │       |   |   ├── 29.wav
│   │       |   └── test
│   │       |   |   ├── 0.wav
│   │       |   |   ├── 1.wav
│   │       |   |   ├── ...
│   │       |   |   ├── 29.wav

# For your own MTurk music experiment, you should follow this structure
...   ├── <condition> # folder for your MTurk music experiment
...   |   ├── <method-1> # with your method folders
...   |   |   ├── 0.wav # containing samples named 0.wav, 1.wav, ... for each method
...   |   |   ├── 1.wav
...   |   |   ├── ...
...   |   ├── <method-2>
...   |   ├── <method-3>
...   |   ├── ...
```

You can then run the following command to organize the data for a batch of HITs (run `python turk_create_batch.py --help` for more details):
```bash
# Reproduce our MOS experiment on YouTubeMix exactly
python turk_create_batch.py \
--condition youtubemix-unconditional-16s-exp-1 \
--input_dir youtubemix/youtubemix-unconditional-16s-exp-1/ \
--output_dir final/ \
--methods wavenet-1024 sashimi-2 sashimi-8 samplernn-3 test \
--batch_size 1

# Run your own
python turk_create_batch.py \
--condition <condition> \
--input_dir path/to/<condition>/ \
--output_dir final/ \
--methods <method-1> <method-2> ... <method-k> \
--batch_size 1
```

The result of this command will generate a folder inside `final/` with the name of the condition. For example, the resulting directory when reproducing our MOS experiment on YouTubeMix will look like:
```bash
final/
├── youtubemix-unconditional-16s-exp-1
│   ├── 0/ # batch, with one wav file per method
│   │   ├── 176ea1b164264cd51ea45cd69371a71f.wav # some uids
│   │   ├── 21e150949efee464da90f534a23d4c9d.wav # ...
│   │   ├── 3405095c8a5006c1ec188efbd080e66e.wav
│   │   ├── 41a93f90dc8215271da3b7e2cad6e514.wav
│   │   ├── e3e70682c2094cac629f6fbed82c07cd.wav
│   ├── 1/
│   ├── 2/
│   ├── ...
│   ├── 29/ # 30 total batches
|   ├── batches.txt # mapping from the batch index to the wav file taken from each method
|   ├── uids.txt # mapping from (method, wav file) to a unique ID
|   ├── urls.csv # list of file URLs for each batch: you'll upload this to MTurk later for creating the HITs
|   ├── urls_0.csv
|   ├── ...
│   └── urls_29.csv
```
The `urls.csv` file will be important for posting the HITs to MTurk.

> _Note about URLs:_ You will need to upload the folder generated inside `final/` above (e.g. `youtubemix-unconditional-16s-exp-1`) to a server where it can be accessed publicly through a URL. We used a public Google Cloud Bucket for this purpose. Depending on your upload location, you will need to change the `--url_templ` argument passed to `turk_create_batch.py` (see the default argument we use inside that file to get an idea of how to change it). This will then change the corresponding URLs in the `urls.csv` file.

If you run an MOS study on YouTubeMix in particular and want to compare to SaShiMi, we recommend reusing the samples we provided above, and selecting `30` samples for your method using the process outlined in the Appendix of our paper. In particular, note that five `.wav` files in `test/` are gold standard examples that we use for quality control on the HITs (the code for this filtering is provided in the `MTurk YouTubeMix MOS` notebook at `mturk/mos`).

For Step 4, the template in `template_music.py` can be used to generate HITs for evaluating any music generation model. Each HIT will consist of a single sample from each model being compared, along with a single dataset sample that will help calibrate the raters' responses. We follow [Dieleman et al. (2019)](https://arxiv.org/pdf/1806.10474.pdf) and collect ratings on the fidelity and musicality of the generated samples.

To generate HTML for a HIT that should contain `n_samples` samples, run the following command:
```bash
python template_music.py <n_samples>
# here `n_samples` should be the number of models being compared + 1 for the dataset
# e.g. we compared two SaShiMi variants to SampleRNN and WaveNet, so we set `n_samples` to 5
```

Steps 5 and 6 are detailed in [Posting HITs](#posting-hits).

Finally for Step 7, to calculate final scores, you can use the `MTurk YouTubeMix MOS` notebook provided in the `mos/` folder. Note that you will need to put the results from MTurk (which can be downloaded from MTurk for your batch of HITs and is typically named `Batch_<batch_index>_batch_results.csv`) in the `mos/` folder. Run the notebook to generate final results. Please be sure to double-check the names of the gold-standard files used in the notebook, to make sure that the worker filtering process is carried out correctly.


### Speech (SC09)
If you want to run MOS studies on SC09 (or for a speech generation dataset and model), the instructions below should help you get setup using our code. These instructions focus mainly on SC09, and you can pick and choose what you might need if you're working with another speech dataset. The steps that you will follow are:
1. Generating samples from each model that is being compared. We generated `2048` samples for each model.
2. Training an SC09 classifier. We use a ResNeXt model as discussed in [SC09 Classifier Training](#sc09-classifier-training).
3. Generating predictions for all models and saving them using the `test_speech_commands.py` script.
4. Using these predictions, selecting the top-50 (or any other number) most confident samples per class for each model using the `prepare_sc09.py` script.
5. Randomizing and generating the batches of samples that will constitute each HIT in the MOS study using `turk_create_batch.py`.
6. Generating an HTML template using `template_speech.py`.
7. Uploading the final batches to public storage.
8. Posting the HITs to Amazon MTurk (see [Posting HITs](#posting-hits)).
9. Downloading results and calculating the MOS scores using our `MTurk SC09 MOS` notebook at `mturk/mos`.

For Step 1, you can use your own generation scripts or refer to our generation code (see [Audio Generation](#audio-generation)).

Step 2 can be completed by referring to the [SC09 Classifier Training](#sc09-classifier-training) section or reusing our provided ResNeXt model checkpoint.

For Step 3, we first output predictions for samples generated by each model. Note that we use the `2048-` prefix for the sample directories used for MOS studies (you should already have downloaded and unzipped these sample directories from the [Huggingface Hub](https://huggingface.co/krandiash/sashimi-release) at `sashimi/samples/sc09/`).

To generate and save predictions for the sample directory corresponding to each model, run the following command using the `test_speech_commands.py` script from the `sc09_classifier/` folder:
```bash
python test_speech_commands.py --sample-dir ../samples/sc09/2048-<model> --save-probs resnext.pth
# this should save a file named `2048-<model>-resnext-probs.npy` at `sc09_classifier/cache/`
```
You should not need to do this for our results, since we provide predictions for all models in the `cache` directory for convenience.

In Step 4, given the `2048` samples generated by each model, we select the top-50 most confident samples for each class using the classifier outputs from Step 3 i.e. the `2048-<model>-resnext-probs.npy` files generated by `test_speech_commands.py`. We provide the `mturk/prepare_sc09.py` script for this.

```bash
python prepare_sc09.py --methods <method-1> <method-2> ... <method-k>

# To reproduce our MOS experiment on SC09, we use the following command:
python prepare_sc09.py --methods diffwave-500k diffwave-1m diffwave-small-500k samplernn-3 sashimi-8-glu sashimi-diffwave-small-500k sashimi-diffwave-500k sashimi-diffwave-800k sashimi-diffwave-small-uni-500k test wavenet-1024 wavegan
```
The `prepare_sc09.py` script takes additional arguments for `cache_dir`, `sample_dir` and `target_dir` to customize paths. By default, it will look for the `2048-<model>-resnext-probs.npy` files in `sc09_classifier/cache/` (`cache_dir`), the samples in `samples/sc09/` (`sample_dir`) and the output directory `mturk/sc09/sc09-unconditional-exp-confident-repro/` (`target_dir`).

As a convenience, we provide the output of this step on the [Huggingface Hub](https://huggingface.co/krandiash/sashimi-release) under the `mturk/sc09` folder (called `sc09-unconditional-exp-confident`). This corresponds to the directory of samples we used for SC09 MOS evaluation. Either download and place this inside `mturk/sc09/` before proceeding to the commands below, OR make sure you've followed the previous steps to generate samples for each method using the `prepare_sc09.py` script.

For Step 5, we then provide the `turk_create_batch.py` script that takes a directory of samples for a collection of methods and organizes them as a batch of HITs. Note that this script only organizes the data that will be contained in each HIT and does not actually post the HITs to MTurk.

You can run the following command to organize the data for a batch of HITs for each method:
```bash
# Reproduce our MOS experiment on SC09 exactly
python turk_create_batch.py \
--condition sc09-unconditional-exp-confident-test \
--input_dir sc09/sc09-unconditional-exp-confident/ \
--output_dir final/ \
--methods test \
--batch_size 10

python turk_create_batch.py \
--condition sc09-unconditional-exp-confident-wavenet-1024 \
--input_dir sc09/sc09-unconditional-exp-confident/ \
--output_dir final/ \
--methods wavenet-1024 \
--batch_size 10

# Do this for all methods under sc09/sc09-unconditional-exp-confident/
python turk_create_batch.py \
--condition sc09-unconditional-exp-confident-<method-name> \
--input_dir sc09/sc09-unconditional-exp-confident/ \
--output_dir final/ \
--methods <method-name> \
--batch_size 10
```
The result of this command will generate a folder inside `final/` with the name of the condition. You can run this command on your methods in an identical manner.


In Step 6, the template in `template_speech.py` can be used to generate HITs for evaluating models on SC09. Each HIT will consist of multiple samples from only a single model. We chose to not mix samples from different models into a single HIT to reduce the complexity of running the study, as we evaluated a large number of models.

To generate HTML for a HIT that will contain `n_samples` samples, run the following command:
```bash
python template_speech.py <n_samples>
# we set `n_samples` to be 10
```

Steps 7 and 8 are detailed in [Posting HITs](#posting-hits).

Finally for Step 9, to calculate final scores, you can use the `MTurk SC09 MOS` notebook provided in the `mos/` folder. Note that you will need to put the results from MTurk (which can be downloaded from MTurk for your batch of HITs and is typically named `Batch_<batch_index>_batch_results.csv`) in the `mos/` folder. Run the notebook to generate final results. Please be sure to double-check the paths and details in the notebook to ensure correctness.


### Posting HITs
To post HITs, the steps are:
- Upload the `final/` samples for each model to a public cloud bucket.
- Login to your MTurk account and create a project. We recommend doing this on the Requester Sandbox first.
    - Go to New Project > Audio Naturalness > Create Project
    - On the Edit Project page, add a suitable title, description, and keywords. Refer to `Appendix C.4` of our paper for details on the payment and qualifications we used.
    - On the Design Layout page, paste in the HTML that you generated using the `template_<music/speech>.py` scripts.
    - Check the Preview to make sure everything looks good.
- Your Project should now show up on the "Create Batch with an Existing Project" page.
- Click "Publish Batch" on the project you just created. Upload the `urls.csv` file generated as a result of running the `turk_create_batch.py` script. Please make sure that you can access the audio files at the URLs in the `urls.csv` file (MTurk Sandbox testing should surface any issues).
- Once you're ready, post the HITs.

To access results, you can go to the "Manage" page and you should see "Batches in progress" populated with the HITs you posted. You should be able to download a CSV file of the results for each batch. You can plug this into the Jupyter Notebooks we include under `mturk/mos/` to get final results.

> _Note:_ for SC09 HITs, we strongly recommend posting HITs for all models at the same time to reduce the possibility of scoring discrepancies due to different populations of workers.
