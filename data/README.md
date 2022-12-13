# Data

Subfolders:
* raw -> symlinked
* instructions -> tracked
* models -> symlinked

## Raw Data Sources

* [RAV](https://zenodo.org/record/1188976#.Y5CRp-zMLDU): Download the speech data for Actor 01-24
* SWBD: Data under licence of LDC, located internally. Split info received via mail, saved in the download package.
  [Speaker Stats](https://isip.piconepress.com/projects/switchboard/doc/statistics/ws97_speaker_stats.text)
  are not provided by default, but saved in the internal dl package as well.
* [MOS](http://immortal.multicomp.cs.cmu.edu/raw_datasets/): Link is dead (as of Dec22). [Split info](https://github.com/A2Zadeh/CMU-MultimodalSDK/blob/master/mmsdk/mmdatasdk/dataset/standard_datasets/CMU_MOSEI/cmu_mosei_std_folds.py) is saved in a python file with  arrays.
* IEM: Data under license, located internally.


## Raw Data Assumptions

to properly begin the training processing analysis etcetera pipelines, we need postulate some assumptions about the
structure the data is saved in.

* The `raw` folder contains a subfolder each corpus named by the corpuses abbreviation.
* In these the audio files are situated.
  * The files are saved in a format that can be read by librosa.
* Next to the audio files is a single `manifest.csv` which contains the aggregated metadata of the specific corpus.
* the manifest should at the start only contain information that is encoded in the basic download package provided
  by the authors of the corpus. i.e. if the dl pacakge contains labels in a labels subfolder these
  should be incorporated, but if the splits are usually done a specific way, and this is not mentioned in the
  initial package then the splits need to be generated at preprocessing time.
* imported as `pandas.DataFrames` the manifests should be continously indexed. each row contains at least a `file_idx`.
  If a file contains multiple utterances the rows should also contain a `clip_idx` as well as `start` and `end`
  timestamps
* Categorical Variables should be in title_case.
* After preprocessing row is assumed to contain either a `Emotion` or `Sentiment` attribute or both. From that the
  labels are extracted.
* If and only if the utterances are directly associated with splits, the splits are given as extra files wich are called
  `train.txt`, `val.txt`,  `test.txt`. In there each row describes the `file_idx` of where each file belongs.

In the `scripts` folder parse scripts for each used corpus can be found. they serve as examples how to get the download
packages in the aforementioned state. They worked in Dec22.

### normalized Emotion Classes

| RAV | IEM | MOS |
|---|---|---|
| Happiness | Happiness | Happiness  |
| Sadness | Sadness | Sadness |
| Anger | Anger | Anger |
| Fear | Excitement\* | Fear |
| Surprise | Neutral | Surprise |
| Disgust | | |
| *No Emotion* | | *No Emotion* |

\* Excitement is often folded into Happiness. We do the same.

### normalized Sentiment Labels

| MOS | SWBD |
|---|---|
|Positive|Positive|
|Negative|Negative|
|Neutral|Neutral|


## Instructions

Preprocessing is controlled by `.yaml` instructions. These are structured with the following keys:

* `src`: the source corpus
* `steps`: the description of thge individual steps to alter the manifest.
  * `name`: the name of the step. must match a step from the `PreproFunc` enum.
  * `args`: arguments for this step
  *
* `label`: information for the label encoder-decoder (encodec).
  * `classes`: a list of possible classes
  * `NofN`: weather labels are exclusive or not. If true the encoder produces MHE labels
* `features`: information for the feature extractor
  * arguments for this depend on the feature extractor class and are given at instantiation time.