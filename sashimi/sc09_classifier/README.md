Convolutional neural networks for [Google speech commands data set](https://research.googleblog.com/2017/08/launching-speech-commands-dataset.html)
with [PyTorch](http://pytorch.org/).

# General
We, [xuyuan](https://github.com/xuyuan) and [tugstugi](https://github.com/tugstugi), have participated
in the Kaggle competition [TensorFlow Speech Recognition Challenge](https://www.kaggle.com/c/tensorflow-speech-recognition-challenge)
and reached the 10-th place. This repository contains a simplified and cleaned up version of our team's code.

# Features
* `1x32x32` mel-spectrogram as network input
* single network implementation both for CIFAR10 and Google speech commands data sets
* faster audio data augmentation on STFT
* Kaggle private LB scores evaluated on 150.000+ audio files

# Results
Due to time limit of the competition, we have trained most of the nets with `sgd` using `ReduceLROnPlateau` for 70 epochs.
For the training parameters and dependencies, see [TRAINING.md](TRAINING.md). Earlier stopping the train process will sometimes produce a better score in Kaggle.

<table><tbody>
<th valign="bottom"><sup><sub>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Model&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</sub></sup></th>
<th valign="bottom"><sup><sub>CIFAR10<br/>test set<br/>accuracy</sub></sup></th>
<th valign="bottom"><sup><sub>Speech Commands<br/>test set<br/>accuracy</sub></sup></th>
<th valign="bottom"><sup><sub>Speech Commands<br/>test set<br/>accuracy with crop</sub></sup></th>
<th valign="bottom"><sup><sub>Speech Commands<br/>Kaggle private LB<br/>score</sub></sup></th>
<th valign="bottom"><sup><sub>Speech Commands<br/>Kaggle private LB<br/>score with crop</sub></sup></th>
<th valign="bottom"><sup><sub>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Remarks&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</sub></sup></th>

<tr>
<td align="left"><sup><sub>VGG19 BN</sub></sup></td>
<td align="center"><sup><sub>93.56%</sub></sup></td>
<td align="center"><sup><sub>97.337235%</sub></sup></td>
<td align="center"><sup><sub>97.527432%</sub></sup></td>
<td align="center"><sup><sub>0.87454</sub></sup></td>
<td align="center"><sup><sub>0.88030</sub></sup></td>
<td align="left"><sup><sub></sub></sup></td>
</tr>

<tr>
<td align="left"><sup><sub>ResNet32</sub></sup></td>
<td align="center"><sup><sub>-</sub></sup></td>
<td align="center"><sup><sub>96.181419%</sub></sup></td>
<td align="center"><sup><sub>96.196050%</sub></sup></td>
<td align="center"><sup><sub>0.87078</sub></sup></td>
<td align="center"><sup><sub>0.87419</sub></sup></td>
<td align="left"><sup><sub></sub></sup></td>
</tr>

<tr>
<td align="left"><sup><sub>WRN-28-10</sub></sup></td>
<td align="center"><sup><sub>-</sub></sup></td>
<td align="center"><sup><sub>97.937089%</sub></sup></td>
<td align="center"><sup><sub>97.922458%</sub></sup></td>
<td align="center"><sup><sub>0.88546</sub></sup></td>
<td align="center"><sup><sub>0.88699</sub></sup></td>
<td align="left"><sup><sub></sub></sup></td>
</tr>

<tr>
<td align="left"><sup><sub>WRN-28-10-dropout</sub></sup></td>
<td align="center"><sup><sub>96.22%</sub></sup></td>
<td align="center"><sup><sub>97.702999%</sub></sup></td>
<td align="center"><sup><sub>97.717630%</sub></sup></td>
<td align="center"><sup><sub>0.89580</sub></sup></td>
<td align="center"><sup><sub>0.89568</sub></sup></td>
<td align="left"><sup><sub></sub></sup></td>
</tr>

<tr>
<td align="left"><sup><sub>WRN-52-10</sub></sup></td>
<td align="center"><sup><sub>-</sub></sup></td>
<td align="center"><sup><sub>98.039503%</sub></sup></td>
<td align="center"><sup><sub>97.980980%</sub></sup></td>
<td align="center"><sup><sub>0.88159</sub></sup></td>
<td align="center"><sup><sub>0.88323</sub></sup></td>
<td align="left"><sup><sub>another trained model has 97.52%/<b>0.89322</b></sub></sup></td>
</tr>

<tr>
<td align="left"><sup><sub>ResNext29 8x64</sub></sup></td>
<td align="center"><sup><sub>-</sub></sup></td>
<td align="center"><sup><sub>97.190929%</sub></sup></td>
<td align="center"><sup><sub>97.161668%</sub></sup></td>
<td align="center"><sup><sub>0.89533</sub></sup></td>
<td align="center"><sup><sub><b>0.89733</b></sub></sup></td>
<td align="left"><sup><sub>our best model during competition</sub></sup></td>
</tr>

<tr>
<td align="left"><sup><sub>DPN92</sub></sup></td>
<td align="center"><sup><sub>-</sub></sup></td>
<td align="center"><sup><sub>97.190929%</sub></sup></td>
<td align="center"><sup><sub>97.249451%</sub></sup></td>
<td align="center"><sup><sub>0.89075</sub></sup></td>
<td align="center"><sup><sub>0.89286</sub></sup></td>
<td align="left"><sup><sub></sub></sup></td>
</tr>

<tr>
<td align="left"><sup><sub>DenseNet-BC (L=100, k=12)</sub></sup></td>
<td align="center"><sup><sub>95.52%</sub></sup></td>
<td align="center"><sup><sub>97.161668%</sub></sup></td>
<td align="center"><sup><sub>97.147037%</sub></sup></td>
<td align="center"><sup><sub>0.88946</sub></sup></td>
<td align="center"><sup><sub>0.89134</sub></sup></td>
<td align="left"><sup><sub></sub></sup></td>
</tr>

<tr>
<td align="left"><sup><sub>DenseNet-BC (L=190, k=40)</sub></sup></td>
<td align="center"><sup><sub>-</sub></sup></td>
<td align="center"><sup><sub>97.117776%</sub></sup></td>
<td align="center"><sup><sub>97.147037%</sub></sup></td>
<td align="center"><sup><sub>0.89369</sub></sup></td>
<td align="center"><sup><sub>0.89521</sub></sup></td>
<td align="left"><sup><sub></sub></sup></td>
</tr>

</tbody></table>

# Results with Mixup

After the competition, some of the networks were retrained using [mixup: Beyond Empirical Risk Minimization](https://arxiv.org/abs/1710.09412) by Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin and David Lopez-Paz.

<table><tbody>
<th valign="bottom"><sup><sub>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Model&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</sub></sup></th>
<th valign="bottom"><sup><sub>CIFAR10<br/>test set<br/>accuracy</sub></sup></th>
<th valign="bottom"><sup><sub>Speech Commands<br/>test set<br/>accuracy</sub></sup></th>
<th valign="bottom"><sup><sub>Speech Commands<br/>test set<br/>accuracy with crop</sub></sup></th>
<th valign="bottom"><sup><sub>Speech Commands<br/>Kaggle private LB<br/>score</sub></sup></th>
<th valign="bottom"><sup><sub>Speech Commands<br/>Kaggle private LB<br/>score with crop</sub></sup></th>
<th valign="bottom"><sup><sub>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Remarks&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</sub></sup></th>

<tr>
<td align="left"><sup><sub>VGG19 BN</sub></sup></td>
<td align="center"><sup><sub>-</sub></sup></td>
<td align="center"><sup><sub>97.483541%</sub></sup></td>
<td align="center"><sup><sub>97.542063%</sub></sup></td>
<td align="center"><sup><sub>0.89521</sub></sup></td>
<td align="center"><sup><sub>0.89839</sub></sup></td>
<td align="left"><sup><sub></sub></sup></td>
</tr>

<tr>
<td align="left"><sup><sub>WRN-52-10</sub></sup></td>
<td align="center"><sup><sub>-</sub></sup></td>
<td align="center"><sup><sub>97.454279%</sub></sup></td>
<td align="center"><sup><sub>97.498171%</sub></sup></td>
<td align="center"><sup><sub>0.90273</sub></sup></td>
<td align="center"><sup><sub><b>0.90355</b></sub></sup></td>
<td align="left"><sup><sub>same score as the 16-th place in Kaggle</sub></sup></td>
</tr>

</tbody></table>
