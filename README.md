# A Prior Guided Wavelet-Spatial Dual Attention Transformer Framework for Heavy Rain Image Restoration (IEEE TMM 2024)
Ronghui Zhang, Jiongze Yu, Junzhou Chen, Guofa Li, Liang Lin, Danwei Wang

> **Abstract:** *Heavy rain significantly reduces image visibility, hindering tasks like autonomous driving and video surveillance. Many existing rain removal methods, while effective in light rain, falter under heavy rain due to their reliance on purely spatial features. Recognizing this challenge, we introduce the Wavelet-Spatial Dual Attention Transformer Framework (WSDformer). This innovative architecture adeptly captures both frequency and spatial characteristics, anchored by the wavelet-spatial dual attention (WSDA) mechanism. While the spatial attention zeroes in on intricate local details, the wavelet attention leverages wavelet decomposition to encompass diverse frequency information, augmenting the spatial representations. Furthermore, addressing the persistent issue of incomplete structural detail restoration, we integrate the PriorFormer Block (PFB). This unique module, underpinned by the Prior Fusion Attention (PFA), synergizes residual channel prior features with input features, thereby enhancing background structures and guiding precise rain feature extraction. To navigate the intrinsic constraints of U-shaped transformers, such as semantic discontinuities and subdued multi-scale interactions from skip connections, our Cross Interaction U-Shaped Transformer Network is introduced. This design empowers superior semantic layers to streamline the extraction of their lower-tier counterparts, optimizing network learning. Empirical analysis reveals our method's leading prowess across rainy image datasets and achieves state-of-the-art performance, with notable supremacy in heavy rainfall conditions. This superiority extends to diverse visual challenges and real-world rainy scenarios, affirming its broad applicability and robustness. The source code is available at https://github.com/Jiongze-Yu/WSDformer.*

## Method
![WSDfprmer](fig/network.png)
**WSDformer architecture.** (a) shows the overall architecture of **W**avelet-**S**patial **D**ual Trans**former** Network (WSDformer), where the green and yellow arrows represent down-sampling and up-sampling, respectively. (b) shows the internal structure of the WSDformer Block, and (c) shows the internal structure of the PriorFormer Block.

![WSDA](fig/wsda.png)
The structure of Wavelet-Spatial Dual Attention.

![Priorformer](fig/priorformer.png)
**The components of Prior Guidance Network.** (a) shows the structure of the Prior Extraction Module (PEM), which uses the residual channel as the prior information. (b) shows the structure of Prior Fusion Attention (PFA) in the PriorFormer Block (PFB).

## Datasets
<table>
<thead>
  <tr>
    <th>Dataset</th>
    <th>Outdoor-rain</th>
    <th>Rain200L</th>
    <th>Rain200H</th>
    <th>DID-Data</th>
    <th>DDN-Data</th>
    <th>SPA-Data</th>
    <th>Rain100L</th>
    <th>Rain100H</th>
    <th>Cityscapes-rain</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>Baidu Cloud</td>
    <td> <a href="">Download </a> </td>
    <td> <a href="">Download </a> </td>
    <td> <a href="">Download </a> </td>
    <td> <a href="">Download </a> </td>
    <td> <a href="">Download </a> </td>
    <td> <a href="">Download </a> </td>
    <td> <a href="">Download </a> </td>
    <td> <a href="">Download </a> </td>
    <td> <a href="">Download </a> </td>
  </tr>
</tbody>
</table>

## Training
Please download the corresponding datasets and put them in the folder `datasets`.  
To initiate the training of the neural network, execute the subsequent commands. Employ the Rain200H dataset as an illustrative instance.
```
python train.py --exp_name WSDformer_Rain200H --train_data_path ./datasets/Rain200H/train --test_data_path ./datasets/Rain200H/test
```
Run the script then you can find the generated experimental logs in the folder `training_log` and the trained weights in the folder `weights`.
Please note that when training SPA-Data, you have to change the counting method to iteration and set iter to 300,000.

## Testing
We refer to the [IDT](https://github.com/jiexiaou/IDT) test methodology. To evaluate the image with arbitrary size, we first split the image to overlapped 256*256 patches, and merge evaluated patches back to original resolution. 
Enter the following commands to test the network. Take the dataset Rain200H as an example.
```
python test_full_size.py --exp_name WSDformer_Rain200H --test_data_path ./datasets/Rain200H/test
```
Run the script then you can find the generated outputs in the folder `results`.

For the task of segmentation after de-raining in Cityscapes-rain, we refer to the experimental setup of [SGINet](https://github.com/OaDsis/SGINet), and used the [PSPNet50](https://github.com/hszhao/semseg) that was well trained on the Cityscapes dataset for the experiments. The experiments were tested using the integrated pspnet in [mmsegmentation](https://mmsegmentation.readthedocs.io/en/latest/). Where the configuration file used is `pspnet_r50-d8_512x1024_40k_cityscapes.py` and the training weights are `pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth`.

## Pre-trained Models
<table>
<thead>
  <tr>
    <th>Dataset</th>
    <th>Outdoor-rain</th>
    <th>Rain200L</th>
    <th>Rain200H</th>
    <th>DID-Data</th>
    <th>DDN-Data</th>
    <th>SPA-Data</th>
    <th>Rain100L</th>
    <th>Rain100H</th>
    <th>Cityscapes-rain</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>Baidu Cloud</td>
    <td> <a href="">Download </a> </td>
    <td> <a href="">Download </a> </td>
    <td> <a href="">Download </a> </td>
    <td> <a href="">Download </a> </td>
    <td> <a href="">Download </a> </td>
    <td> <a href="">Download </a> </td>
    <td> <a href="">Download </a> </td>
    <td> <a href="">Download </a> </td>
    <td> <a href="">Download </a> </td>
  </tr>
</tbody>
</table>

## Evaluation
See folder `evaluations`

Outdoor-rain/Rain200L/Rain200H/SPA-Data/Rain100L/Rain100H/Cityscapes-rain: [Matlab Code](https://github.com/Jiongze-Yu/WSDformer/blob/main/evaluations/Evaluation_1/evaluaterain.m)

DID-Data: [Matlab Code](https://github.com/Jiongze-Yu/WSDformer/blob/main/evaluations/Evaluation_2/statistic_DID.m)

DDN-Data: [Matlab Code](https://github.com/Jiongze-Yu/WSDformer/blob/main/evaluations/Evaluation_2/statistic_DID.m)

![tab1](fig/tab1.png)

![tab2](fig/tab2.png)

![tab3](fig/tab3.png)

![tab4](fig/tab4.png)

## Deraining Results
<table>
<thead>
  <tr>
    <th>Dataset</th>
    <th>Outdoor-rain</th>
    <th>Rain200L</th>
    <th>Rain200H</th>
    <th>DID-Data</th>
    <th>DDN-Data</th>
    <th>SPA-Data</th>
    <th>Rain100L</th>
    <th>Rain100H</th>
    <th>Cityscapes-rain</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>Baidu Cloud</td>
    <td> <a href="">Download </a> </td>
    <td> <a href="">Download </a> </td>
    <td> <a href="">Download </a> </td>
    <td> <a href="">Download </a> </td>
    <td> <a href="">Download </a> </td>
    <td> <a href="">Download </a> </td>
    <td> <a href="">Download </a> </td>
    <td> <a href="">Download </a> </td>
    <td> <a href="">Download </a> </td>
  </tr>
</tbody>
</table>
