<h1 align='center'> Deep Bayesian Blind Color Deconvolution of Histological Images <br>(ICIP 2023)<br>
    [<a href="https://arxiv.org/abs/2005.08926">arXiv</a>] 
</h1>


<p align="center">
<img align="middle" src="./img/BCD_net_reduced_ICIP_lateral.png" width="1000" />
</p>


Histological images are often tainted with two or more stains to reveal their underlying structures and conditions. Blind Color
Deconvolution (BCD) techniques separate colors (stains) and structural information (concentrations), which is useful for the processing, data augmentation, and classification of such images. Classical BCD methods are not _amortized_: they rely on a complicated optimization procedure that has to be carried out on each image. In contrast, once they have been trained, Deep Neural Networks (DNNs) can be used in a fast, amortized manner on unseen inputs. Unfortunately, the lack of large databases of ground truth color and concentrations has limited the development of DNNs for BCD. In this work, we propose BCD-Net, the first Deep Variational Bayesian BCD Neural Network for stain separation and concentration estimation. BCD-Net is trained by maximizing the evidence lower bound of the observed images, which does not require the use of ground truth examples of stains and concentrations. Results obtained using two multicenter databases demonstrate the effectiveness of BCD-Net in the stain separation tasks, while drastically reducing the computation time compared to classical non-amortized methods.

----

### Requirements

See `requirements.txt` for a list of required packages. The experiments were performed with Python 3.9.13 and PyTorch 2.0.1.

### Replicating the experiments from the paper


### Citation
```bibtex
@article{,
    title={{D}eep {B}ayesian {B}lind {C}olor {D}econvolution of {H}istological {I}mages},
    author={},
    journal={},
    year={2023}
}
```