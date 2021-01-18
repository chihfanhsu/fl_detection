# Facial landmarks detection

Comparing different methods (losses) for the facial landmark detection. For a fair comparison, the backbone network of each method is designed with a similar structure.

# Tested method

1. Regression Method: predict the coordinate of each landmark directly.<br />
![Regression Method](https://github.com/chihfanhsu/fl_detection/blob/master/figs/regression.PNG?raw=true)
2. KL-divergence (Heatmap) Method: predict a distribution for each landmark<br />
![Heatmap Method](https://github.com/chihfanhsu/fl_detection/blob/master/figs/heatmap.PNG?raw=true)
2. Pixel-wise Classification Method (PWC): predict the class of each pixel belonged to a certain landmark or the background<br />
![PWC Method](https://github.com/chihfanhsu/fl_detection/blob/master/figs/pwc.PNG?raw=true)

3. A Hybrid Network: Combining the regression method and PWC method (two heterogeneous loss function)<br />
![PWC+disc Method](https://github.com/chihfanhsu/fl_detection/blob/master/figs/pwc%2Bdisc.PNG?raw=true)

4. Combining with a Discrimination Network: adding a discriminator to test the predicted result is a face-like-shape or not<br />
![PWC+disc Method](https://github.com/chihfanhsu/fl_detection/blob/master/figs/pwc%2Bdisc.PNG?raw=true)

# Training and Testing Sets
![Training&Testing Sets](https://github.com/chihfanhsu/fl_detection/blob/master/figs/testing%20set.PNG?raw=true)

# Results
![Qualitative](https://github.com/chihfanhsu/fl_detection/blob/master/figs/comparing.PNG?raw=true)

![Quantative](https://github.com/chihfanhsu/fl_detection/blob/master/figs/result.PNG?raw=true)


# Paper

[Paper Link](https://arxiv.org/abs/2005.08649)

@misc{hsu2020detailed,
      title={A Detailed Look At CNN-based Approaches In Facial Landmark Detection}, 
      author={Chih-Fan Hsu and Chia-Ching Lin and Ting-Yang Hung and Chin-Laung Lei and Kuan-Ta Chen},
      year={2020},
      eprint={2005.08649},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

## Download the Training and Testing Sets
[Raw images and pts](https://drive.google.com/file/d/1bCcnXII2Dc2dGstt8w_x3fE7SrwCWfVu/view?usp=sharing)<br/>
[Pickle](https://drive.google.com/file/d/1MkmGLtS_5g_LykovbRQg_JnnbZo6VAOS/view?usp=sharing)