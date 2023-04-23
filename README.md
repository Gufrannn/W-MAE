# W-MAE: Pre-trained weather model with masked autoencoder for multi-variable weather forecasting
Created by [Xin Man](https://github.com/Gufrannn), [Chenghong Zhang](), [Changyu Li](), [Jie Shao](https://cfm.uestc.edu.cn/~shaojie/). ([Paper](https://arxiv.org/pdf/2210.02199))

[PyTorch](https://github.com/Gufrannn/W-MAE/tree/main/PyTorch) & [MindSpore](https://github.com/Gufrannn/W-MAE/tree/main/MindSpore) Implementations

This is the official PyTorch implementation of W-MAE presented by paper [W-MAE: Pre-trained weather model with masked autoencoder for multi-variable weather forecasting](https://arxiv.org/pdf/2210.02199). The codes are used to reproduce experimental results of the proposed W-MAE framework in the paper.
This repository currently supports finetuning W-MAE on multi-variable forecasting and precipitation forecasting tasks.
The best pre-trained checkpoint on the [fifth-generation ECMWF Reanalysis (ERA5) dataset](https://rmets.onlinelibrary.wiley.com/doi/full/10.1002/qj.3803) are released.

![Overview of W-MAE](https://github.com/Gufrannn/W-MAE/blob/main/imgs/Showcase.jpg)

Abstract: Weather forecasting is a long-standing computational challenge with direct societal and
economic impacts. This task involves a large amount of continuous data collection and
exhibits rich spatiotemporal dependencies over long periods, making it highly suitable
for deep learning models. In this paper, we apply pre-training techniques to weather
forecasting and propose W-MAE, a Weather model with Masked AutoEncoder
pretraining for multi-variable weather forecasting. W-MAE is pre-trained in a selfsupervised
manner to reconstruct spatial correlations within meteorological variables.
On the temporal scale, we fine-tune the pretrained W-MAE to predict the future states
of meteorological variables, thereby modeling the temporal dependencies present in
weather data. We pre-train W-MAE using the fifth-generation ECMWF Reanalysis
(ERA5) data, with samples selected every six hours and using only two years of data.
Under the same training data conditions, we compare W-MAE with FourCastNet, and
W-MAE outperforms FourCastNet in precipitation forecasting. In the setting where the
training data is far less than that of FourCastNet, our model still performs much better
in precipitation prediction (0.80 vs. 0.98). Additionally, experiments show that our
model has a stable and significant advantage in shortto-medium-range forecasting
(i.e., forecasting time ranges from 6 hours to one week), and the longer the prediction
time, the more evident the performance advantage of W-MAE, further proving its
robustness.
