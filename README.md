# U-Evolve


Urban vibrancy describes the prosperity, diversity, and accessibility of urban areas, which is vital to a city's socio-economic development and sustainability. While many efforts have been made for statically measuring and evaluating urban vibrancy, there are few studies on the evolutionary process of urban vibrancy, yet we know little about the relationship between urban vibrancy evolution and sophisticated spatiotemporal dynamics. In this paper, we make use of multi-sourced urban data to develop a data-driven framework, U-Evolve, to investigate urban vibrancy evolution. 
Specifically, we first exploit the spatiotemporal characteristics of urban areas to create multi-view time-dependent graphs.
Then, we analyze the contextual features and graph patterns of multi-view time-dependent graphs in terms of informing future urban vibrancy variations.
Our analysis validates the informativeness of multi-view time-dependent graphs for characterizing and informing future urban vibrancy evolution.
After that, we construct a feature based model to forecast future urban vibrancy evolution and quantify each feature's importance.
Moreover, to further enhance the forecasting effectiveness, we propose a graph learning based model to capture spatiotemporal autocorrelation of urban areas based on multi-view time-dependent graphs in an end-to-end manner.
Finally, extensive experiments on two metropolises, Beijing and Shanghai, demonstrate the effectiveness of our forecasting models.
The U-Evolve framework has also been deployed in the production environment to deliver real-world urban development and planning insights for various cities in China.

A pytorch implementation for the paper:
Characterizing and Forecasting Urban Vibrancy Evolution: A Multi-View Graph Mining Perspective in ACM Transactions on Knowledge Discovery from Data (TKDD).

Due to privacy and commerical concerns of the company, we are unable to publish the data.


## License and Citation
If you find our code or paper useful, please cite our paper:
```bibtex
@article{uevolve2022tkdd,
 author =  {Liu, Hao and Guo, Qingyu and Zhu, Hengshu and Fu, Yanjie and Zhuang, Fuzhen and Ma, Xiaojuan and Xiong, Hui},
 title = {Characterizing and Forecasting Urban Vibrancy Evolution: A Multi-View Graph Mining Perspective},
 journal = {ACM Transactions on Knowledge Discovery from Data (TKDD)},
 year = {2022}
 }
```

## Contact
Please contact [@QingyuGuo](qguoag@connect.ust.hk) and [@HaoLiu](liuh@ust.hk) for questions, comments and reporting bugs.