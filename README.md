# Any-Quantile Probabilistic Forecasting of Short-Term Electricity Demand

## Citation
If you use this code in any context, please cite the following paper:
```
@misc{smyl2024anyquantile,
      title={Any-Quantile Probabilistic Forecasting of Short-Term Electricity Demand}, 
      author={Slawek Smyl and Boris N. Oreshkin and Paweł Pełka and Grzegorz Dudek},
      year={2024},
      eprint={2404.17451},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2404.17451}, 
}
```

## Create workspace and clone this repository
```
mkdir workspace
cd workspace
git clone git@github.com:boreshkinai/any-quantile.git 
```

## Build docker image and launch container
```
cd any-quantile
docker build -f Dockerfile -t any_quantile:$USER .

nvidia-docker run -p 8888:8888 -p 6000-6010:6000-6010 -v ~/workspace/any-quantile:/workspace/any-quantile -t -d --shm-size="16g" --name any_quantile_$USER any_quantile:$USER
```

## Enter inside docker container and train N-BEATS-AQUA model
```
docker exec -i -t any_quantile_$USER  /bin/bash
python run.py --config=config/nbeatsaq-mhlv.yaml
```
