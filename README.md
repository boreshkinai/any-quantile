# any-quantile

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
