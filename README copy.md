# <span style="color:#AB1E00">  Model Training Repository  </span>



## <span style="color:#A5EF00">  Setup  </span>
### <span style="color:#3D9F03"> Conda Installation
```bash
http://192.168.6.6:8081/computer-vision/conda
```

### <span style="color:#3D9F03"> Create Enviroment

Type the conda command below for creating enviroment.

```bash
conda create --name model_training python==3.8
```
After that press 'y' for continue creating enviroment.
When finished download some python packages type the command below for activate enviroment.

```bash
conda activate model_training
```
Upgrade setuptools and pip.

```bash
pip install --upgrade pip setuptools
```
### <span style="color:#3D9F03"> Install Requirements

```bash
pip install -r requirements.txt
```
### <span style="color:#3D9F03"> Change Ultralytics Settings

```bash
sed -i 's|datasets_dir: .*|datasets_dir: /home/$USER/$pwd(dir till the working repo)/datasets|' ~/.config/Ultralytics/settings.yaml
```

## <span style="color:#A5EF00">  Usage  </span>
### <span style="color:#3D9F03"> Run </span>
#### <span style="color:#3D9F03"> Native </span>
#### <span style="color:#3D9F03"> Train YOLO</span>
```bash
python app.py --env local --train --yolo
```
#### <span style="color:#3D9F03"> Train Classification</span>
```bash
python app.py --env local --train --classification
```

#### <span style="color:#3D9F03"> Inference YOLO</span>
```bash
python app.py --env local --infer --yolo
```
#### <span style="color:#3D9F03"> Inference Classification</span>
```bash
python app.py --env local --infer --classification
```

#### <span style="color:#3D9F03"> Model Architecture YOLO in Detail</span>
```bash
python app.py --env local --yolo --summary
```
#### <span style="color:#3D9F03"> Model Architecture Classification in Detail</span>
```bash
python app.py --env local --classification --summary
```

#### <span style="color:#3D9F03"> Board To Visualize </span>
```bash
tensorboard --logdir=runs --load_fast=false
```

#### <span style="color:#3D9F03"> Docker-Compose (not active)</span>
##### <span style="color:#B8DD21"> Debug </span>
```bash
docker-compose up --build
```
##### <span style="color:#B8DD21"> Prod </span>
```bash
docker-compose up --build --detach
```
