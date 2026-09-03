# Pet-ReID-IMAG
 The 3rd place solution to CVPR2022 Biometrics Workshop Pet Biometric Challenge
---- 

## Pawprint ID 本地工作台

Windows 下可以直接双击仓库根目录的启动器：

- `start-pet-reid.cmd`：CUDA ONNX 模式；
- `start-pet-reid-cpu.cmd`：纯 CPU ONNX 模式；
- `stop-pet-reid.cmd`：停止由启动器管理的三个服务。

两种模式都使用 `http://localhost:3000`，并共享同一个临时图库。启动器会依次检查模型、
Java 包、前端依赖与健康状态；运行日志和 PID 状态位于
`../../artifacts/workspace_logs/quick_start`。
管理员工具使用每次启动随机生成的密钥，密钥仅在服务运行期间保存在
`../../artifacts/workspace_logs/quick_start/admin-key.txt`，执行停止脚本后会自动删除。

命令行也可使用：

```powershell
.\scripts\pet-reid-stack.ps1 start -Provider cpu
.\scripts\pet-reid-stack.ps1 start -Provider cuda
.\scripts\pet-reid-stack.ps1 status
.\scripts\pet-reid-stack.ps1 stop
```

## Introduction
- :blush: We only trained one model (ResNeSt) with different scales (i.e., 224, 256, and 288), respectivel achieved 91.7% and 86.27% in phase A and B.
- :rocket: Traing time cost ~1.5 hour with a V100 16GB, so easy, no bells and whistles! 
- :eyes: Techical details are described in our [PDF](https://arxiv.org/pdf/2205.15934.pdf). 
- :point_right: The train/test data can be obtained from [百度云](https://pan.baidu.com/s/17tnCE8b-oSh8xGMHczPzqQ?pwd=imag), [Google drive](https://drive.google.com/drive/folders/1_7pdSRTvD_XdTu8z0MxrM9PDoEuX-tjf?usp=drive_link).
- :point_right: The weights can be obtained from [百度云](https://pan.baidu.com/s/17tnCE8b-oSh8xGMHczPzqQ?pwd=imag), [Google drive](https://drive.google.com/drive/folders/1_7pdSRTvD_XdTu8z0MxrM9PDoEuX-tjf?usp=drive_link).
- Click on the star  :star:, Thank you :heart:
## Requirements

* PyTorch  1.7.0+cu101
* torchvision  0.8.1+cu101 

### Prepare data

```
cd ./Pet-ReID-IMAG
mkidr data

# Download train_dir.zip  
unzip train_dir.zip  

# move train_dir  to ./pet_ReID-IMAG/data
````
## Training instruction
```
pip install -r  requirements.txt; cd fastreid/evaluation/rank_cylib; make all
```
```
bash train_resnest101_224.sh
bash train_resnest101_256.sh
bash train_resnest101_288.sh
bash train_resnest200_224.sh
```


## Test on Pet Challenge
```
bash predict.sh
```

## Acknowledgement
A large portion of code is borrowed from [fast-reid](https://github.com/JDAI-CV/fast-reid), many thanks  :+1: to their wonderful work!  

Thanks to my teammate Zijun Huang for his great support :blush:!
