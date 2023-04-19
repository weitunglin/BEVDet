## Environment Setup

1. create environment (docker)
```shell=
cd docker && docker build -t .
docker run --gpus all --shm-size=8g -it --rm -v /home/allenlin/workspace/BEVDet:/home/allenlin/workspace/BEVDet -v /mnt/sda/allenlin/dataset/:/mmdetection3d/data mmdetection3d
```

2. install packages
```shell=
cd /home/allenlin/workspace/BEVDet
pip install -e .
pip install numpy==1.23
```

## Run BEVDet on ITRI Dataset

1. train the model
```shell=
export name=bevdet-r50-cbgs-front
export config=configs/bevdet/${name}.py
export work_dir=work_dir/${name}/
./tools/dist_train.sh $config 4 --work-dir $work_dir
```

2. eval the model
```shell=
export name=bevdet-r50-cbgs-front
export config=configs/bevdet/${name}.py
export work_dir=work_dir/${name}/
export checkpoint=${work_dir}/latest.pth
export out=${work_dir}/${name}-eval-result.pkl

# test & eval
./tools/dist_test.sh $config $checkpoint 4 --eval bbox --out $out

# visualization
./tools/dist_test.sh $config $checkpoint 4 --format-only --eval-options jsonfile_prefix=$work_dir --resume-result $out
python tools/analysis_tools/itri_vis.py $work_dir/pts_bbox/results_nusc.json --root_path /mmdetection3d/data/nuscenes/ --canva-size 600 --scale-factor 1 --vis-thred 0.4 --vis-frames 1000 --save_path $work_dir
```


## Run BEVDet on nuScenes

1.  setup configs
```shell=
export config=configs/bevdet/bevdet4d-r50-cbgs.py
export checkpoint=bevdet-r50-4d-cbgs.pth
export savepath=vis/
export work_dir=trt_dir/
```

2. run inference
```shell=
# multi-gpus testing
./tools/dist_test.sh $config $checkpoint 4 --format-only --eval-options jsonfile_prefix=$savepath --fuse-conv-bn

# single-gpu testing
python tools/test.py $config $checkpoint --format-only --eval-options jsonfile_prefix=$savepath --fuse-conv-bn
```

3. generate visualization result
```shell=
# render `without` gt
python tools/analysis_tools/vis.py $work_dir/pts_bbox/results_nusc.json --root_path /mmdetection3d/data/nuscenes/ --save_path $work_dir

# render with gt
python tools/analysis_tools/vis.py $work_dir/pts_bbox/results_nusc.json --root_path /mmdetection3d/data/nuscenes/ --draw-gt --video-prefix vis_with_gt --save_path $work_dir

# render itri dataset
python tools/analysis_tools/itri_vis.py $work_dir/pts_bbox/results_nusc.json --root_path /mmdetection3d/data/nuscenes/ --canva-size 600 --scale-factor 1 --vis-thred 0.4 --vis-frames 3500 --save_path $work_dir
```
