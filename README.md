# Getting relationship between detected object pairs in frames of videos

This project aims to detect relationships like looking_at/sitting_on/in_front_of between two objects in frames of a video. So we need to have the list of detected objects and their coordinates in all frames of the videos. Also, we need the extracted ROI features of the objects saved in a .npz file. Some of our codes are taken from the repository https://github.com/sayaknag/unbiasedsgg?tab=readme-ov-file

# Contents

The file Relation.py has main function, where relationships between pair objects are generated.

Before running this file, download other folders in this repository , e.g., dataloader, fasterRCNN and lib folder, where required .py files are saved.

# Inference
Finally run this line:

```python
python Relation.py -mode predcls -datasize large -data_path abc  -model_path xyz  -input_dir MNP -original_video_path bcd.txt  -output_dir NMP -rel_mem_compute joint -rel_mem_weight_type simple -mem_fusion late -mem_feat_selection manual  -mem_feat_lambda 0.5  -rel_head gmm -obj_head linear -K 6 
```

where,
* original_video_path is the text file where, paths for videos of particular dataset are saved,
* input_dir is the directory where detected object details for each videos are saved,
* data_path is the directory where action_genome annotations are saved
* model_path is the directory where the predcls model is saved from https://drive.google.com/drive/folders/1m1xSUbqBELpogHRl_4J3ED7tlyp3ebv8
* output_dir is the directory where detected relationships will be saved




# organization of data
|--Video_1 \n
   |--"npz_fmap.npz" \n
   |--Activity_BBox.json \n
   |--frame_1 \n
     |-- "npz_file.npz" \n
   |--frame_2 \n
     |--"npz_file.npz"
