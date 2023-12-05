#Getting relationship from Relation.py


This file has main function, where relationships between pair objects are generated.

Before running this file, download dataloader, fasterRCNN and lib folder, where required .py files are saved.

Finally run this line:

```python
python Relation.py -mode predcls -datasize large -data_path /data/AmitRoyChowdhury/Sayak/ag/  -model_path /data/AmitRoyChowdhury/Rinki/tempura_models/predcls/best_Mrecall_model.tar  -input_dir /data/AmitRoyChowdhury/Rinki/Activity_box_test -original_video_path /data/AmitRoyChowdhury/sayak/activity-net-captions/test_paths.txt  -output_dir /data/AmitRoyChowdhury/Rinki/Activity_test_relation -rel_mem_compute joint -rel_mem_weight_type simple -mem_fusion late -mem_feat_selection manual  -mem_feat_lambda 0.5  -rel_head gmm -obj_head linear -K 6 
```

where,
original_video_path is the text file where, paths for videos of particular dataset are saved,
input_dir is the directory where detected object details for each videos are saved,
data_path is the directory where action_genome annotations are saved


**** Name change *******************


Within each frame folder,
roi_features are saved as "npz_file.npz"

Within each video folder,
json files are saved as "Activity_BBox.json"

fmap files are saved as "npz_fmap.npz"
*********************************************

